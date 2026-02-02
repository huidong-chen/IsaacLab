# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image

# Set environment variables for OVRTX
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
# Set LD_PRELOAD if needed (libcarb.so)
libcarb_path = Path.home() / "dev/kit.0/kit/_build/linux-x86_64/release/libcarb.so"
if libcarb_path.exists() and "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = str(libcarb_path)

# Add ovrtx Python bindings to path
ovrtx_bindings_path = Path("/home/ncournia/dev/kit.0/rendering/ovrtx/public/bindings/python")
if str(ovrtx_bindings_path) not in sys.path:
    sys.path.insert(0, str(ovrtx_bindings_path))

# Set library path hint before importing ovrtx
from ovrtx._src import bindings
bindings.OVRTX_LIBRARY_PATH_HINT = "/home/ncournia/dev/kit.0/rendering/_build/linux-x86_64/release"

from ovrtx import Renderer, RendererConfig

from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .renderer import RendererBase


@wp.kernel
def _create_camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),  # type: ignore
    orientations: wp.array(dtype=wp.quatf),  # type: ignore
    transforms: wp.array(dtype=wp.mat44d),  # type: ignore
):
    """Kernel to create camera transforms from positions and orientations.

    Args:
        positions: Array of camera positions, shape (num_cameras,)
        orientations: Array of camera orientations, shape (num_cameras,)
        transforms: Output array of camera transforms, shape (num_cameras,)
    """
    i = wp.tid()
    # Convert warp quaternion to rotation matrix and combine with translation
    pos = positions[i]
    quat = orientations[i]
    
    # Quaternion to rotation matrix (3x3)
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    #qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    
    # Row 0
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    
    # Row 1
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    
    # Row 2
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    
    # Build 4x4 homogeneous transform matrix
    # IMPORTANT: Matrix is stored in COLUMN-MAJOR order for OVRTX
    # So we transpose the rotation part: columns become rows
    _0 = wp.float64(0.0)
    _1 = wp.float64(1.0)
    # Note: Type issues with warp vec3 indexing are expected
    transforms[i] = wp.mat44d(  # type: ignore
        wp.float64(r00), wp.float64(r10), wp.float64(r20), _0,
        wp.float64(r01), wp.float64(r11), wp.float64(r21), _0,
        wp.float64(r02), wp.float64(r12), wp.float64(r22), _0,
        wp.float64(float(pos[0])), wp.float64(float(pos[1])), wp.float64(float(pos[2])), _1
    )


@wp.kernel
def _extract_tile_from_tiled_buffer_kernel(
    tiled_buffer: wp.array(dtype=wp.uint8, ndim=3),  # type: ignore (tiled_height, tiled_width, 4)
    tile_buffer: wp.array(dtype=wp.uint8, ndim=3),  # type: ignore (height, width, 4)
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
):
    """Extract a single tile from a tiled buffer.
    
    Args:
        tiled_buffer: Input tiled buffer, shape (tiled_height, tiled_width, 4)
        tile_buffer: Output buffer for single tile, shape (tile_height, tile_width, 4)
        tile_x: Tile position in x (horizontal)
        tile_y: Tile position in y (vertical)
        tile_width: Width of each tile
        tile_height: Height of each tile
    """
    y, x = wp.tid()
    
    # Calculate source position in tiled buffer
    src_x = tile_x * tile_width + x
    src_y = tile_y * tile_height + y
    
    # Copy RGBA channels
    tile_buffer[y, x, 0] = tiled_buffer[src_y, src_x, 0]
    tile_buffer[y, x, 1] = tiled_buffer[src_y, src_x, 1]
    tile_buffer[y, x, 2] = tiled_buffer[src_y, src_x, 2]
    tile_buffer[y, x, 3] = tiled_buffer[src_y, src_x, 3]


@wp.kernel
def _sync_newton_transforms_kernel(
    ovrtx_transforms: wp.array(dtype=wp.mat44d),  # type: ignore
    newton_body_indices: wp.array(dtype=wp.int32),  # type: ignore
    newton_body_q: wp.array(dtype=wp.transformf),  # type: ignore
):
    """Kernel to sync Newton physics transforms to OVRTX render transforms.
    
    Converts Newton's wp.transformf (position + quaternion) to OVRTX's wp.mat44d
    (4x4 column-major matrix) for each object in the scene.
    
    Args:
        ovrtx_transforms: Output array of OVRTX transforms, shape (num_objects,)
        newton_body_indices: Newton body indices for each object, shape (num_objects,)
        newton_body_q: Newton body transforms from state.body_q, shape (num_bodies,)
    """
    i = wp.tid()
    body_idx = newton_body_indices[i]
    transform = newton_body_q[body_idx]
    
    # Use warp's built-in conversion and transpose for column-major format
    ovrtx_transforms[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(transform)))


class OVRTXRenderer(RendererBase):
    """OVRTX Renderer implementation using the ovrtx library.
    
    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    _renderer: Renderer | None = None
    _usd_handles: list | None = None
    _camera_binding = None
    _object_binding = None  # Binding for scene objects (robot, manipulated objects, etc.)
    _object_newton_indices: wp.array | None = None  # Newton body indices for objects
    _render_product_paths: list[str] = []
    _initialized_scene = False
    _frame_counter: int = 0  # Track frame number for image filenames

    def __init__(self, cfg: OVRTXRendererCfg):
        super().__init__(cfg)
        self._usd_handles = []
        self._render_product_paths = []
        self._frame_counter = 0
        
        # Calculate tiled dimensions (same as Newton renderer)
        self._num_tiles_per_side = math.ceil(math.sqrt(self._num_envs))
        self._tiled_width = self._num_tiles_per_side * self._width
        self._tiled_height = self._num_tiles_per_side * self._height

    def initialize(self, usd_scene_path: str | None = None):
        """Initialize the OVRTX renderer.
        
        Args:
            usd_scene_path: Optional path to USD scene to load as root layer.
                           If provided, cameras will be injected into this file.
                           If not provided, cameras are created as the root layer.
        """
        print("Creating OVRTX renderer...")
        OVRTX_CONFIG = RendererConfig(
            startup_options={
                "crashreporter/dumpDir": "/tmp",
                # WAR to avoid startup crash due to unsafe FoundationUtils getStringBuffer on log/file which ovrtx doesn't set
                "log/file": "/tmp/ovrtx_renderer.log",
            }
        )
        self._renderer = Renderer(OVRTX_CONFIG)
        assert self._renderer, "Renderer should be valid after creation"
        print("OVRTX renderer created successfully!")

        # Initialize output buffers
        self._initialize_output()
        
        # If a USD scene is provided, inject cameras into it
        if usd_scene_path is not None:
            print(f"Injecting cameras into USD scene: {usd_scene_path}")
            combined_usd_path = self._inject_cameras_into_usd(usd_scene_path)
            handle = self._renderer.add_usd(combined_usd_path, path_prefix=None)
            self._usd_handles.append(handle)
            print(f"   ✓ Combined scene loaded (handle: {handle})")
            self._initialized_scene = True
            
            # Create binding for camera transforms
            camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(self._num_envs)]
            
            print(f"\n[DEBUG] OVRTX Camera Binding Setup:")
            print(f"  Total cameras: {self._num_envs}")
            print(f"  Camera paths: {camera_paths}")
            print(f"  Binding to attribute: omni:fabric:worldMatrix")
            
            self._camera_binding = self._renderer.bind_attribute(
                prim_paths=camera_paths,
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4",
                prim_mode="must_exist",
            )
            
            if self._camera_binding is not None:
                print(f"  ✓ Camera binding created successfully")
            else:
                print(f"  ✗ WARNING: Camera binding is None!")
            
            # Setup object bindings for Newton physics sync
            self._setup_object_bindings()
        else:
            # Setup cameras as root layer
            #self._setup_scene(as_root_layer=True)
            pass
    
    def _inject_cameras_into_usd(self, usd_scene_path: str) -> str:
        """Inject camera and render product definitions into an existing USD file.
        
        Args:
            usd_scene_path: Path to the USD scene file
            
        Returns:
            Path to the combined USD file with cameras injected
        """
        import tempfile
        
        # Read the original USD
        with open(usd_scene_path, 'r') as f:
            original_usd = f.read()
        
        # Generate camera USD content (as a top-level Render scope)
        camera_parts = []
        camera_parts.append('\ndef Scope "Render"\n{\n')
        
        # Collect all camera paths
        camera_paths = [f"/World/envs/env_{env_idx}/Camera" for env_idx in range(self._num_envs)]
        
        # Create a SINGLE RenderProduct that references all cameras
        render_product_name = "RenderProduct"
        render_product_path = f"/Render/{render_product_name}"
        self._render_product_paths.append(render_product_path)
        
        # Build the camera relationship list: rel camera = [<path1>, <path2>, ...]
        camera_rel_list = ", ".join([f"<{path}>" for path in camera_paths])
        
        # Calculate tiled resolution: each tile is width x height, arranged in a grid
        print(f"\n[DEBUG] OvRTX Tiled Resolution:")
        print(f"  Individual camera resolution: {self._width} x {self._height}")
        print(f"  Number of environments: {self._num_envs}")
        print(f"  Tiles per side: {self._num_tiles_per_side}")
        print(f"  Total tiled resolution: {self._tiled_width} x {self._tiled_height}")
        
        camera_parts.append(f'''
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{camera_rel_list}]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "RealTimePathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = </Render/Vars/LdrColor>
        uniform int2 resolution = ({self._tiled_width}, {self._tiled_height})
    }}
''')
        
        # Add shared RenderVar
        camera_parts.append('''
    def "Vars"
    {
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
''')
        
        camera_parts.append('}\n')
        camera_content = ''.join(camera_parts)
        
        # Simply append the Render scope to the end of the file
        # This is safe since USD files are declarative
        combined_usd = original_usd.rstrip() + '\n\n' + camera_content
        
        # Save to temp file
        Path("/tmp/ovrtx_test").mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False, dir='/tmp/ovrtx_test') as f:
            f.write(combined_usd)
            temp_path = f.name
        
        print(f"   Created combined USD: {temp_path}")
        return temp_path
    
    def _setup_object_bindings(self):
        """Setup OVRTX bindings for scene objects to sync with Newton physics.
        
        This creates bindings for all dynamic objects (robot bodies, manipulated objects)
        that need to be updated each frame from Newton's physics state.
        """
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            
            newton_model = NewtonManager.get_model()
            if newton_model is None:
                print("[OVRTX] Newton model not available, skipping object bindings")
                return
            
            # Get all body paths from Newton
            # Filter out static objects (plane, lights) and cameras
            all_body_paths = newton_model.body_key
            
            # Filter to only dynamic objects in envs
            # Typically: /World/envs/env_X/Robot/..., /World/envs/env_X/object, etc.
            object_paths = []
            newton_indices = []
            
            for idx, path in enumerate(all_body_paths):
                # Include objects in /World/envs/ but exclude cameras and ground plane
                if "/World/envs/" in path and "Camera" not in path and "GroundPlane" not in path:
                    object_paths.append(path)
                    newton_indices.append(idx)
            
            if len(object_paths) == 0:
                print("[OVRTX] No dynamic objects found for binding")
                return
            
            print(f"\n[DEBUG] OVRTX Object Binding Setup:")
            print(f"  Total dynamic objects: {len(object_paths)}")
            print(f"  Example paths: {object_paths[:5]}")
            
            # Create OVRTX binding for all objects at once
            self._object_binding = self._renderer.bind_attribute(
                prim_paths=object_paths,
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4",
                prim_mode="must_exist",
            )
            
            if self._object_binding is not None:
                print(f"  ✓ Object binding created successfully")
                # Store Newton body indices for later lookup
                self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device="cuda:0")
            else:
                print(f"  ✗ WARNING: Object binding is None!")
                
        except ImportError:
            print("[OVRTX] Newton not available, skipping object bindings")
        except Exception as e:
            print(f"[OVRTX] Error setting up object bindings: {e}")
    
    def add_usd_scene(self, usd_file_path: str, path_prefix: str | None = None):
        """Add a USD scene file to the renderer.
        
        This allows loading geometry and scene content into the renderer.
        
        Args:
            usd_file_path: Path to the USD file to load
            path_prefix: Optional path prefix for the USD content
            
        Returns:
            USD handle that can be used to remove the scene later
        """
        if self._renderer is None:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")
        
        print(f"Loading USD scene: {usd_file_path}")
        handle = self._renderer.add_usd(usd_file_path, path_prefix)
        
        if self._usd_handles is not None:
            self._usd_handles.append(handle)
        
        print(f"USD scene loaded successfully (handle: {handle})")
        return handle

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        self._data_types = ["rgba", "rgb", "depth"]

        # Create output buffers on GPU
        # RGBA buffer: (num_envs, height, width, 4) of uint8
        self._output_data_buffers["rgba"] = wp.zeros(
            (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device="cuda:0"
        )
        # Create RGB view that references the same underlying array as RGBA, but only first 3 channels
        self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]
        # Depth buffer: (num_envs, height, width, 1) of float32
        self._output_data_buffers["depth"] = wp.zeros(
            (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device="cuda:0"
        )

###    def _setup_scene(self, as_root_layer: bool = True):
###        """Set up the USD scene with cameras and render products.
###        
###        This creates a USD scene with camera definitions and their corresponding
###        RenderProduct prims according to the OVRTX requirements.
###        
###        Args:
###            as_root_layer: If True, creates cameras as root layer with 'def Scope "Render"'.
###                          If False, creates cameras as sublayer with 'over "Render"'.
###                          Use False when a USD scene has already been loaded as root layer.
###        """
###        if self._initialized_scene:
###            return
###            
###        print("Setting up OVRTX scene...")
###        
###        # Create a USD layer with cameras and render products
###        usda_parts = []
###        
###        if as_root_layer:
###            # Creating as root layer: must set defaultPrim and use 'def'
###            usda_parts.append('#usda 1.0\n')
###            usda_parts.append('(\n    defaultPrim = "Render"\n)\n\n')
###            usda_parts.append('def Scope "Render" {\n')
###        else:
###            # Creating as sublayer: NO header, use 'over' to extend existing scene
###            # The header would make this a root layer!
###            usda_parts.append('over "Render" {\n')
###        
###        # Create cameras and render products for each environment
###        for env_idx in range(self._num_envs):
###            camera_name = f"Camera_{env_idx}"
###            render_product_name = f"RenderProduct_{env_idx}"
###            camera_path = f"/Render/{camera_name}"
###            render_product_path = f"/Render/{render_product_name}"
###            
###            # Store render product path for later use
###            if env_idx == 1: # XXX
###                self._render_product_paths.append(render_product_path)
###            
###            # Camera definition with RTX API schemas
###            usda_parts.append(f'''
###    def Camera "{camera_name}" (
###        prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]
###    ) {{
###        float focalLength = 18.0
###        float horizontalAperture = 20.955
###        float verticalAperture = 15.2908
###        token projection = "perspective"
###        float2 clippingRange = (1, 10000000)
###        bool omni:rtx:autoExposure:enabled = 1
###        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
###        uniform token[] xformOpOrder = ["xformOp:transform"]
###    }}
###''')
###            
###            # RenderProduct definition with RTX settings
###            usda_parts.append(f'''
###    def RenderProduct "{render_product_name}" (
###        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
###    ) {{
###        rel camera = <{camera_path}>
###        token omni:rtx:background:source:type = "domeLight"
###        token omni:rtx:rendermode = "RealTimePathTracing"
###        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
###        rel orderedVars = </Render/Vars/LdrColor>
###        uniform int2 resolution = ({self._width}, {self._height})
###    }}
###''')
###        
###        # Add shared RenderVar if it doesn't exist
###        usda_parts.append('''
###    def "Vars"
###    {
###        def RenderVar "LdrColor"
###        {
###            uniform string sourceName = "LdrColor"
###        }
###    }
###''')
###        
###        usda_parts.append('}\n')
###        usda_content = ''.join(usda_parts)
###        
###        # Add the USD to the renderer
###        if self._renderer is not None:
###            if as_root_layer:
###                # Save to temp file and use add_usd for root layer
###                import tempfile
###                with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False) as f:
###                    f.write(usda_content)
###                    temp_path = f.name
###                handle = self._renderer.add_usd(temp_path, path_prefix=None)
###                # Clean up temp file
###                Path(temp_path).unlink()
###            else:
###                # Use add_usd_layer for sublayer
###                handle = self._renderer.add_usd_layer(usda_content, path_prefix=None)
###            
###            if self._usd_handles is not None:
###                self._usd_handles.append(handle)
###        
###        # Create binding for camera transforms (all cameras at once)
###        camera_paths = [f"/Render/Camera_{i}" for i in range(self._num_envs)]
###        if self._renderer is not None:
###            self._camera_binding = self._renderer.bind_attribute(
###                prim_paths=camera_paths,
###                attribute_name="omni:fabric:worldMatrix",
###                semantic="transform_4x4",
###                prim_mode="must_exist",
###            )
###        
###        self._initialized_scene = True
###        print(f"OVRTX scene setup complete: {self._num_envs} cameras and render products created")
###        print(f"Render product paths: {self._render_product_paths[:3]}{'...' if self._num_envs > 3 else ''}")

    def render(self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor):
        """Render the scene using OVRTX.
        
        Args:
            camera_positions: Tensor of shape (num_envs, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_envs, 4) - camera quaternions (x, y, z, w) in world frame
            intrinsic_matrices: Tensor of shape (num_envs, 3, 3) - camera intrinsic matrices
        """
        # Scene should already be set up during initialize()
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. This should not happen - scene setup should occur in initialize()")
        
        # Increment frame counter
        self._frame_counter += 1
        
        num_envs = camera_positions.shape[0]
        
        # Camera orientations are already in OpenGL convention from USD
        # No conversion needed!
        camera_quats_opengl = camera_orientations
        
        # Convert torch tensors to warp arrays
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_orientations_wp = wp.from_torch(camera_quats_opengl.contiguous(), dtype=wp.quatf)
        
        # Create camera transforms array
        camera_transforms = wp.zeros(num_envs, dtype=wp.mat44d, device="cuda:0")
        
        # Launch kernel to populate transforms
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device="cuda:0",
        )
        
        # Update camera transforms in the scene using the binding
        if self._camera_binding is not None:
            with self._camera_binding.map(device="cuda", device_id=0) as attr_mapping:
                wp_transforms_view = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                
                # Debug: Print transforms before and after update (first frame only)
                if self._frame_counter == 1:
                    self._print_camera_transforms_debug(
                        wp_transforms_view, 
                        camera_transforms, 
                        camera_positions, 
                        camera_orientations
                    )
                
                # Copy our computed transforms to the mapped buffer
                wp.copy(wp_transforms_view, camera_transforms)
                # Unmap will commit the changes
        
        # Update object transforms from Newton physics
        self._update_object_transforms()
        
        # Step the renderer to produce frames
        # We now have a single RenderProduct that references all cameras and outputs a tiled image
        print(f"[DEBUG] render_product_paths: {self._render_product_paths}")
        if self._renderer is not None and len(self._render_product_paths) > 0:
            try:
                # Render using the single render product
                render_product_set = set(self._render_product_paths)
                
                products = self._renderer.step(
                    render_products=render_product_set, 
                    delta_time=1.0/60.0
                )
                print(f"[DEBUG] Products: {products}")
                
                # Extract rendered images from the single render product
                # The product should contain a single tiled frame
                product_path = self._render_product_paths[0]
                if product_path in products:
                    product = products[product_path]
                    
                    if len(product.frames) > 0:
                        frame = product.frames[0]
                        print(f"[DEBUG] Frame has {len(product.frames)} frame(s)")
                        
                        # Extract LdrColor (RGBA) if available - this is the tiled image
                        if "LdrColor" in frame.render_vars:
                            with frame.render_vars["LdrColor"].map(device="cuda") as mapping:
                                tiled_data = wp.from_dlpack(mapping.tensor)
                                print(f"[DEBUG] Tiled data shape: {tiled_data.shape}")
                                
                                # Save the full tiled image
                                self._save_tiled_image_to_disk(tiled_data)
                                
                                # Extract individual tiles for each environment
                                for env_idx in range(self._num_envs):
                                    # Calculate tile position in grid
                                    tile_x = env_idx % self._num_tiles_per_side
                                    tile_y = env_idx // self._num_tiles_per_side
                                    
                                    # Extract this tile using kernel
                                    wp.launch(
                                        kernel=_extract_tile_from_tiled_buffer_kernel,
                                        dim=(self._height, self._width),
                                        inputs=[
                                            tiled_data,
                                            self._output_data_buffers["rgba"][env_idx],
                                            tile_x,
                                            tile_y,
                                            self._width,
                                            self._height,
                                        ],
                                        device="cuda:0",
                                    )
                                    
                                    # Save individual image
                                    self._save_image_to_disk(self._output_data_buffers["rgba"][env_idx], env_idx)
                        
                        # Extract depth if available and configured
                        # TODO: Add depth RenderVar to the RenderProduct USD definition
                        # if self._cfg.output_annotators and "depth" in self._cfg.output_annotators:
                        #     if "distance_to_camera" in frame.render_vars:
                        #         with frame.render_vars["distance_to_camera"].map(device="cuda") as mapping:
                        #             depth_data = wp.from_dlpack(mapping.tensor)
                        #             # Extract tiles for depth as well
        
            except Exception as e:
                print(f"Warning: OVRTX rendering failed: {e}")
                import traceback
                traceback.print_exc()
                # Keep the output buffers as-is (zeros from initialization)

    def _print_camera_transforms_debug(
        self, 
        ovrtx_transforms: wp.array, 
        new_transforms: wp.array,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor
    ):
        """Print camera transforms before and after update for debugging.
        
        Args:
            ovrtx_transforms: Current transforms in OVRTX (before update)
            new_transforms: New transforms to apply
            camera_positions: Camera positions from Isaac Lab
            camera_orientations: Camera orientations from Isaac Lab
        """
        print("\n" + "="*80)
        print("CAMERA TRANSFORM DEBUG (Frame 1)")
        print("="*80)
        
        # Convert to torch for easier printing
        ovrtx_transforms_torch = wp.to_torch(ovrtx_transforms).cpu()
        new_transforms_torch = wp.to_torch(new_transforms).cpu()
        
        # Print first 3 cameras (or all if less than 3)
        num_to_print = min(3, self._num_envs)
        
        for i in range(num_to_print):
            print(f"\n--- Camera {i} ---")
            
            # Print Isaac Lab inputs
            print(f"Isaac Lab Input:")
            print(f"  Position (world): {camera_positions[i].cpu().numpy()}")
            print(f"  Orientation (world, xyzw): {camera_orientations[i].cpu().numpy()}")
            
            # Print OVRTX current transform (before update)
            print(f"\nOVRTX Current Transform (BEFORE update):")
            ovrtx_mat = ovrtx_transforms_torch[i]
            for row in range(4):
                print(f"  [{ovrtx_mat[row, 0]:8.4f}, {ovrtx_mat[row, 1]:8.4f}, "
                      f"{ovrtx_mat[row, 2]:8.4f}, {ovrtx_mat[row, 3]:8.4f}]")
            
            # Print new transform (after conversion, before update)
            print(f"\nNew Transform (AFTER conversion, to be applied):")
            new_mat = new_transforms_torch[i]
            for row in range(4):
                print(f"  [{new_mat[row, 0]:8.4f}, {new_mat[row, 1]:8.4f}, "
                      f"{new_mat[row, 2]:8.4f}, {new_mat[row, 3]:8.4f}]")
            
            # Extract translation from new transform for easy verification
            translation = new_mat[3, :3]
            print(f"\n  Translation from matrix: {translation.numpy()}")
        
        if self._num_envs > 3:
            print(f"\n... ({self._num_envs - 3} more cameras not shown)")
        
        print("\n" + "="*80 + "\n")
    
    def _update_object_transforms(self):
        """Update object transforms from Newton physics state to OVRTX.
        
        Syncs all dynamic objects (robot bodies, manipulated objects) from Newton's
        physics simulation to OVRTX's render state using GPU kernels for efficiency.
        """
        if self._object_binding is None or self._object_newton_indices is None:
            return
        
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            
            # Get Newton physics state
            newton_state = NewtonManager.get_state_0()
            if newton_state is None:
                return
            
            # Map OVRTX transforms and update from Newton
            with self._object_binding.map(device="cuda", device_id=0) as attr_mapping:
                ovrtx_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                
                # Launch kernel to sync transforms
                wp.launch(
                    kernel=_sync_newton_transforms_kernel,
                    dim=len(self._object_newton_indices),
                    inputs=[ovrtx_transforms, self._object_newton_indices, newton_state.body_q],
                    device="cuda:0",
                )
                # Unmap will commit the changes
                
        except Exception as e:
            # Silently fail to avoid spamming console
            if self._frame_counter == 1:
                print(f"[OVRTX] Warning: Failed to update object transforms: {e}")

    def _save_image_to_disk(self, rendered_data_wp: wp.array, env_idx: int):
        """Save rendered image to disk.
        
        Args:
            rendered_data_wp: Warp array containing RGBA data, shape (height, width, 4)
            env_idx: Environment index for filename
        """
        try:
            # Convert warp array to torch tensor, then to numpy
            rendered_data_torch = wp.to_torch(rendered_data_wp)
            rendered_data_np = rendered_data_torch.cpu().numpy()
            
            # Convert from float [0, 1] to uint8 [0, 255]
            if rendered_data_np.dtype in [np.float32, np.float64]:
                rendered_data_np = (rendered_data_np * 255).astype(np.uint8)
            
            # Create output directory if it doesn't exist
            output_dir = Path("ovrtx_rendered_images")
            output_dir.mkdir(exist_ok=True)
            
            # Save as PNG
            # rendered_data_np is shape (height, width, 4) for RGBA
            if len(rendered_data_np.shape) == 3 and rendered_data_np.shape[2] == 4:
                # RGBA image
                image = Image.fromarray(rendered_data_np, mode='RGBA')
            elif len(rendered_data_np.shape) == 3 and rendered_data_np.shape[2] == 3:
                # RGB image
                image = Image.fromarray(rendered_data_np, mode='RGB')
            elif len(rendered_data_np.shape) == 2:
                # Grayscale image
                image = Image.fromarray(rendered_data_np, mode='L')
            else:
                print(f"Warning: Unexpected image shape {rendered_data_np.shape}, cannot save")
                return
            
            # Save with frame and environment index in filename
            output_path = output_dir / f"frame_{self._frame_counter:06d}_env_{env_idx:04d}.png"
            image.save(output_path)
            
            # Only print for first environment and first few frames to avoid spam
            if env_idx == 0 and self._frame_counter <= 5:
                print(f"[OVRTX] Saved rendered image: {output_path}")
                
        except Exception as e:
            print(f"Warning: Failed to save image for env {env_idx}: {e}")
    
    def _save_tiled_image_to_disk(self, tiled_data_wp: wp.array):
        """Save tiled image (all environments in a grid) to disk.
        
        Args:
            tiled_data_wp: Warp array containing tiled RGBA data, shape (tiled_height, tiled_width, 4)
        """
        try:
            # Convert warp array to torch tensor, then to numpy
            tiled_data_torch = wp.to_torch(tiled_data_wp)
            tiled_data_np = tiled_data_torch.cpu().numpy()
            
            # Convert from float [0, 1] to uint8 [0, 255]
            if tiled_data_np.dtype in [np.float32, np.float64]:
                tiled_data_np = (tiled_data_np * 255).astype(np.uint8)
            
            # Create output directory if it doesn't exist
            output_dir = Path("ovrtx_rendered_images")
            output_dir.mkdir(exist_ok=True)
            
            # Save as PNG
            image = Image.fromarray(tiled_data_np, mode='RGBA')
            output_path = output_dir / f"frame_{self._frame_counter:06d}_tiled.png"
            image.save(output_path)
            
            # Print only for first few frames
            if self._frame_counter <= 5:
                print(f"[OVRTX] Saved tiled image ({self._num_envs} envs in {self._num_tiles_per_side}x{self._num_tiles_per_side} grid): {output_path}")
                
        except Exception as e:
            print(f"Warning: Failed to save tiled image: {e}")
            import traceback
            traceback.print_exc()

    def step(self):
        """Step the renderer."""
        # The actual rendering happens in render()
        # This is called each simulation step but we don't need to do anything here
        pass

    def reset(self):
        """Reset the renderer."""
        if self._renderer:
            self._renderer.reset(time=0.0)

    def close(self):
        """Close the renderer and release resources."""
        if self._camera_binding:
            try:
                self._camera_binding.unbind()
            except Exception as e:
                print(f"Warning: Error unbinding camera transforms: {e}")
            self._camera_binding = None
        
        if self._renderer:
            # Remove any USD content we added
            if self._usd_handles is not None:
                for handle in self._usd_handles:
                    try:
                        self._renderer.remove_usd(handle)
                    except Exception as e:
                        print(f"Warning: Error removing USD: {e}")
                self._usd_handles.clear()
            
            # Renderer cleanup is handled automatically by __del__
            self._renderer = None
        
        # Clear render product paths
        self._render_product_paths.clear()
        self._initialized_scene = False
