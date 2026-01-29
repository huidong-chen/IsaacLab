# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation."""

import math
import os
import sys
from pathlib import Path

import torch
import warp as wp

# Set environment variables for OVRTX
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
# Set LD_PRELOAD if needed (libcarb.so)
libcarb_path = Path.home() / "dev/kit.0/kit/_build/linux-x86_64/release/libcarb.so"
if libcarb_path.exists() and "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = str(libcarb_path)

# Add ovrtx Python bindings to path
ovrtx_bindings_path = Path("/home/ncournia/dev/kit.0/rendering/source/bindings/python")
if str(ovrtx_bindings_path) not in sys.path:
    sys.path.insert(0, str(ovrtx_bindings_path))

# Set library path hint before importing ovrtx
import ovrtx._src.bindings as bindings
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
    _0 = wp.float64(0.0)
    _1 = wp.float64(1.0)
    # Note: Type issues with warp vec3 indexing are expected
    transforms[i] = wp.mat44d(  # type: ignore
        wp.float64(r00), wp.float64(r01), wp.float64(r02), _0,
        wp.float64(r10), wp.float64(r11), wp.float64(r12), _0,
        wp.float64(r20), wp.float64(r21), wp.float64(r22), _0,
        wp.float64(float(pos[0])), wp.float64(float(pos[1])), wp.float64(float(pos[2])), _1
    )


class OVRTXRenderer(RendererBase):
    """OVRTX Renderer implementation using the ovrtx library.
    
    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    _renderer: Renderer | None = None
    _usd_handles: list | None = None
    _camera_binding = None
    _render_product_paths: list[str] = []
    _initialized_scene = False

    def __init__(self, cfg: OVRTXRendererCfg):
        super().__init__(cfg)
        self._usd_handles = []
        self._render_product_paths = []

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
            camera_paths = [f"/Render/Camera_{i}" for i in range(self._num_envs)]
            self._camera_binding = self._renderer.bind_attribute(
                prim_paths=camera_paths,
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4",
                prim_mode="must_exist",
            )
        else:
            # Setup cameras as root layer
            self._setup_scene(as_root_layer=True)
    
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
        
        for env_idx in range(self._num_envs):
            camera_name = f"Camera_{env_idx}"
            render_product_name = f"RenderProduct_{env_idx}"
            camera_path = f"/Render/{camera_name}"
            render_product_path = f"/Render/{render_product_name}"
            
            self._render_product_paths.append(render_product_path)
            
            camera_parts.append(f'''
    def Camera "{camera_name}" (
        prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]
    ) {{
        float focalLength = 18.0
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        token projection = "perspective"
        float2 clippingRange = (1, 10000000)
        bool omni:rtx:autoExposure:enabled = 1
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
''')
            
            camera_parts.append(f'''
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = <{camera_path}>
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "RealTimePathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = </Render/Vars/LdrColor>
        uniform int2 resolution = ({self._width}, {self._height})
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

    def _setup_scene(self, as_root_layer: bool = True):
        """Set up the USD scene with cameras and render products.
        
        This creates a USD scene with camera definitions and their corresponding
        RenderProduct prims according to the OVRTX requirements.
        
        Args:
            as_root_layer: If True, creates cameras as root layer with 'def Scope "Render"'.
                          If False, creates cameras as sublayer with 'over "Render"'.
                          Use False when a USD scene has already been loaded as root layer.
        """
        if self._initialized_scene:
            return
            
        print("Setting up OVRTX scene...")
        
        # Create a USD layer with cameras and render products
        usda_parts = []
        
        if as_root_layer:
            # Creating as root layer: must set defaultPrim and use 'def'
            usda_parts.append('#usda 1.0\n')
            usda_parts.append('(\n    defaultPrim = "Render"\n)\n\n')
            usda_parts.append('def Scope "Render" {\n')
        else:
            # Creating as sublayer: NO header, use 'over' to extend existing scene
            # The header would make this a root layer!
            usda_parts.append('over "Render" {\n')
        
        # Create cameras and render products for each environment
        for env_idx in range(self._num_envs):
            camera_name = f"Camera_{env_idx}"
            render_product_name = f"RenderProduct_{env_idx}"
            camera_path = f"/Render/{camera_name}"
            render_product_path = f"/Render/{render_product_name}"
            
            # Store render product path for later use
            self._render_product_paths.append(render_product_path)
            
            # Camera definition with RTX API schemas
            usda_parts.append(f'''
    def Camera "{camera_name}" (
        prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]
    ) {{
        float focalLength = 18.0
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        token projection = "perspective"
        float2 clippingRange = (1, 10000000)
        bool omni:rtx:autoExposure:enabled = 1
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
''')
            
            # RenderProduct definition with RTX settings
            usda_parts.append(f'''
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = <{camera_path}>
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "RealTimePathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = </Render/Vars/LdrColor>
        uniform int2 resolution = ({self._width}, {self._height})
    }}
''')
        
        # Add shared RenderVar if it doesn't exist
        usda_parts.append('''
    def "Vars"
    {
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
''')
        
        usda_parts.append('}\n')
        usda_content = ''.join(usda_parts)
        
        # Add the USD to the renderer
        if self._renderer is not None:
            if as_root_layer:
                # Save to temp file and use add_usd for root layer
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False) as f:
                    f.write(usda_content)
                    temp_path = f.name
                handle = self._renderer.add_usd(temp_path, path_prefix=None)
                # Clean up temp file
                Path(temp_path).unlink()
            else:
                # Use add_usd_layer for sublayer
                handle = self._renderer.add_usd_layer(usda_content, path_prefix=None)
            
            if self._usd_handles is not None:
                self._usd_handles.append(handle)
        
        # Create binding for camera transforms (all cameras at once)
        camera_paths = [f"/Render/Camera_{i}" for i in range(self._num_envs)]
        if self._renderer is not None:
            self._camera_binding = self._renderer.bind_attribute(
                prim_paths=camera_paths,
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4",
                prim_mode="must_exist",
            )
        
        self._initialized_scene = True
        print(f"OVRTX scene setup complete: {self._num_envs} cameras and render products created")
        print(f"Render product paths: {self._render_product_paths[:3]}{'...' if self._num_envs > 3 else ''}")

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
        
        num_envs = camera_positions.shape[0]
        
        # Convert camera orientations from Isaac Lab convention to OpenGL convention
        camera_quats_converted = convert_camera_frame_orientation_convention(
            camera_orientations, origin="world", target="opengl"
        )
        
        # Convert torch tensors to warp arrays
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_orientations_wp = wp.from_torch(camera_quats_converted.contiguous(), dtype=wp.quatf)
        
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
                # Copy our computed transforms to the mapped buffer
                wp.copy(wp_transforms_view, camera_transforms)
                # Unmap will commit the changes
        
        # Step the renderer to produce frames
        # Now we have properly configured render products
        if self._renderer is not None and len(self._render_product_paths) > 0:
            try:
                # Render using all configured render products
                render_product_set = set(self._render_product_paths)
                
                products = self._renderer.step(
                    render_products=render_product_set, 
                    delta_time=1.0/60.0
                )
                
                # Extract rendered images from each render product
                for env_idx, product_path in enumerate(self._render_product_paths):
                    if env_idx >= self._num_envs:
                        break
                    
                    if product_path in products:
                        product = products[product_path]
                        
                        # Get the first frame from this product
                        if len(product.frames) > 0:
                            frame = product.frames[0]
                            
                            # Extract LdrColor (RGBA) if available
                            if "LdrColor" in frame.render_vars:
                                with frame.render_vars["LdrColor"].map(device="cuda") as mapping:
                                    rendered_data = wp.from_dlpack(mapping.tensor)
                                    # Copy to our output buffer for this environment
                                    print("copied rgba data")
                                    wp.copy(self._output_data_buffers["rgba"][env_idx], rendered_data)
                            
                            # Extract depth if available and configured
                            # TODO: Add depth RenderVar to the RenderProduct USD definition
                            # if self._cfg.output_annotators and "depth" in self._cfg.output_annotators:
                            #     if "distance_to_camera" in frame.render_vars:
                            #         with frame.render_vars["distance_to_camera"].map(device="cuda") as mapping:
                            #             depth_data = wp.from_dlpack(mapping.tensor)
                            #             wp.copy(self._output_data_buffers["depth"][env_idx], depth_data)
        
            except Exception as e:
                print(f"Warning: OVRTX rendering failed: {e}")
                # Keep the output buffers as-is (zeros from initialization)

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
