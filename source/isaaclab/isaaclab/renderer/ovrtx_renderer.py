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
    _render_product_path = "/Render/Camera"
    _initialized_scene = False

    def __init__(self, cfg: OVRTXRendererCfg):
        super().__init__(cfg)
        self._usd_handles = []

    def initialize(self):
        """Initialize the OVRTX renderer."""
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

    def _setup_scene(self):
        """Set up the USD scene with cameras.
        
        This creates a minimal USD scene with camera definitions.
        OVRTX uses the default viewport render product.
        """
        if self._initialized_scene:
            return
            
        print("Setting up OVRTX scene...")
        
        # Create a simple USD layer with camera(s)
        # For now, we create one camera per environment
        camera_usda_parts = ['#usda 1.0\n(defaultPrim = "Render")\n\n']
        camera_usda_parts.append('def Scope "Render" {\n')
        
        # Create cameras for each environment
        for env_idx in range(self._num_envs):
            camera_path = f"/Render/Camera_{env_idx}"
            camera_usda_parts.append(f'''
    def Camera "Camera_{env_idx}" {{
        float focalLength = 18.0
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        token projection = "perspective"
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
''')
        
        camera_usda_parts.append('}\n')
        camera_usda = ''.join(camera_usda_parts)
        
        # Add the camera layer to the renderer
        if self._renderer is not None:
            handle = self._renderer.add_usd_layer(camera_usda, path_prefix="/Render")
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
        print(f"OVRTX scene setup complete: {self._num_envs} cameras created")

    def render(self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor):
        """Render the scene using OVRTX.
        
        Args:
            camera_positions: Tensor of shape (num_envs, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_envs, 4) - camera quaternions (x, y, z, w) in world frame
            intrinsic_matrices: Tensor of shape (num_envs, 3, 3) - camera intrinsic matrices
        """
        # Setup scene on first render call
        if not self._initialized_scene:
            self._setup_scene()
        
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
        
        # Step the renderer to produce a frame
        # Use the standard Hydra viewport texture render product
        # TODO: For multiple environments, we need multiple render products
        # For now, render using the default viewport
        if self._renderer is not None:
            try:
                products = self._renderer.step(
                    render_products={"/Render/OmniverseKit/HydraTextures/ViewportTexture0"}, 
                    delta_time=1.0/60.0  # Assume 60 Hz
                )
                
                # Extract rendered images
                for product_name, product in products.items():
                    for frame_idx, frame in enumerate(product.frames):
                        # Get LdrColor (RGB) render variable
                        if "LdrColor" in frame.render_vars:
                            with frame.render_vars["LdrColor"].map(device="cuda") as mapping:
                                # Copy to our output buffer
                                rendered_data = wp.from_dlpack(mapping.tensor)
                                # TODO: Handle the case where we have fewer frames than environments
                                # For now, just copy to the first environment's buffer
                                if frame_idx < self._num_envs:
                                    wp.copy(self._output_data_buffers["rgba"][frame_idx], rendered_data)
                        
                        # Get depth if available
                        # TODO: Depth rendering needs to be configured in the render product
                        if "depth" in frame.render_vars:
                            with frame.render_vars["depth"].map(device="cuda") as mapping:
                                depth_data = wp.from_dlpack(mapping.tensor)
                                if frame_idx < self._num_envs:
                                    wp.copy(self._output_data_buffers["depth"][frame_idx], depth_data)
        
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
        
        self._initialized_scene = False
