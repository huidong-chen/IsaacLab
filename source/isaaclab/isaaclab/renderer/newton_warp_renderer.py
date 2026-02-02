# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

import math
import numpy as np
import torch
from pathlib import Path

import warp as wp
from newton.sensors import TiledCameraSensor
from PIL import Image

from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .renderer import RendererBase


@wp.kernel
def _create_camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quatf),
    transforms: wp.array(dtype=wp.transformf, ndim=2),
):
    """Kernel to create camera transforms from positions and orientations.

    Args:
        positions: Array of camera positions, shape (num_cameras,)
        orientations: Array of camera orientations, shape (num_cameras,)
        transforms: Output array of camera transforms, shape (num_cameras, 1)
    """
    i = wp.tid()
    transforms[i, 0] = wp.transformf(positions[i], orientations[i])


@wp.kernel
def _convert_raw_rgb_tiled(
    raw_buffer: wp.array(dtype=wp.uint32, ndim=3),
    output_buffer: wp.array(dtype=wp.uint8, ndim=3),
    image_width: int,
    image_height: int,
    num_tiles_x: int,
):
    """Convert raw tiled RGB buffer (uint32 packed) to tiled RGBA (uint8). For debugging purposes.

    The raw buffer has shape (num_worlds num_cameras, width * height) where each uint32 encodes RGBA as 4 bytes.
    This kernel converts it to the tiled format (tiled_height, tiled_width, 4) of uint8.

    Args:
        raw_buffer: Input raw buffer from TiledCameraSensor, shape (num_worlds, num_cameras, width * height) of uint32
        output_buffer: Output buffer in tiled format (tiled_height, tiled_width, 4) of uint8
        image_width: Width of each camera image
        image_height: Height of each camera image
        num_tiles_x: Number of tiles in x-direction (horizontal)
    """
    y, x = wp.tid()

    # Determine which tile and which pixel within the tile
    # x is width (horizontal), y is height (vertical)
    tile_x = x // image_width
    tile_y = y // image_height
    pixel_x = x % image_width
    pixel_y = y % image_height

    # Compute camera ID from tile position
    camera_id = tile_y * num_tiles_x + tile_x

    # Compute the pixel index within this camera's buffer
    # The buffer is flattened as (width * height), so row-major indexing
    pixel_idx = pixel_y * image_width + pixel_x

    # Read the packed uint32 value from raw_buffer[camera_id, 0, pixel_idx]
    packed_color = raw_buffer[camera_id, 0, pixel_idx]

    # Compute output x coordinate
    output_x = tile_x * image_width + pixel_x

    # Unpack the uint32 into 4 uint8 values (RGBA channels)
    # Assuming little-endian byte order: ABGR format in memory
    output_buffer[y, output_x, 0] = wp.uint8((packed_color >> wp.uint32(0)) & wp.uint32(0xFF))  # R
    output_buffer[y, output_x, 1] = wp.uint8((packed_color >> wp.uint32(8)) & wp.uint32(0xFF))  # G
    output_buffer[y, output_x, 2] = wp.uint8((packed_color >> wp.uint32(16)) & wp.uint32(0xFF))  # B
    output_buffer[y, output_x, 3] = wp.uint8((packed_color >> wp.uint32(24)) & wp.uint32(0xFF))  # A


@wp.kernel
def _convert_raw_depth_tiled(
    raw_buffer: wp.array(dtype=wp.float32, ndim=3),
    output_buffer: wp.array(dtype=wp.float32, ndim=3),
    image_width: int,
    image_height: int,
    num_tiles_x: int,
):
    """Convert raw tiled depth buffer to tiled depth format. For debugging purposes.

    The raw buffer has shape (num_worlds, num_cameras, width * height) of float32 depth values.
    This kernel converts it to the tiled format (tiled_height, tiled_width, 1) of float32.

    Args:
        raw_buffer: Input raw buffer from TiledCameraSensor, shape (num_worlds, num_cameras, width * height) of float32
        output_buffer: Output buffer in tiled format (tiled_height, tiled_width, 1) of float32
        image_width: Width of each camera image
        image_height: Height of each camera image
        num_tiles_x: Number of tiles in x-direction (horizontal)
    """
    y, x = wp.tid()

    # Determine which tile and which pixel within the tile
    # x is width (horizontal), y is height (vertical)
    tile_x = x // image_width
    tile_y = y // image_height
    pixel_x = x % image_width
    pixel_y = y % image_height

    # Compute camera ID from tile position
    camera_id = tile_y * num_tiles_x + tile_x

    # Compute the pixel index within this camera's buffer
    # The buffer is flattened as (width * height), so row-major indexing
    pixel_idx = pixel_y * image_width + pixel_x

    # Compute output x coordinate
    output_x = tile_x * image_width + pixel_x

    # Copy the depth value from raw_buffer[camera_id, 0, pixel_idx]
    output_buffer[y, output_x, 0] = raw_buffer[camera_id, 0, pixel_idx]


class NewtonWarpRenderer(RendererBase):
    """Newton Warp Renderer implementation."""

    _model = None

    # tiled camerae sensor from warp trace
    _tiled_camera_sensor = None
    
    # Frame counter for debug output
    _frame_counter: int = 0

    def __init__(self, cfg: NewtonWarpRendererCfg):
        super().__init__(cfg)
        self._frame_counter = 0

    def initialize(self):
        """Initialize the renderer."""
        self._model = NewtonManager.get_model()

        self._tiled_camera_sensor = TiledCameraSensor(
            model=self._model,
            num_cameras=1,  # TODO: currently only supports 1 camera per world
            width=self._width,
            height=self._height,
            options=TiledCameraSensor.Options(colors_per_shape=True),
        )

        # Note: camera rays will be computed when we have access to TiledCamera
        # for now use default 45 degree FOV
        self._camera_rays = None

        # Initialize output buffers
        self._initialize_output()

    def set_camera_rays_from_intrinsics(self, intrinsic_matrices: torch.Tensor):
        """Set camera FOV from intrinsic matrices (vectorized for all cameras).

        Args:
            intrinsic_matrices: Camera intrinsic matrices of shape (num_cameras, 3, 3)
                             Format: [[f_x,   0, c_x],
                                     [  0, f_y, c_y],
                                     [  0,   0,   1]]
        """
        # Extract vertical focal lengths for all cameras (vectorized)
        # Shape: (num_cameras,)
        f_y_all = intrinsic_matrices[:, 1, 1]  # All cameras' vertical focal lengths in pixels

        # Calculate vertical FOV for all cameras (vectorized)
        # fov = 2 * atan(height / (2 * f_y))
        # Shape: (num_cameras,)
        fov_radians_all = 2.0 * torch.atan(self._height / (2.0 * f_y_all))

        # Convert to warp array
        fov_radians_wp = wp.from_torch(fov_radians_all, dtype=wp.float32)

        # Compute camera rays with per-camera FOVs (vectorized)
        # TiledCameraSensor.compute_pinhole_camera_rays accepts array of FOVs
        self._camera_rays = self._tiled_camera_sensor.compute_pinhole_camera_rays(fov_radians_wp)

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        self._data_types = ["rgba", "rgb", "depth"]
        self._num_tiles_per_side = math.ceil(math.sqrt(self._num_envs))

        # Raw buffer to hold data from the tiled camera sensor
        self._raw_output_rgb_buffer = self._tiled_camera_sensor.create_color_image_output()
        self._raw_output_depth_buffer = self._tiled_camera_sensor.create_depth_image_output()

        self._output_data_buffers["rgba"] = wp.zeros(
            (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=self._raw_output_rgb_buffer.device
        )
        # Create RGB view that references the same underlying array as RGBA, but only first 3 channels
        self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]
        self._output_data_buffers["depth"] = wp.zeros(
            (self._num_envs, self._height, self._width, 1),
            dtype=wp.float32,
            device=self._raw_output_depth_buffer.device,
        )
    
    def _print_camera_transforms_debug(
        self,
        newton_transforms: wp.array,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
    ):
        """Print debug information about camera transforms for Newton renderer.
        
        Args:
            newton_transforms: Warp array of wp.transformf (num_envs, 1)
            camera_positions: Isaac Lab camera positions (num_envs, 3) in world frame
            camera_orientations: Isaac Lab camera orientations (num_envs, 4) in OpenGL convention (xyzw)
        """
        print("\n" + "="*80)
        print("NEWTON WARP RENDERER - Camera Transform Debug (Frame 1)")
        print("="*80)
        
        # Convert to CPU for printing
        camera_positions_cpu = camera_positions.cpu()
        camera_orientations_cpu = camera_orientations.cpu()
        
        # Copy Newton transforms to numpy via torch
        newton_transforms_torch = wp.to_torch(newton_transforms)
        newton_transforms_cpu = newton_transforms_torch.cpu().numpy()
        
        # Print for first 3 cameras (or all if fewer)
        num_to_print = min(3, camera_positions.shape[0])
        
        for i in range(num_to_print):
            print(f"\n--- Camera {i} ---")
            
            # Isaac Lab Input (OpenGL convention)
            print(f"Isaac Lab Input (OpenGL convention):")
            pos = camera_positions_cpu[i]
            quat_opengl = camera_orientations_cpu[i]
            print(f"  Position (world): {pos.numpy()}")
            print(f"  Orientation (opengl, xyzw): {quat_opengl.numpy()}")
            
            # Newton Transform (wp.transformf accessed via numpy array)
            print(f"\nNewton Transform (wp.transformf):")
            print(f"  (Full transform struct shown via torch conversion)")
            print(f"  Transform data: {newton_transforms_cpu[i, 0]}")
            
        print("\n" + "="*80 + "\n")

    def render(
        self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor
    ):
        """Render the scene.

        Args:
            camera_positions: Tensor of shape (num_envs, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_envs, 4) - camera quaternions (x, y, z, w) in world frame
            intrinsic_matrices: Tensor of shape (num_envs, 3, 3) - camera intrinsic matrices
        """
        # Increment frame counter
        self._frame_counter += 1
        
        if self._camera_rays is None:
            self.set_camera_rays_from_intrinsics(intrinsic_matrices)
        num_envs = camera_positions.shape[0]

        # Camera orientations are already in OpenGL convention from USD
        # No conversion needed!
        camera_quats_opengl = camera_orientations

        # Convert torch tensors to warp arrays directly on GPU
        # Positions: shape (num_envs, 3) -> shape (num_envs,) of vec3
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_orientations_wp = wp.from_torch(camera_quats_opengl, dtype=wp.quat)

        # Create camera transforms array, TODO: num_cameras = 1
        # Format: wp.array of shape (num_envs, num_cameras), dtype=wp.transformf
        camera_transforms = wp.empty((num_envs, 1), dtype=wp.transformf, device=camera_positions_wp.device)

        # Launch kernel to populate transforms (vectorized operation)
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=camera_positions_wp.device,
        )
        
        # Debug: Print transforms on first frame
        if self._frame_counter == 1:
            self._print_camera_transforms_debug(
                camera_transforms,
                camera_positions,
                camera_orientations,
            )

        # Render using TiledCameraSensor
        self._tiled_camera_sensor.render(
            state=NewtonManager.get_state_0(),  # Use current physics state
            camera_transforms=camera_transforms,
            camera_rays=self._camera_rays,
            color_image=self._raw_output_rgb_buffer,
            depth_image=self._raw_output_depth_buffer,
        )

        # Convert uint32 to uint8 RGBA
        reshape_rgba = self._raw_output_rgb_buffer.reshape((self._num_envs, self._height, self._width))
        self._output_data_buffers["rgba"] = wp.array(
            ptr=reshape_rgba.ptr, shape=(*reshape_rgba.shape, 4), dtype=wp.uint8
        )

        self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]

        # Reshape depth buffer: (num_envs, num_cameras, 1, width*height) -> (num_envs, num_cameras, height, width, 1), TODO: num_cameras = 1
        self._output_data_buffers["depth"] = self._raw_output_depth_buffer.reshape(
            (self._num_envs, self._height, self._width, 1)
        )
        
        # Save rendered images to disk
        for env_idx in range(self._num_envs):
            self._save_image_to_disk(self._output_data_buffers["rgba"][env_idx], env_idx)
        
        # Save tiled image (all environments in a grid)
        self._save_tiled_image_to_disk()

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
            
            # Data is already uint8 from Newton renderer
            if rendered_data_np.dtype not in [np.uint8]:
                # Convert from float [0, 1] to uint8 [0, 255] if needed
                if rendered_data_np.dtype in [np.float32, np.float64]:
                    rendered_data_np = (rendered_data_np * 255).astype(np.uint8)
            
            # Create output directory if it doesn't exist
            output_dir = Path("newton_rendered_images")
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
                print(f"[Newton] Saved rendered image: {output_path}")
                
        except Exception as e:
            print(f"Warning: Failed to save Newton image for env {env_idx}: {e}")
    
    def _save_tiled_image_to_disk(self):
        """Save tiled image (all environments in a grid) to disk.
        
        This creates a single image with all environment views arranged in a grid.
        """
        try:
            # Calculate tiled dimensions
            tiled_height = self._num_tiles_per_side * self._height
            tiled_width = self._num_tiles_per_side * self._width
            
            # Create tiled output buffer
            tiled_buffer = wp.zeros(
                (tiled_height, tiled_width, 4),
                dtype=wp.uint8,
                device=self._raw_output_rgb_buffer.device
            )
            
            # Launch kernel to convert raw buffer to tiled format
            wp.launch(
                kernel=_convert_raw_rgb_tiled,
                dim=(tiled_height, tiled_width),
                inputs=[
                    self._raw_output_rgb_buffer,
                    tiled_buffer,
                    self._width,
                    self._height,
                    self._num_tiles_per_side,
                ],
                device=self._raw_output_rgb_buffer.device,
            )
            
            # Convert to numpy
            tiled_data_torch = wp.to_torch(tiled_buffer)
            tiled_data_np = tiled_data_torch.cpu().numpy()
            
            # Create output directory
            output_dir = Path("newton_rendered_images")
            output_dir.mkdir(exist_ok=True)
            
            # Save as PNG
            image = Image.fromarray(tiled_data_np, mode='RGBA')
            output_path = output_dir / f"frame_{self._frame_counter:06d}_tiled.png"
            image.save(output_path)
            
            # Print only for first few frames
            if self._frame_counter <= 5:
                print(f"[Newton] Saved tiled image ({self._num_envs} envs in {self._num_tiles_per_side}x{self._num_tiles_per_side} grid): {output_path}")
                
        except Exception as e:
            print(f"Warning: Failed to save Newton tiled image: {e}")
            import traceback
            traceback.print_exc()

    def step(self):
        """Step the renderer."""
        pass

    def reset(self):
        """Reset the renderer."""
        pass

    def close(self):
        """Close the renderer."""
        pass
