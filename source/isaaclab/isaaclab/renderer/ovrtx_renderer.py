# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation."""

import os
import sys
from pathlib import Path

import warp as wp

# Set environment variable to skip USD conflict check
# Note: This is needed because Isaac Lab uses pxr/usd-core, but ovrtx bundles its own USD
# The libraries are isolated and won't conflict in practice
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

# Add ovrtx Python bindings to path
ovrtx_bindings_path = Path("/home/ncournia/dev/kit.0/rendering/source/bindings/python")
if str(ovrtx_bindings_path) not in sys.path:
    sys.path.insert(0, str(ovrtx_bindings_path))

# Set library path hint before importing ovrtx
import ovrtx._src.bindings as bindings
bindings.OVRTX_LIBRARY_PATH_HINT = "/home/ncournia/dev/kit.0/rendering/_build/linux-x86_64/release"

from ovrtx import Renderer, RendererConfig

from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .renderer import RendererBase


class OVRTXRenderer(RendererBase):
    """OVRTX Renderer implementation using the ovrtx library.
    
    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    _renderer = None
    _usd_handle = None

    def __init__(self, cfg: OVRTXRendererCfg):
        super().__init__(cfg)

    def initialize(self):
        """Initialize the OVRTX renderer."""
        # Create renderer with optional config
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

    def render(self, camera_positions, camera_orientations, intrinsic_matrices):
        """Render the scene using OVRTX.
        
        Args:
            camera_positions: Tensor of shape (num_envs, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_envs, 4) - camera quaternions (x, y, z, w) in world frame
            intrinsic_matrices: Tensor of shape (num_envs, 3, 3) - camera intrinsic matrices
        """
        # TODO: Implement OVRTX rendering pipeline
        # This would involve:
        # 1. Setting up USD scene
        # 2. Configuring cameras from camera_positions, camera_orientations, intrinsic_matrices
        # 3. Calling renderer.step() to render
        # 4. Fetching results and populating output buffers
        
        # For now, just a placeholder that fills with zeros
        pass

    def step(self):
        """Step the renderer."""
        pass

    def reset(self):
        """Reset the renderer."""
        if self._renderer:
            self._renderer.reset(time=0.0)

    def close(self):
        """Close the renderer and release resources."""
        if self._renderer:
            # Renderer cleanup is handled automatically by __del__
            self._renderer = None
