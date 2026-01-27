# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.
    
    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    """

    renderer_type: str = "ov_rtx"
    """Type identifier for OVRTX renderer."""
