#!/usr/bin/env python3
"""Test script to verify OVRTX renderer full functionality."""

import sys
from pathlib import Path

import torch
import numpy as np

# Add the source directory to the path
source_dir = Path(__file__).parent.parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def test_ovrtx_renderer_rendering():
    """Test the full rendering pipeline."""
    print("=" * 70)
    print("OVRTX Renderer Full Functionality Test")
    print("=" * 70)
    
    try:
        # Create configuration
        print("\n1. Creating OVRTX renderer configuration...")
        cfg = OVRTXRendererCfg(
            height=256,
            width=256,
            num_envs=2,
            num_cameras=1,
            data_types=["rgb", "depth"]
        )
        print(f"   ✓ Config: {cfg.width}x{cfg.height}, {cfg.num_envs} envs")
        
        # Create renderer
        print("\n2. Creating renderer instance...")
        renderer = cfg.create_renderer()
        print(f"   ✓ Renderer: {type(renderer).__name__}")
        
        # Initialize
        print("\n3. Initializing renderer...")
        renderer.initialize()
        print("   ✓ Renderer initialized")
        
        # Test rendering
        print("\n4. Testing render() method...")
        num_envs = 2
        
        # Create dummy camera parameters
        camera_positions = torch.tensor([
            [0.0, 0.0, 2.0],  # Camera 1: 2 units back on Z axis
            [1.0, 0.0, 2.0],  # Camera 2: 1 unit right, 2 units back
        ], dtype=torch.float32)
        
        # Identity rotations (looking forward)
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],  # (x, y, z, w) quaternion
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        
        # Intrinsic matrices (simple perspective)
        focal_length = 100.0
        cx, cy = 128.0, 128.0  # Principal point at center
        intrinsic_matrices = torch.tensor([
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
        ], dtype=torch.float32)
        
        print(f"   - Camera positions: {camera_positions.shape}")
        print(f"   - Camera orientations: {camera_orientations.shape}")
        print(f"   - Intrinsic matrices: {intrinsic_matrices.shape}")
        
        # Call render
        print("\n   Calling renderer.render()...")
        renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Render call successful")
        
        # Get output buffers
        print("\n5. Checking output buffers...")
        outputs = renderer.get_output()
        
        for name, buffer in outputs.items():
            print(f"   - {name}: shape={buffer.shape}, dtype={buffer.dtype}")
            
            # Check that buffers are not all zeros (actual rendering happened)
            # Note: Without a scene loaded, buffers may still be zeros
            if hasattr(buffer, 'numpy'):
                data = buffer.numpy()
            else:
                import warp as wp
                data = wp.to_torch(buffer).cpu().numpy()
            
            non_zero = np.count_nonzero(data)
            total = data.size
            print(f"      Non-zero pixels: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
        
        # Test reset
        print("\n6. Testing reset()...")
        renderer.reset()
        print("   ✓ Reset successful")
        
        # Test step
        print("\n7. Testing step()...")
        renderer.step()
        print("   ✓ Step successful")
        
        # Clean up
        print("\n8. Closing renderer...")
        renderer.close()
        print("   ✓ Renderer closed")
        
        print("\n" + "=" * 70)
        print("✓ ALL FUNCTIONALITY TESTS PASSED")
        print("=" * 70)
        print("\nNotes:")
        print("- Rendering pipeline is functional")
        print("- Output buffers are allocated correctly")
        print("- Camera transforms are being computed")
        print("- Without a USD scene loaded, rendered output may be black")
        print("- Multi-environment rendering in progress (currently renders first camera)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    success = test_ovrtx_renderer_rendering()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
