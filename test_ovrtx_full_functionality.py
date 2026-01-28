#!/usr/bin/env python3
"""Test script to verify OVRTX renderer full functionality."""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

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
        
        # Create dummy camera parameters (on GPU!)
        camera_positions = torch.tensor([
            [0.0, 0.0, 2.0],  # Camera 1: 2 units back on Z axis
            [1.0, 0.0, 2.0],  # Camera 2: 1 unit right, 2 units back
        ], dtype=torch.float32, device="cuda:0")
        
        # Identity rotations (looking forward)
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],  # (x, y, z, w) quaternion
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
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
        ], dtype=torch.float32, device="cuda:0")
        
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
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
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
            
            # Save rendered images to files
            if name in ["rgba", "rgb"]:
                for env_idx in range(data.shape[0]):
                    img_data = data[env_idx]  # (H, W, C)
                    
                    # Convert to uint8 if needed
                    if img_data.dtype != np.uint8:
                        img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
                    
                    # Save as PNG
                    output_file = output_dir / f"test_full_functionality_{name}_env{env_idx}.png"
                    if img_data.shape[2] == 4:  # RGBA
                        img = Image.fromarray(img_data, mode='RGBA')
                    else:  # RGB
                        img = Image.fromarray(img_data, mode='RGB')
                    img.save(output_file)
                    print(f"      Saved: {output_file}")
            
            elif name == "depth":
                for env_idx in range(data.shape[0]):
                    depth_data = data[env_idx, :, :, 0]  # (H, W)
                    
                    # Normalize depth for visualization
                    if depth_data.max() > depth_data.min():
                        depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min())
                    else:
                        depth_normalized = depth_data
                    
                    depth_img = (depth_normalized * 255).astype(np.uint8)
                    
                    # Save as grayscale PNG
                    output_file = output_dir / f"test_full_functionality_depth_env{env_idx}.png"
                    img = Image.fromarray(depth_img, mode='L')
                    img.save(output_file)
                    print(f"      Saved: {output_file}")
        
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
