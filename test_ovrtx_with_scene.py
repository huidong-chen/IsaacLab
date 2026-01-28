#!/usr/bin/env python3
"""Enhanced test for OVRTX renderer with USD scene loading."""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add the source directory to the path
source_dir = Path(__file__).parent.parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def create_simple_cube_usd(output_path: Path):
    """Create a simple USD scene with a cube for testing."""
    usd_content = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World"
{
    def Cube "Cube"
    {
        double size = 1.0
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        color3f[] primvars:displayColor = [(0.8, 0.2, 0.2)]
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def DistantLight "Light"
    {
        float intensity = 3000
        float3 color = (1, 1, 1)
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,5,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
"""
    output_path.write_text(usd_content)
    print(f"Created test USD scene: {output_path}")
    return output_path


def test_with_usd_scene():
    """Test OVRTX renderer with an actual USD scene."""
    print("=" * 70)
    print("OVRTX Renderer Test with USD Scene")
    print("=" * 70)
    
    try:
        # Create test scene
        test_dir = Path("/tmp/ovrtx_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        scene_path = test_dir / "test_cube.usda"
        create_simple_cube_usd(scene_path)
        
        # Create configuration
        print("\n1. Creating OVRTX renderer...")
        cfg = OVRTXRendererCfg(
            height=256,
            width=256,
            num_envs=2,
            num_cameras=1,
            data_types=["rgb", "depth"]
        )
        
        renderer = cfg.create_renderer()
        renderer.initialize()
        print("   ✓ Renderer initialized")
        
        # Load the USD scene
        print(f"\n2. Loading USD scene: {scene_path}")
        usd_handle = renderer.add_usd_scene(str(scene_path))
        print(f"   ✓ Scene loaded (handle: {usd_handle})")
        
        # Test rendering with camera parameters
        print("\n3. Testing render with camera parameters...")
        
        # Create camera parameters
        camera_positions = torch.tensor([
            [0.0, -3.0, 1.5],  # Camera 1: behind and above
            [2.0, -2.0, 1.0],  # Camera 2: side view
        ], dtype=torch.float32, device="cuda:0")
        
        # Looking at origin (simple forward orientation for now)
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
        # Intrinsic matrices
        focal_length = 100.0
        cx, cy = 128.0, 128.0
        intrinsic_matrices = torch.tensor([
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
        ], dtype=torch.float32, device="cuda:0")
        
        print(f"   Camera positions: {camera_positions.shape}")
        print(f"   Camera orientations: {camera_orientations.shape}")
        
        # Render
        renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Render completed")
        
        # Check outputs
        print("\n4. Validating output buffers...")
        outputs = renderer.get_output()
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        for name, buffer in outputs.items():
            print(f"   - {name}: shape={buffer.shape}, dtype={buffer.dtype}")
            
            # Convert to numpy for analysis
            import warp as wp
            data = wp.to_torch(buffer).cpu().numpy()
            
            non_zero = np.count_nonzero(data)
            total = data.size
            mean_val = np.mean(data)
            
            print(f"      Stats: non-zero={non_zero}/{total} ({100*non_zero/total:.2f}%), mean={mean_val:.4f}")
            
            # Save rendered images to files
            if name in ["rgba", "rgb"]:
                for env_idx in range(data.shape[0]):
                    img_data = data[env_idx]  # (H, W, C)
                    
                    # Convert to uint8 if needed
                    if img_data.dtype != np.uint8:
                        img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
                    print(f"px: {img_data}")
                    
                    # Save as PNG
                    output_file = output_dir / f"test_with_scene_{name}_env{env_idx}.png"
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
                    output_file = output_dir / f"test_with_scene_depth_env{env_idx}.png"
                    img = Image.fromarray(depth_img, mode='L')
                    img.save(output_file)
                    print(f"      Saved: {output_file}")
        
        # Test multiple render calls
        print("\n5. Testing multiple render calls...")
        for i in range(3):
            renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Multiple renders successful")
        
        # Test reset
        print("\n6. Testing reset...")
        renderer.reset()
        print("   ✓ Reset successful")
        
        # Clean up
        print("\n7. Cleaning up...")
        renderer.close()
        print("   ✓ Cleanup successful")
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("- USD scene loading: ✓")
        print("- Camera setup: ✓")
        print("- Render pipeline: ✓")
        print("- Output buffers: ✓")
        print("- Resource cleanup: ✓")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    success = test_with_usd_scene()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
