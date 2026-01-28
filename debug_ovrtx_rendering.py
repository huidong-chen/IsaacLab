#!/usr/bin/env python3
"""Debug test to investigate why OVRTX rendering produces black images."""

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


def debug_render_pipeline():
    """Debug the OVRTX rendering pipeline."""
    print("=" * 70)
    print("OVRTX Renderer Debug Test")
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
            num_envs=1,  # Just 1 environment for debugging
            num_cameras=1,
            data_types=["rgb", "depth"]
        )
        
        renderer = cfg.create_renderer()
        
        # Add debug wrapper to the renderer's step method
        original_step = renderer._renderer.step if renderer._renderer else None
        
        def debug_step(*args, **kwargs):
            print("\n[DEBUG] renderer.step() called with:")
            print(f"  render_products: {kwargs.get('render_products', 'NOT SET')}")
            print(f"  delta_time: {kwargs.get('delta_time', 'NOT SET')}")
            
            result = original_step(*args, **kwargs)
            
            print(f"[DEBUG] renderer.step() returned:")
            print(f"  Type: {type(result)}")
            print(f"  Products: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")
            
            if hasattr(result, 'items'):
                for product_name, product in result.items():
                    print(f"\n[DEBUG] Product: {product_name}")
                    print(f"  Type: {type(product)}")
                    print(f"  Frames: {len(product.frames) if hasattr(product, 'frames') else 'N/A'}")
                    
                    if hasattr(product, 'frames') and len(product.frames) > 0:
                        frame = product.frames[0]
                        print(f"  Frame[0] type: {type(frame)}")
                        print(f"  Render vars: {list(frame.render_vars.keys()) if hasattr(frame, 'render_vars') else 'N/A'}")
                        
                        if hasattr(frame, 'render_vars') and 'LdrColor' in frame.render_vars:
                            render_var = frame.render_vars['LdrColor']
                            print(f"  LdrColor type: {type(render_var)}")
                            
                            with render_var.map(device="cuda") as mapping:
                                import warp as wp
                                data = wp.from_dlpack(mapping.tensor)
                                data_np = wp.to_torch(data).cpu().numpy()
                                print(f"  LdrColor shape: {data_np.shape}")
                                print(f"  LdrColor dtype: {data_np.dtype}")
                                print(f"  LdrColor min/max: {data_np.min()}/{data_np.max()}")
                                print(f"  LdrColor mean: {data_np.mean():.4f}")
                                print(f"  LdrColor non-zero: {np.count_nonzero(data_np)}/{data_np.size}")
            
            return result
        
        if renderer._renderer and original_step:
            renderer._renderer.step = debug_step
        
        renderer.initialize()
        print("   ✓ Renderer initialized")
        
        # Load the USD scene
        print(f"\n2. Loading USD scene: {scene_path}")
        usd_handle = renderer.add_usd_scene(str(scene_path))
        print(f"   ✓ Scene loaded (handle: {usd_handle})")
        
        # Test rendering with camera parameters
        print("\n3. Testing render with camera parameters...")
        
        # Create camera parameters - camera looking at origin from behind
        camera_positions = torch.tensor([
            [0.0, -3.0, 1.5],  # Behind and above the cube
        ], dtype=torch.float32, device="cuda:0")
        
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
        # Intrinsic matrices
        focal_length = 100.0
        cx, cy = 128.0, 128.0
        intrinsic_matrices = torch.tensor([
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
        ], dtype=torch.float32, device="cuda:0")
        
        print(f"   Camera position: {camera_positions[0].cpu().numpy()}")
        print(f"   Camera orientation: {camera_orientations[0].cpu().numpy()}")
        
        # Render
        print("\n4. Calling renderer.render()...")
        renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Render completed")
        
        # Check outputs
        print("\n5. Checking output buffers...")
        outputs = renderer.get_output()
        
        for name, buffer in outputs.items():
            import warp as wp
            data = wp.to_torch(buffer).cpu().numpy()
            
            print(f"\n   {name}:")
            print(f"     Shape: {data.shape}")
            print(f"     Dtype: {data.dtype}")
            print(f"     Min/Max: {data.min()}/{data.max()}")
            print(f"     Mean: {data.mean():.6f}")
            print(f"     Non-zero: {np.count_nonzero(data)}/{data.size} ({100*np.count_nonzero(data)/data.size:.2f}%)")
            
            if name == "rgba":
                # Check each channel separately
                for i, channel_name in enumerate(['R', 'G', 'B', 'A']):
                    channel_data = data[0, :, :, i]
                    print(f"     {channel_name} channel: min={channel_data.min()}, max={channel_data.max()}, mean={channel_data.mean():.6f}")
        
        # Clean up
        print("\n6. Cleaning up...")
        renderer.close()
        print("   ✓ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the debug test."""
    success = debug_render_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
