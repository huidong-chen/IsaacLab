#!/usr/bin/env python3
"""Test OVRTX with a complete USD scene including cameras."""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

source_dir = Path(__file__).parent.parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def create_complete_usd_scene(output_path: Path):
    """Create a complete USD scene with geometry, lights, AND cameras."""
    usd_content = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World"
{
    def Cube "RedCube"
    {
        double size = 2.0
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        color3f[] primvars:displayColor = [(1.0, 0.0, 0.0)]
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def SphereLight "Key"
    {
        float intensity = 50000
        float radius = 0.5
        color3f color = (1, 1, 1)
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (5,5,10,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def DistantLight "Sun"
    {
        float intensity = 1000
        color3f color = (1, 0.95, 0.9)
        matrix4d xformOp:transform = ( (0.707,0.707,0,0), (-0.707,0.707,0,0), (0,0,1,0), (0,0,5,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}

def Scope "Render"
{
    def Camera "TestCamera"
    {
        float focalLength = 24.0
        float horizontalAperture = 20.955
        float verticalAperture = 15.2908
        token projection = "perspective"
        float2 clippingRange = (0.1, 10000.0)
        matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,-5,60,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    
    def RenderProduct "TestRenderProduct"
    {
        rel camera = </Render/TestCamera>
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</Render/TestRenderProduct/LdrColor>]
        uniform int2 resolution = (512, 512)
        
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
"""
    output_path.write_text(usd_content)
    print(f"Created complete USD scene: {output_path}")
    return output_path


def test_complete_usd():
    """Test rendering with a complete USD scene."""
    print("=" * 70)
    print("OVRTX Test: Complete USD Scene")
    print("=" * 70)
    
    # Create complete scene
    test_dir = Path("/tmp/ovrtx_complete_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    scene_path = test_dir / "complete_scene.usda"
    create_complete_usd_scene(scene_path)
    
    # Initialize OVRTX directly
    print("\nInitializing OVRTX...")
    import ovrtx
    from ovrtx import Renderer, RendererConfig
    
    config = RendererConfig(
        startup_options={
            "crashreporter/dumpDir": "/tmp",
            "log/file": "/tmp/ovrtx_complete_test.log",
        }
    )
    renderer = Renderer(config)
    print("✓ OVRTX initialized")
    
    # Load complete scene
    print(f"\nLoading scene: {scene_path}")
    handle = renderer.add_usd(str(scene_path))
    print("✓ Scene loaded")
    
    # Render using the camera in the USD
    print("\nRendering...")
    products = renderer.step(
        render_products={"/Render/TestRenderProduct"},
        delta_time=1.0/60.0
    )
    print("✓ Rendered")
    
    # Extract image
    print("\nExtracting render output...")
    if "/Render/TestRenderProduct" in products:
        product = products["/Render/TestRenderProduct"]
        if len(product.frames) > 0:
            frame = product.frames[0]
            if "LdrColor" in frame.render_vars:
                import warp as wp
                with frame.render_vars["LdrColor"].map(device="cuda") as mapping:
                    data = wp.from_dlpack(mapping.tensor)
                    data_np = wp.to_torch(data).cpu().numpy()
                    
                    print(f"  Shape: {data_np.shape}")
                    print(f"  Dtype: {data_np.dtype}")
                    print(f"  Min/Max: {data_np.min()}/{data_np.max()}")
                    print(f"  Mean: {data_np.mean():.4f}")
                    
                    # Check RGB channels
                    if len(data_np.shape) == 3 and data_np.shape[2] >= 3:
                        for i, name in enumerate(['R', 'G', 'B', 'A']):
                            if i < data_np.shape[2]:
                                channel = data_np[:, :, i]
                                print(f"  {name}: min={channel.min()}, max={channel.max()}, mean={channel.mean():.4f}")
                    
                    # Save image
                    output_dir = Path("test_output")
                    output_dir.mkdir(exist_ok=True)
                    
                    img_data = data_np
                    if img_data.dtype != np.uint8:
                        img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
                    
                    if img_data.shape[2] == 4:
                        img = Image.fromarray(img_data, mode='RGBA')
                    else:
                        img = Image.fromarray(img_data, mode='RGB')
                    
                    output_file = output_dir / "ovrtx_complete_usd_test.png"
                    img.save(output_file)
                    print(f"\n✓ Saved: {output_file}")
    
    # Cleanup
    renderer.remove_usd(handle)
    print("\n✓ Test complete")
    
    return True


if __name__ == "__main__":
    try:
        test_complete_usd()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
