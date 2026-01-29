#!/usr/bin/env python3
"""Test OVRTX with very basic materials."""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

source_dir = Path(__file__).parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def test_basic_materials():
    """Test with simplest possible materials."""
    print("=" * 70)
    print("OVRTX Basic Materials Test")
    print("=" * 70)
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create scene with basic displayColor (no materials)
        print("\n1. Creating scene with basic displayColor...")
        test_dir = Path("/tmp/ovrtx_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Scene with displayColor instead of materials
        scene_content = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Sphere "RedSphere"
    {
        color3f[] primvars:displayColor = [(1, 0, 0)]
        double radius = 50
        double3 xformOp:translate = (0, 50, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    
    def Cube "BlueCube"
    {
        color3f[] primvars:displayColor = [(0, 0, 1)]
        double size = 50
        double3 xformOp:translate = (120, 25, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    
    def Mesh "GroundPlane"
    {
        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]
        float3[] extent = [(-500, 0, -500), (500, 0, 500)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        normal3f[] normals = [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
        point3f[] points = [(-500, 0, -500), (500, 0, -500), (500, 0, 500), (-500, 0, 500)]
    }
    
    def DistantLight "SunLight"
    {
        float intensity = 5000
        float3 xformOp:rotateXYZ = (315, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
    
    def DomeLight "SkyLight"
    {
        float inputs:intensity = 2000
    }
}
"""
        scene_path = test_dir / "basic_materials_scene.usda"
        scene_path.write_text(scene_content)
        print(f"   ✓ Scene created: {scene_path}")
        
        # Initialize renderer
        print("\n2. Initializing renderer...")
        cfg = OVRTXRendererCfg(
            height=512,
            width=512,
            num_envs=1,
            num_cameras=1,
            data_types=["rgb"]
        )
        
        renderer = cfg.create_renderer()
        renderer.initialize(usd_scene_path=str(scene_path))
        print("   ✓ Renderer initialized")
        
        # Set camera position
        print("\n3. Positioning camera...")
        camera_positions = torch.tensor([
            [0.0, 100.0, 400.0],
        ], dtype=torch.float32, device="cuda:0")
        
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
        focal_length = 100.0
        intrinsic_matrices = torch.tensor([
            [[focal_length, 0.0, 256.0],
             [0.0, focal_length, 256.0],
             [0.0, 0.0, 1.0]],
        ], dtype=torch.float32, device="cuda:0")
        
        # Render with accumulation
        print("\n4. Rendering...")
        for i in range(10):
            renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Render complete")
        
        # Check output
        print("\n5. Checking output...")
        rgba = renderer._output_data_buffers["rgba"]
        rgba_np = rgba.numpy()
        
        img_data = rgba_np[0]
        r = img_data[:, :, 0]
        g = img_data[:, :, 1]
        b = img_data[:, :, 2]
        a = img_data[:, :, 3]
        
        r_nz = np.count_nonzero(r)
        g_nz = np.count_nonzero(g)
        b_nz = np.count_nonzero(b)
        a_nz = np.count_nonzero(a)
        total = 512 * 512
        
        print(f"   Red:   {r_nz:6d}/{total} ({100*r_nz/total:.1f}%)")
        print(f"   Green: {g_nz:6d}/{total} ({100*g_nz/total:.1f}%)")
        print(f"   Blue:  {b_nz:6d}/{total} ({100*b_nz/total:.1f}%)")
        print(f"   Alpha: {a_nz:6d}/{total} ({100*a_nz/total:.1f}%)")
        
        if r_nz > 0 or g_nz > 0 or b_nz > 0:
            print(f"   ✅ RGB rendering works with basic materials!")
            print(f"   Value ranges: R[{r.min()},{r.max()}] G[{g.min()},{g.max()}] B[{b.min()},{b.max()}]")
        else:
            print(f"   ⚠️  Still black with basic materials")
        
        # Save image
        img = Image.fromarray(img_data, mode='RGBA')
        output_path = output_dir / "test_basic_materials.png"
        img.save(output_path)
        print(f"   📸 Saved: {output_path}")
        
        # Dump scene
        print("\n6. Dumping scene...")
        if renderer._renderer:
            products = renderer._renderer.step(
                render_products={"ovrtx_debug_dump_stage"},
                delta_time=0.0
            )
            if "ovrtx_debug_dump_stage" in products:
                frame = products["ovrtx_debug_dump_stage"].frames[0]
                if "debug" in frame.render_vars:
                    with frame.render_vars["debug"].map(device="cpu") as mapping:
                        usd_dump = mapping.tensor.to_bytes().decode("utf-8")
                        dump_path = output_dir / "basic_materials_dump.usda"
                        dump_path.write_text(usd_dump)
                        print(f"   ✓ Dumped to: {dump_path}")
                        
                        # Check if displayColor survived
                        has_display_color = "displayColor" in usd_dump
                        has_sphere = "RedSphere" in usd_dump
                        print(f"   - RedSphere present: {'✓' if has_sphere else '✗'}")
                        print(f"   - displayColor present: {'✓' if has_display_color else '✗'}")
        
        renderer.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_materials()
    sys.exit(0 if success else 1)
