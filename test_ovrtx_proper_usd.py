#!/usr/bin/env python3
"""Test OVRTX with a proper USD scene following the working example pattern."""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

source_dir = Path(__file__).parent.parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def create_proper_usd_scene(output_path: Path):
    """Create a proper USD scene with materials, lights, and render settings."""
    usd_content = """#usda 1.0
(
    customLayerData = {
        dictionary renderSettings = {
            bool "rtx:post:histogram:enabled" = 1
        }
    }
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "World"
{
    def Sphere "RedSphere" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        float3[] extent = [(-50, -50, -50), (50, 50, 50)]
        rel material:binding = </World/Materials/RedMaterial>
        double radius = 50
        double3 xformOp:translate = (0, 50, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "BlueCube" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        float3[] extent = [(-25, -25, -25), (25, 25, 25)]
        rel material:binding = </World/Materials/BlueMaterial>
        double size = 50
        double3 xformOp:translate = (100, 25, -50)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Mesh "Plane" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        float3[] extent = [(-200, 0, -200), (200, 0, 200)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </World/Materials/GroundMaterial>
        normal3f[] normals = [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)] (
            interpolation = "faceVarying"
        )
        point3f[] points = [(-200, 0, -200), (200, 0, -200), (200, 0, 200), (-200, 0, 200)]
    }

    def DistantLight "DistantLight" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float angle = 0.53
        color3f color = (1, 1, 1)
        float intensity = 3000
        float3 xformOp:rotateXYZ = (315, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }

    def DomeLight "DomeLight" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:intensity = 1000
        double3 xformOp:rotateXYZ = (270, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }

    def Xform "Materials"
    {
        def Material "RedMaterial"
        {
            token outputs:mdl:displacement.connect = </World/Materials/RedMaterial/Shader.outputs:out>
            token outputs:mdl:surface.connect = </World/Materials/RedMaterial/Shader.outputs:out>
            token outputs:mdl:volume.connect = </World/Materials/RedMaterial/Shader.outputs:out>

            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.9, 0.1, 0.1)
                float inputs:metallic_constant = 0
                float inputs:reflection_roughness_constant = 0.5
                token outputs:out
            }
        }

        def Material "BlueMaterial"
        {
            token outputs:mdl:displacement.connect = </World/Materials/BlueMaterial/Shader.outputs:out>
            token outputs:mdl:surface.connect = </World/Materials/BlueMaterial/Shader.outputs:out>
            token outputs:mdl:volume.connect = </World/Materials/BlueMaterial/Shader.outputs:out>

            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.1, 0.2, 0.9)
                float inputs:metallic_constant = 0
                float inputs:reflection_roughness_constant = 0.3
                token outputs:out
            }
        }

        def Material "GroundMaterial"
        {
            token outputs:mdl:displacement.connect = </World/Materials/GroundMaterial/Shader.outputs:out>
            token outputs:mdl:surface.connect = </World/Materials/GroundMaterial/Shader.outputs:out>
            token outputs:mdl:volume.connect = </World/Materials/GroundMaterial/Shader.outputs:out>

            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.7, 0.7, 0.7)
                float inputs:reflection_roughness_constant = 0.8
                token outputs:out
            }
        }
    }
}

def "Render"
{
    def RenderSettings "OmniverseGlobalRenderSettings" (
        prepend apiSchemas = ["OmniRtxSettingsGlobalRtAdvancedAPI_1", "OmniRtxSettingsGlobalPtAdvancedAPI_1"]
    )
    {
    }

    def "Vars"
    {
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
"""
    output_path.write_text(usd_content)
    print(f"Created proper USD scene: {output_path}")
    return output_path


def test_with_proper_usd():
    """Test rendering with a properly structured USD scene."""
    print("=" * 70)
    print("OVRTX Test: Proper USD Scene with Materials")
    print("=" * 70)
    
    try:
        # Create proper scene
        test_dir = Path("/tmp/ovrtx_proper_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        scene_path = test_dir / "proper_scene.usda"
        create_proper_usd_scene(scene_path)
        
        # Create configuration
        print("\n1. Creating OVRTX renderer...")
        cfg = OVRTXRendererCfg(
            height=512,
            width=512,
            num_envs=1,
            num_cameras=1,
            data_types=["rgb", "depth"]
        )
        
        renderer = cfg.create_renderer()
        renderer.initialize()
        print("   ✓ Renderer initialized")
        
        # Load the USD scene
        print(f"\n2. Loading proper USD scene: {scene_path}")
        usd_handle = renderer.add_usd_scene(str(scene_path))
        print(f"   ✓ Scene loaded (handle: {usd_handle})")
        
        # Test rendering with camera parameters
        print("\n3. Rendering with camera looking at scene...")
        
        # Camera positioned to look at the scene
        # Position from example: (297.7, 297.7, 297.7)
        # Scaled down for our smaller scene
        camera_positions = torch.tensor([
            [300.0, 300.0, 300.0],
        ], dtype=torch.float32, device="cuda:0")
        
        # Looking toward origin
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
        # Intrinsic matrices
        focal_length = 100.0
        cx, cy = 256.0, 256.0
        intrinsic_matrices = torch.tensor([
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
        ], dtype=torch.float32, device="cuda:0")
        
        # Render
        renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Render completed")
        
        # Check outputs and save images
        print("\n4. Checking output and saving images...")
        outputs = renderer.get_output()
        
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        import warp as wp
        
        for name, buffer in outputs.items():
            data = wp.to_torch(buffer).cpu().numpy()
            
            print(f"\n   {name}:")
            print(f"     Shape: {data.shape}")
            print(f"     Dtype: {data.dtype}")
            print(f"     Min/Max: {data.min()}/{data.max()}")
            print(f"     Mean: {data.mean():.6f}")
            print(f"     Non-zero: {np.count_nonzero(data)}/{data.size}")
            
            if name == "rgba":
                # Check each channel
                img_data = data[0]
                for i, ch in enumerate(['R', 'G', 'B', 'A']):
                    channel = img_data[:, :, i]
                    print(f"     {ch}: min={channel.min()}, max={channel.max()}, mean={channel.mean():.4f}, non-zero={np.count_nonzero(channel)}")
                
                # Save
                if img_data.dtype != np.uint8:
                    img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(img_data, mode='RGBA')
                output_file = output_dir / "proper_usd_test_rgba.png"
                img.save(output_file)
                print(f"     Saved: {output_file}")
            
            elif name == "rgb":
                img_data = data[0]
                if img_data.dtype != np.uint8:
                    img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(img_data, mode='RGB')
                output_file = output_dir / "proper_usd_test_rgb.png"
                img.save(output_file)
                print(f"     Saved: {output_file}")
        
        # Cleanup
        print("\n5. Cleaning up...")
        renderer.close()
        print("   ✓ Cleanup successful")
        
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETE")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_proper_usd()
    sys.exit(0 if success else 1)
