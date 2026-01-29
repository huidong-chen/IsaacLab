#!/usr/bin/env python3
"""Test OVRTX rendering with geometry after fixing camera setup."""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

source_dir = Path(__file__).parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def test_rendering_with_geometry():
    """Test complete rendering pipeline: cameras + geometry + render."""
    print("=" * 70)
    print("OVRTX Rendering Test with Geometry")
    print("=" * 70)
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Create geometry USD first
        print("\n1. Creating geometry USD...")
        test_dir = Path("/tmp/ovrtx_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create colorful scene with proper materials
        scene_content = """#usda 1.0
(
    defaultPrim = "World"
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
        double3 xformOp:translate = (120, 25, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    
    
    def DistantLight "SunLight"
    {
        float intensity = 5000
        float3 xformOp:rotateXYZ = (315, 45, 0)
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
                float inputs:metallic_constant = 0.1
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
                color3f inputs:diffuse_color_constant = (0.1, 0.3, 0.9)
                float inputs:metallic_constant = 0.2
                float inputs:reflection_roughness_constant = 0.3
                token outputs:out
            }
        }

        def Material "GreenMaterial"
        {
            token outputs:mdl:displacement.connect = </World/Materials/GreenMaterial/Shader.outputs:out>
            token outputs:mdl:surface.connect = </World/Materials/GreenMaterial/Shader.outputs:out>
            token outputs:mdl:volume.connect = </World/Materials/GreenMaterial/Shader.outputs:out>

            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.1, 0.9, 0.2)
                float inputs:metallic_constant = 0.0
                float inputs:reflection_roughness_constant = 0.6
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
                color3f inputs:diffuse_color_constant = (0.6, 0.6, 0.6)
                float inputs:reflection_roughness_constant = 0.9
                token outputs:out
            }
        }
    }
}
"""
        scene_path = test_dir / "colorful_scene.usda"
        scene_path.write_text(scene_content)
        print(f"   ✓ Scene USD created: {scene_path}")
        
        # 2. Create renderer and initialize with geometry scene
        print("\n2. Creating OVRTX renderer with geometry scene...")
        cfg = OVRTXRendererCfg(
            height=512,
            width=512,
            num_envs=2,
            num_cameras=1,
            data_types=["rgb"]
        )
        
        renderer = cfg.create_renderer()
        # Initialize with scene - this loads scene first, then adds cameras as sublayer
        renderer.initialize(usd_scene_path=str(scene_path))
        print("   ✓ Renderer initialized with geometry and cameras")
        
        # 3. Set up camera to look at the scene
        print("\n3. Positioning cameras to view scene...")
        # Camera looking at origin from elevated angle
        camera_positions = torch.tensor([
            [0.0, 50.0, 300.0],  # Environment 0
            [0.0, 0.0, 50.0],  # Environment 1
        ], dtype=torch.float32, device="cuda:0")
        
        # Point cameras toward origin (scene center)
        # For now use identity quaternions, will implement proper lookAt later
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
        # Set up intrinsics for 512x512 render
        focal_length = 100.0
        cx, cy = 256.0, 256.0
        intrinsic_matrices = torch.tensor([
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
            [[focal_length, 0.0, cx],
             [0.0, focal_length, cy],
             [0.0, 0.0, 1.0]],
        ], dtype=torch.float32, device="cuda:0")
        
        print("   ✓ Camera parameters set")
        
        # 4. Render multiple frames for path tracer accumulation
        print("\n4. Rendering scene (with accumulation frames)...")
        for frame_idx in range(10):  # Render 10 frames to accumulate samples
            renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
            if frame_idx == 0:
                print(f"   Frame {frame_idx}: Initial render")
            elif frame_idx == 9:
                print(f"   Frame {frame_idx}: Final accumulated render")
        print("   ✓ Render complete with accumulation")
        
        # 5. Check output
        print("\n5. Analyzing rendered output...")
        rgba = renderer._output_data_buffers["rgba"]
        
        # Convert to numpy for analysis
        rgba_np = rgba.numpy()
        
        for env_idx in range(2):
            img_data = rgba_np[env_idx]
            
            # Extract channels
            r_channel = img_data[:, :, 0]
            g_channel = img_data[:, :, 1]
            b_channel = img_data[:, :, 2]
            a_channel = img_data[:, :, 3]
            
            # Count non-zero pixels
            r_nonzero = np.count_nonzero(r_channel)
            g_nonzero = np.count_nonzero(g_channel)
            b_nonzero = np.count_nonzero(b_channel)
            a_nonzero = np.count_nonzero(a_channel)
            total_pixels = 512 * 512
            
            print(f"\n   Environment {env_idx}:")
            print(f"   - Red channel:   {r_nonzero:6d}/{total_pixels} ({100*r_nonzero/total_pixels:.1f}%) non-zero")
            print(f"   - Green channel: {g_nonzero:6d}/{total_pixels} ({100*g_nonzero/total_pixels:.1f}%) non-zero")
            print(f"   - Blue channel:  {b_nonzero:6d}/{total_pixels} ({100*b_nonzero/total_pixels:.1f}%) non-zero")
            print(f"   - Alpha channel: {a_nonzero:6d}/{total_pixels} ({100*a_nonzero/total_pixels:.1f}%) non-zero")
            
            if r_nonzero > 0 or g_nonzero > 0 or b_nonzero > 0:
                print(f"   ✅ RGB has visible content!")
                print(f"   - Red range:   [{r_channel.min()}, {r_channel.max()}]")
                print(f"   - Green range: [{g_channel.min()}, {g_channel.max()}]")
                print(f"   - Blue range:  [{b_channel.min()}, {b_channel.max()}]")
            else:
                print(f"   ⚠️  RGB channels still black")
            
            # Save image
            img = Image.fromarray(img_data, mode='RGBA')
            output_path = output_dir / f"test_geometry_render_env{env_idx}.png"
            img.save(output_path)
            print(f"   📸 Saved: {output_path}")
        
        # 6. Dump final scene to see what was loaded
        print("\n6. Dumping final scene for inspection...")
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
                        dump_file = output_dir / "final_scene_with_geometry.usda"
                        dump_file.write_text(usd_dump, encoding="utf-8")
                        
                        # Check contents
                        has_world = "/World" in usd_dump
                        has_sphere = "RedSphere" in usd_dump
                        has_cube = "BlueCube" in usd_dump
                        has_cylinder = "GreenCylinder" in usd_dump
                        has_materials = "RedMaterial" in usd_dump
                        
                        print(f"   ✓ Scene dumped: {dump_file} ({len(usd_dump)} bytes)")
                        print(f"   Contents:")
                        print(f"   - World scope: {'✓' if has_world else '✗'}")
                        print(f"   - RedSphere: {'✓' if has_sphere else '✗'}")
                        print(f"   - BlueCube: {'✓' if has_cube else '✗'}")
                        print(f"   - GreenCylinder: {'✓' if has_cylinder else '✗'}")
                        print(f"   - Materials: {'✓' if has_materials else '✗'}")
        
        print("\n" + "=" * 70)
        print("✓ Test Complete!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'renderer' in locals():
            renderer.close()


if __name__ == "__main__":
    success = test_rendering_with_geometry()
    sys.exit(0 if success else 1)
