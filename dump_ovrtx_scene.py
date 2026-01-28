#!/usr/bin/env python3
"""Dump OVRTX scene to USD for debugging."""

import sys
from pathlib import Path
import torch

source_dir = Path(__file__).parent.parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg


def dump_ovrtx_scene():
    """Dump the OVRTX scene to USD to see what's actually loaded."""
    print("=" * 70)
    print("OVRTX Scene Debug Dump")
    print("=" * 70)
    
    try:
        # Create renderer
        print("\n1. Creating OVRTX renderer...")
        cfg = OVRTXRendererCfg(
            height=256,
            width=256,
            num_envs=2,
            num_cameras=1,
            data_types=["rgb"]
        )
        
        renderer = cfg.create_renderer()
        renderer.initialize()
        print("   ✓ Renderer initialized with cameras")
        
        # Dump immediately after initialization to see what's in the stage
        print("\n2. Dumping scene immediately after init...")
        if renderer._renderer:
            try:
                products = renderer._renderer.step(
                    render_products={"ovrtx_debug_dump_stage"},
                    delta_time=0.0
                )
                
                if "ovrtx_debug_dump_stage" in products:
                    frame = products["ovrtx_debug_dump_stage"].frames[0]
                    if "debug" in frame.render_vars:
                        with frame.render_vars["debug"].map(device="cpu") as mapping:
                            usd_dump = mapping.tensor.to_bytes().decode("utf-8")
                            
                            # Save to file
                            output_dir = Path("test_output")
                            output_dir.mkdir(exist_ok=True)
                            dump_file = output_dir / "ovrtx_scene_dump_after_init.usda"
                            dump_file.write_text(usd_dump, encoding="utf-8")
                            
                            print(f"   ✓ Scene dumped to: {dump_file}")
                            print(f"   File size: {len(usd_dump)} bytes")
                            
                            # Check what's in the scene
                            has_cameras = "/Render/Camera" in usd_dump
                            has_render_products = "/Render/RenderProduct" in usd_dump
                            has_render_scope = 'def "Render"' in usd_dump or 'over "Render"' in usd_dump
                            
                            print(f"\n   Scene contents:")
                            print(f"   - /Render scope: {'✓' if has_render_scope else '✗'}")
                            print(f"   - Cameras: {'✓' if has_cameras else '✗'}")
                            print(f"   - RenderProducts: {'✓' if has_render_products else '✗'}")
                            
                            # Print snippet
                            lines = usd_dump.split('\n')
                            print(f"\n   Showing lines with 'Camera' or 'Render':")
                            for i, line in enumerate(lines[:200]):
                                if 'Camera' in line or 'Render' in line or 'def "' in line or 'over "' in line:
                                    print(f"   {i+1:3d}: {line}")
                            
                            return True
            except Exception as e:
                print(f"   ⚠ Debug dump failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Don't continue with rendering since it will fail
        # Optionally load a GEOMETRY-ONLY scene (no /Render scope!)
        print("\n2. Loading geometry-only USD scene...")
        test_dir = Path("/tmp/ovrtx_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create geometry-only scene (NO /Render scope to avoid conflict with cameras)
        scene_content = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Sphere "TestSphere" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        float3[] extent = [(-50, -50, -50), (50, 50, 50)]
        rel material:binding = </World/Materials/RedMaterial>
        double radius = 50
        double3 xformOp:translate = (0, 50, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    
    def Cube "TestCube" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        float3[] extent = [(-25, -25, -25), (25, 25, 25)]
        rel material:binding = </World/Materials/BlueMaterial>
        double size = 50
        double3 xformOp:translate = (100, 25, 0)
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
    
    def DistantLight "Light"
    {
        float intensity = 3000
        float3 xformOp:rotateXYZ = (315, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
    
    def DomeLight "DomeLight"
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
"""
        scene_path = test_dir / "geometry_only.usda"
        scene_path.write_text(scene_content)
        
        # Load geometry AFTER cameras are set up
        try:
            handle = renderer.add_usd_scene(str(scene_path))
            print(f"   ✓ Scene loaded (handle: {handle})")
        except Exception as e:
            print(f"   ⚠ Scene load failed: {e}")
            print("   Continuing without geometry...")
        
        # Dump scene after geometry load to see what's there
        print("\n3. Dumping scene after geometry load...")
        if renderer._renderer:
            try:
                products = renderer._renderer.step(
                    render_products={"ovrtx_debug_dump_stage"},
                    delta_time=0.0
                )
                
                if "ovrtx_debug_dump_stage" in products:
                    frame = products["ovrtx_debug_dump_stage"].frames[0]
                    if "debug" in frame.render_vars:
                        with frame.render_vars["debug"].map(device="cpu") as mapping:
                            usd_dump = mapping.tensor.to_bytes().decode("utf-8")
                            
                            # Save to file
                            output_dir = Path("test_output")
                            output_dir.mkdir(exist_ok=True)
                            dump_file = output_dir / "ovrtx_scene_dump_with_geometry.usda"
                            dump_file.write_text(usd_dump, encoding="utf-8")
                            
                            print(f"   ✓ Scene dumped to: {dump_file}")
                            print(f"   File size: {len(usd_dump)} bytes")
                            
                            # Check what's in the scene
                            has_world = "/World" in usd_dump or 'def Xform "World"' in usd_dump
                            has_sphere = "TestSphere" in usd_dump
                            has_cube = "TestCube" in usd_dump
                            has_materials = "Materials" in usd_dump
                            has_lights = "Light" in usd_dump or "DomeLight" in usd_dump
                            
                            print(f"\n   Scene contents:")
                            print(f"   - World: {'✓' if has_world else '✗'}")
                            print(f"   - Sphere: {'✓' if has_sphere else '✗'}")
                            print(f"   - Cube: {'✓' if has_cube else '✗'}")
                            print(f"   - Materials: {'✓' if has_materials else '✗'}")
                            print(f"   - Lights: {'✓' if has_lights else '✗'}")
            except Exception as e:
                print(f"   ⚠ Debug dump failed: {e}")
        
        return True
        test_dir = Path("/tmp/ovrtx_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simple scene (without /Render scope to avoid conflict)
        scene_content = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Sphere "TestSphere" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double radius = 50
        double3 xformOp:translate = (0, 50, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    
    def Cube "TestCube"
    {
        double size = 50
        double3 xformOp:translate = (100, 25, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    
    def DistantLight "Light"
    {
        float intensity = 3000
        float3 xformOp:rotateXYZ = (315, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
"""
        scene_path = test_dir / "debug_scene.usda"
        scene_path.write_text(scene_content)
        
        # Load geometry AFTER cameras are set up
        # Note: This will fail if scene has /Render scope
        try:
            handle = renderer.add_usd_scene(str(scene_path))
            print(f"   ✓ Scene loaded (handle: {handle})")
        except Exception as e:
            print(f"   ⚠ Scene load failed (expected): {e}")
            print("   Continuing with camera-only scene...")
        
        # Set up camera transforms
        print("\n3. Setting camera transforms...")
        camera_positions = torch.tensor([
            [300.0, 300.0, 300.0],
            [200.0, 200.0, 200.0],
        ], dtype=torch.float32, device="cuda:0")
        
        camera_orientations = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda:0")
        
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
        
        # Update camera transforms (this prepares them but doesn't render)
        renderer.render(camera_positions, camera_orientations, intrinsic_matrices)
        print("   ✓ Camera transforms updated")
        
        # Dump the scene using OVRTX's debug feature
        print("\n4. Dumping scene to USD...")
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
                        
                        # Save to file
                        output_dir = Path("test_output")
                        output_dir.mkdir(exist_ok=True)
                        dump_file = output_dir / "ovrtx_scene_dump.usda"
                        dump_file.write_text(usd_dump, encoding="utf-8")
                        
                        print(f"   ✓ Scene dumped to: {dump_file}")
                        print(f"   File size: {len(usd_dump)} bytes")
                        
                        # Print first few lines
                        lines = usd_dump.split('\n')
                        print(f"\n   First 20 lines of dump:")
                        for i, line in enumerate(lines[:20]):
                            print(f"   {i+1:3d}: {line}")
                        
                        print(f"\n   ... ({len(lines)} total lines)")
                        
                        # Check what's in the scene
                        has_cameras = "/Render/Camera" in usd_dump
                        has_render_products = "/Render/RenderProduct" in usd_dump
                        has_world = "/World" in usd_dump
                        has_geometry = "Sphere" in usd_dump or "Cube" in usd_dump
                        
                        print(f"\n5. Scene contents analysis:")
                        print(f"   - Cameras found: {'✓' if has_cameras else '✗'}")
                        print(f"   - RenderProducts found: {'✓' if has_render_products else '✗'}")
                        print(f"   - World scope found: {'✓' if has_world else '✗'}")
                        print(f"   - Geometry found: {'✓' if has_geometry else '✗'}")
                        
                        return True
        
        print("\n❌ Failed to dump scene")
        return False
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'renderer' in locals():
            renderer.close()
            print("\n✓ Renderer closed")


if __name__ == "__main__":
    success = dump_ovrtx_scene()
    sys.exit(0 if success else 1)
