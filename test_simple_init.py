#!/usr/bin/env python3
"""Simple test to verify scene-first initialization works."""

import sys
from pathlib import Path

source_dir = Path(__file__).parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

print("1. Importing renderer...")
from isaaclab.renderer import OVRTXRendererCfg

print("2. Creating simple USD scene...")
test_dir = Path("/tmp/ovrtx_test")
test_dir.mkdir(parents=True, exist_ok=True)

# Minimal scene with /Render scope for cameras to extend
scene_content = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Sphere "TestSphere"
    {
        double radius = 50
    }
}

def Scope "Render"
{
}
"""
scene_path = test_dir / "simple_scene.usda"
scene_path.write_text(scene_content)
print(f"   Scene created: {scene_path}")

print("3. Creating renderer config...")
cfg = OVRTXRendererCfg(
    height=256,
    width=256,
    num_envs=1,
    num_cameras=1,
    data_types=["rgb"]
)
print("   Config created")

print("4. Creating renderer instance...")
renderer = cfg.create_renderer()
print("   Renderer instance created")

print("5. Initializing with scene...")
try:
    renderer.initialize(usd_scene_path=str(scene_path))
    print("   ✓ Initialization successful!")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("6. Closing renderer...")
renderer.close()
print("   ✓ Done!")
