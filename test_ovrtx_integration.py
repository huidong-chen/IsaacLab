#!/usr/bin/env python3
"""Test script to verify OVRTX renderer integration."""

import sys
from pathlib import Path

# Add the source directory to the path
source_dir = Path(__file__).parent.parent / "source" / "isaaclab"
sys.path.insert(0, str(source_dir))

from isaaclab.renderer import OVRTXRendererCfg, get_renderer_class


def test_ovrtx_renderer_registration():
    """Test that OVRTX renderer is properly registered."""
    print("Testing OVRTX renderer registration...")
    
    # Check if renderer class can be retrieved
    renderer_cls = get_renderer_class("ov_rtx")
    
    if renderer_cls is None:
        print("❌ FAILED: OVRTX renderer class not found")
        return False
    
    print(f"✓ OVRTX renderer class found: {renderer_cls}")
    return True


def test_ovrtx_renderer_config():
    """Test that OVRTX renderer config can be created."""
    print("\nTesting OVRTX renderer configuration...")
    
    try:
        cfg = OVRTXRendererCfg(
            height=512,
            width=512,
            num_envs=1,
            num_cameras=1,
            data_types=["rgb", "depth"]
        )
        print(f"✓ Config created: {cfg}")
        print(f"  - Renderer type: {cfg.renderer_type}")
        print(f"  - Resolution: {cfg.width}x{cfg.height}")
        print(f"  - Num envs: {cfg.num_envs}")
        return True
    except Exception as e:
        print(f"❌ FAILED to create config: {e}")
        return False


def test_ovrtx_renderer_instantiation():
    """Test that OVRTX renderer can be instantiated."""
    print("\nTesting OVRTX renderer instantiation...")
    
    try:
        cfg = OVRTXRendererCfg(
            height=256,
            width=256,
            num_envs=1,
            num_cameras=1,
            data_types=["rgb", "depth"]
        )
        
        renderer_cls = get_renderer_class("ov_rtx")
        renderer = renderer_cls(cfg)
        
        print(f"✓ Renderer instantiated: {renderer}")
        print(f"  - Type: {type(renderer)}")
        
        # Try to initialize
        print("\n  Attempting to initialize renderer...")
        renderer.initialize()
        print("  ✓ Renderer initialized successfully")
        
        # Clean up
        renderer.close()
        print("  ✓ Renderer closed successfully")
        
        return True
    except Exception as e:
        print(f"❌ FAILED to instantiate/initialize renderer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("OVRTX Renderer Integration Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Registration", test_ovrtx_renderer_registration()))
    results.append(("Configuration", test_ovrtx_renderer_config()))
    results.append(("Instantiation", test_ovrtx_renderer_instantiation()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    print("=" * 60)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
