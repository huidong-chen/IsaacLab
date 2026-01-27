#!/usr/bin/env python3
"""Example usage of the OVRTX renderer in Isaac Lab.

This example demonstrates:
1. Creating an OVRTX renderer configuration
2. Instantiating the renderer
3. Initializing it
4. Basic usage patterns
"""

import sys
from pathlib import Path

# Add source directory to path if running standalone
source_dir = Path(__file__).parent / "source" / "isaaclab"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir.parent))

from isaaclab.renderer import OVRTXRendererCfg


def example_basic_usage():
    """Basic usage example of OVRTX renderer."""
    print("=" * 70)
    print("OVRTX Renderer Basic Usage Example")
    print("=" * 70)
    
    # Step 1: Create configuration
    print("\n1. Creating OVRTX renderer configuration...")
    cfg = OVRTXRendererCfg(
        height=512,
        width=512,
        num_envs=1,
        num_cameras=1,
        data_types=["rgb", "depth"]
    )
    print(f"   ✓ Configuration created:")
    print(f"     - Renderer type: {cfg.renderer_type}")
    print(f"     - Resolution: {cfg.width}x{cfg.height}")
    print(f"     - Environments: {cfg.num_envs}")
    print(f"     - Cameras: {cfg.num_cameras}")
    print(f"     - Data types: {cfg.data_types}")
    
    # Step 2: Create renderer instance
    print("\n2. Creating renderer instance...")
    renderer = cfg.create_renderer()
    print(f"   ✓ Renderer created: {type(renderer).__name__}")
    
    # Step 3: Initialize renderer
    print("\n3. Initializing renderer...")
    renderer.initialize()
    print("   ✓ Renderer initialized successfully")
    
    # Step 4: Get output buffers info
    print("\n4. Output buffer information:")
    outputs = renderer.get_output()
    for name, buffer in outputs.items():
        print(f"   - {name}: shape={buffer.shape}, dtype={buffer.dtype}")
    
    # Step 5: Reset renderer
    print("\n5. Resetting renderer...")
    renderer.reset()
    print("   ✓ Renderer reset successfully")
    
    # Step 6: Clean up
    print("\n6. Cleaning up...")
    renderer.close()
    print("   ✓ Renderer closed successfully")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def example_multi_environment():
    """Example with multiple environments."""
    print("\n" + "=" * 70)
    print("OVRTX Renderer Multi-Environment Example")
    print("=" * 70)
    
    # Create configuration for multiple environments
    print("\nCreating configuration for 4 environments...")
    cfg = OVRTXRendererCfg(
        height=256,
        width=256,
        num_envs=4,
        num_cameras=1,
        data_types=["rgb", "rgba", "depth"]
    )
    
    renderer = cfg.create_renderer()
    renderer.initialize()
    
    outputs = renderer.get_output()
    print(f"\nOutput buffers for {cfg.num_envs} environments:")
    for name, buffer in outputs.items():
        print(f"   - {name}: shape={buffer.shape}, dtype={buffer.dtype}")
    
    renderer.close()
    print("\n✓ Multi-environment example completed")


def example_high_resolution():
    """Example with high-resolution rendering."""
    print("\n" + "=" * 70)
    print("OVRTX Renderer High-Resolution Example")
    print("=" * 70)
    
    # Create configuration for high-resolution rendering
    print("\nCreating configuration for 1920x1080 resolution...")
    cfg = OVRTXRendererCfg(
        height=1080,
        width=1920,
        num_envs=1,
        num_cameras=1,
        data_types=["rgb", "depth"]
    )
    
    renderer = cfg.create_renderer()
    renderer.initialize()
    
    outputs = renderer.get_output()
    print(f"\nOutput buffers at {cfg.width}x{cfg.height}:")
    for name, buffer in outputs.items():
        memory_mb = (buffer.size * buffer.dtype.itemsize) / (1024 * 1024)
        print(f"   - {name}: shape={buffer.shape}, memory={memory_mb:.2f} MB")
    
    renderer.close()
    print("\n✓ High-resolution example completed")


def main():
    """Run all examples."""
    try:
        # Run basic example
        example_basic_usage()
        
        # Run multi-environment example
        example_multi_environment()
        
        # Run high-resolution example
        example_high_resolution()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
