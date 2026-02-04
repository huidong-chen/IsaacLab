#!/usr/bin/env python3
"""
Create side-by-side comparison video of rendered images from two directories.

This script:
1. Finds matching frames from two source directories
2. Combines them side-by-side
3. Creates an MP4 video with H.264 encoding (Slack-compatible) or animated GIF

By default, uses tiled images (frame_XXXXXX_tiled.png) which show all environments
in a grid. Use --no-tiled to compare individual environment images instead.

Usage:
    # Compare Newton and OVRTX (default directories)
    python create_comparison_video.py
    
    # Compare OVRTX RGB and depth renders
    python create_comparison_video.py --dir1 ovrtx_rendered_images --dir2 ovrtx_rendered_images_depth \
        --label1 "OVRTX RGB" --label2 "OVRTX Depth" --output ovrtx_comparison.mp4
    
    # Custom filename patterns per directory
    python create_comparison_video.py \
        --dir1 images1 --pattern1 "frame_*_rgb.png" \
        --dir2 images2 --pattern2 "frame_*_depth.png"
    
    # Create half-size video (smaller file, faster)
    python create_comparison_video.py --half-size
    
    # Create animated GIF (lower FPS recommended)
    python create_comparison_video.py --output comparison.gif --gif-fps 10
    
    # Create both MP4 and GIF, half size
    python create_comparison_video.py --half-size --gif --gif-fps 10
    
    # Custom scaling (e.g., 75% size)
    python create_comparison_video.py --scale 0.75
    
    # Use individual environment images
    python create_comparison_video.py --no-tiled [--env-id 0] [--fps 30]
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def convert_to_h264(input_path: Path, output_path: Path, crf: int = 23):
    """Convert video to H.264 format using ffmpeg.
    
    Args:
        input_path: Input video file (OpenCV output)
        output_path: Output video file (H.264 encoded)
        crf: Quality setting (18-28, lower=better, default: 23)
    """
    print(f"\n  Converting to H.264 format...")
    
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', str(crf),
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-y',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Show file sizes
            input_size = input_path.stat().st_size / (1024 * 1024)
            output_size = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ H.264 conversion complete")
            print(f"    Input:  {input_size:.2f} MB")
            print(f"    Output: {output_size:.2f} MB")
            if output_size < input_size:
                savings = (1 - output_size / input_size) * 100
                print(f"    Saved:  {savings:.1f}%")
            return True
        else:
            print(f"  ⚠ FFmpeg conversion failed")
            return False
    except Exception as e:
        print(f"  ⚠ FFmpeg conversion error: {e}")
        return False


def create_gif_from_frames(frames: List[np.ndarray], output_path: Path, fps: int = 10, optimize: bool = True):
    """Create an animated GIF from a list of frames.
    
    Args:
        frames: List of numpy arrays (RGB format) representing frames
        output_path: Output GIF file path
        fps: Frames per second for the GIF (default: 10, common for GIFs)
        optimize: Whether to optimize the GIF (reduces file size, default: True)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n  Creating GIF...")
    
    try:
        # Convert numpy arrays to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Calculate duration per frame in milliseconds
        duration_ms = int(1000 / fps)
        
        # Save as animated GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,  # 0 means infinite loop
            optimize=optimize,
        )
        
        # Show file size
        output_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ GIF creation complete")
        print(f"    Output: {output_size:.2f} MB")
        print(f"    Frames: {len(frames)}")
        print(f"    FPS: {fps}")
        
        return True
    except Exception as e:
        print(f"  ⚠ GIF creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_frame_info(filename: str) -> Tuple[int, int] | None:
    """Parse frame number and env ID from filename.
    
    Expected formats:
        - frame_XXXXXX_tiled.png
        - frame_XXXXXX_env_YYYY.png
        - [prefix_]frame_XXXXXX_tiled.png (e.g., depth_frame_000153_tiled.png)
        - [prefix_]frame_XXXXXX_env_YYYY.png
    
    Args:
        filename: Image filename
        
    Returns:
        Tuple of (frame_number, env_id) or None if parsing fails
        For tiled images, env_id will be -1
    """
    # Try tiled format first (with optional prefix)
    match = re.match(r'(?:.*_)?frame_(\d{6})_tiled\.png', filename)
    if match:
        return int(match.group(1)), -1
    
    # Try individual env format (with optional prefix)
    match = re.match(r'(?:.*_)?frame_(\d{6})_env_(\d{4})\.png', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def find_matching_images(dir1: Path, dir2: Path, use_tiled: bool = True, env_id: int = 0, pattern1: str | None = None, pattern2: str | None = None) -> List[Tuple[int, Path, Path]]:
    """Find matching image pairs from both directories.
    
    Args:
        dir1: Path to first directory
        dir2: Path to second directory
        use_tiled: Whether to use tiled images (default: True)
        env_id: Environment ID to filter when not using tiled images (default: 0)
        pattern1: Custom glob pattern for dir1 files (overrides use_tiled/env_id)
        pattern2: Custom glob pattern for dir2 files (overrides use_tiled/env_id)
        
    Returns:
        List of tuples (frame_number, dir1_path, dir2_path) sorted by frame number
    """
    # Determine the pattern to use for dir1
    if pattern1:
        glob_pattern1 = pattern1
    elif use_tiled:
        glob_pattern1 = 'frame_*_tiled.png'
    else:
        glob_pattern1 = f'frame_*_env_{env_id:04d}.png'
    
    # Determine the pattern to use for dir2
    if pattern2:
        glob_pattern2 = pattern2
    elif use_tiled:
        glob_pattern2 = 'frame_*_tiled.png'
    else:
        glob_pattern2 = f'frame_*_env_{env_id:04d}.png'
    
    # Get all images from dir1
    dir1_images = {}
    if dir1.exists():
        print(f"pattern1: {glob_pattern1}")
        for img_path in dir1.glob(glob_pattern1):
            info = parse_frame_info(img_path.name)
            if info:
                frame_num, _ = info
                dir1_images[frame_num] = img_path
    
        print(f"dir1 images: {dir1_images}")

    # Get all images from dir2
    dir2_images = {}
    if dir2.exists():
        print(f"pattern2: {glob_pattern2}")
        for img_path in dir2.glob(glob_pattern2):
            print(f"found img_path: {img_path.name}")
            info = parse_frame_info(img_path.name)
            if info:
                frame_num, _ = info
                dir2_images[frame_num] = img_path
        print(f"dir2 images: {dir2_images}")
    
    # Find matching pairs
    matching_frames = []
    all_frames = sorted(set(dir1_images.keys()) & set(dir2_images.keys()))
    
    for frame_num in all_frames:
        matching_frames.append((
            frame_num,
            dir1_images[frame_num],
            dir2_images[frame_num]
        ))
    
    return matching_frames


def create_side_by_side_image(path1: Path, path2: Path, add_labels: bool = True, label1: str = "Left", label2: str = "Right") -> np.ndarray:
    """Create side-by-side comparison image.
    
    Args:
        path1: Path to first image
        path2: Path to second image
        add_labels: Whether to add text labels to images
        label1: Label text for first image (default: "Left")
        label2: Label text for second image (default: "Right")
        
    Returns:
        Combined image as numpy array (RGB)
    """
    # Load images
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    
    # Ensure same size (resize to match if needed)
    if img1.size != img2.size:
        # Resize to the larger dimensions
        max_width = max(img1.width, img2.width)
        max_height = max(img1.height, img2.height)
        img1 = img1.resize((max_width, max_height), Image.Resampling.LANCZOS)
        img2 = img2.resize((max_width, max_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    np1 = np.array(img1)
    np2 = np.array(img2)
    
    # Add labels if requested
    if add_labels:
        # Add text labels at the top
        label_height = 40
        labeled1 = np.ones((np1.shape[0] + label_height, np1.shape[1], 3), dtype=np.uint8) * 255
        labeled2 = np.ones((np2.shape[0] + label_height, np2.shape[1], 3), dtype=np.uint8) * 255
        
        # Copy images below labels
        labeled1[label_height:, :, :] = np1
        labeled2[label_height:, :, :] = np2
        
        # Convert to BGR for OpenCV text rendering
        bgr1 = cv2.cvtColor(labeled1, cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(labeled2, cv2.COLOR_RGB2BGR)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 0, 0)  # Black text
        
        cv2.putText(bgr1, label1, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(bgr2, label2, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Convert back to RGB
        labeled1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        labeled2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        
        np1 = labeled1
        np2 = labeled2
    
    # Add separator line
    separator = np.ones((np1.shape[0], 4, 3), dtype=np.uint8) * 128  # Gray line
    
    # Concatenate side by side
    combined = np.concatenate([np1, separator, np2], axis=1)
    
    return combined


def create_comparison_video(
    dir1: Path,
    dir2: Path,
    output_path: Path,
    fps: int = 30,
    use_tiled: bool = True,
    env_id: int = 0,
    add_labels: bool = True,
    label1: str = "Left",
    label2: str = "Right",
    use_h264: bool = True,
    crf: int = 23,
    output_gif: bool = False,
    gif_fps: int = 10,
    scale_factor: float = 1.0,
    pattern1: str | None = None,
    pattern2: str | None = None,
):
    """Create side-by-side comparison video.
    
    Args:
        dir1: Path to first directory
        dir2: Path to second directory
        output_path: Output video file path
        fps: Frames per second for output video
        use_tiled: Whether to use tiled images (default: True)
        env_id: Environment ID to use when not using tiled images
        add_labels: Whether to add text labels
        label1: Label for first directory images (default: "Left")
        label2: Label for second directory images (default: "Right")
        use_h264: Whether to convert to H.264 using ffmpeg (default: True)
        crf: Quality for H.264 encoding (18-28, default: 23)
        output_gif: Whether to also output as GIF (default: False)
        gif_fps: Frames per second for GIF output (default: 10)
        scale_factor: Scale output by this factor (e.g., 0.5 = half size, default: 1.0)
        pattern1: Custom glob pattern for dir1 files (overrides use_tiled/env_id)
        pattern2: Custom glob pattern for dir2 files (overrides use_tiled/env_id)
    """
    print("="*80)
    print("Creating Comparison Video")
    print("="*80)
    
    # Find matching images
    if pattern1 or pattern2:
        print(f"\nSearching for images with custom patterns:")
        if pattern1:
            print(f"  Dir 1 pattern: {pattern1}")
        if pattern2:
            print(f"  Dir 2 pattern: {pattern2}")
    elif use_tiled:
        print(f"\nSearching for tiled images...")
    else:
        print(f"\nSearching for images (env {env_id})...")
    
    matching_frames = find_matching_images(dir1, dir2, use_tiled=use_tiled, env_id=env_id, pattern1=pattern1, pattern2=pattern2)
    
    if len(matching_frames) == 0:
        print("\n✗ No matching frames found!")
        print(f"  Dir 1: {dir1} (exists: {dir1.exists()})")
        print(f"  Dir 2: {dir2} (exists: {dir2.exists()})")
        if pattern1 or pattern2:
            print(f"  Dir 1 pattern: {pattern1 or 'default'}")
            print(f"  Dir 2 pattern: {pattern2 or 'default'}")
        elif use_tiled:
            print(f"  Looking for: frame_*_tiled.png")
        else:
            print(f"  Looking for: frame_*_env_{env_id:04d}.png")
        return
    
    print(f"✓ Found {len(matching_frames)} matching frames")
    print(f"  Frame range: {matching_frames[0][0]} to {matching_frames[-1][0]}")
    
    # Create first combined image to get dimensions
    first_combined = create_side_by_side_image(
        matching_frames[0][1],
        matching_frames[0][2],
        add_labels=add_labels,
        label1=label1,
        label2=label2
    )
    
    # Apply scaling if requested
    if scale_factor != 1.0:
        original_height, original_width = first_combined.shape[:2]
        scaled_width = int(original_width * scale_factor)
        scaled_height = int(original_height * scale_factor)
        first_combined = cv2.resize(first_combined, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    
    height, width = first_combined.shape[:2]
    
    print(f"\nVideo settings:")
    print(f"  Resolution: {width}x{height}")
    if scale_factor != 1.0:
        print(f"  Scale factor: {scale_factor}x (reduced from original)")
    print(f"  FPS: {fps}")
    print(f"  Image type: {'Tiled' if use_tiled else f'Individual (env {env_id})'}")
    if output_gif:
        print(f"  GIF output: Enabled (FPS: {gif_fps})")
    
    # Determine output format
    is_gif_only = output_path.suffix.lower() == '.gif'
    
    if is_gif_only:
        # GIF-only mode: collect all frames and create GIF
        print(f"\nOutput format: GIF only")
        print(f"  Output: {output_path}")
        
        # Collect all frames
        print(f"\nProcessing frames...")
        all_frames = []
        for frame_num, path1, path2 in tqdm(matching_frames, desc="Loading frames"):
            combined_rgb = create_side_by_side_image(path1, path2, add_labels=add_labels, label1=label1, label2=label2)
            
            # Apply scaling if requested
            if scale_factor != 1.0:
                combined_rgb = cv2.resize(combined_rgb, (width, height), interpolation=cv2.INTER_AREA)
            
            all_frames.append(combined_rgb)
        
        # Create GIF
        if create_gif_from_frames(all_frames, output_path, fps=gif_fps, optimize=True):
            print(f"\n✓ GIF output: {output_path}")
            print(f"  Duration: {len(matching_frames) / gif_fps:.2f} seconds")
            print(f"  Total frames: {len(matching_frames)}")
        
    else:
        # Video mode (with optional GIF)
        # Check if we'll use ffmpeg for H.264 encoding
        has_ffmpeg = check_ffmpeg() if use_h264 else False
        
        if use_h264 and has_ffmpeg:
            print(f"  Encoding: H.264 via ffmpeg (CRF {crf})")
            # Create temporary file for OpenCV output
            temp_path = output_path.parent / (output_path.stem + '_temp.mp4')
            print(f"  Temp file: {temp_path}")
            print(f"  Final output: {output_path}")
        else:
            if use_h264 and not has_ffmpeg:
                print("  ⚠ FFmpeg not found, using OpenCV codec")
            print(f"  Output: {output_path}")
            temp_path = None
        
        # Create video writer with mp4v codec (OpenCV)
        working_path = temp_path if temp_path else output_path
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
        video_writer = cv2.VideoWriter(str(working_path), fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("\n✗ Failed to create video writer!")
            return
        
        # Process all frames (and optionally collect for GIF)
        print(f"\nProcessing frames...")
        gif_frames: List[np.ndarray] | None = [] if output_gif else None
        
        for frame_num, path1, path2 in tqdm(matching_frames, desc="Creating video"):
            # Create combined image
            combined_rgb = create_side_by_side_image(path1, path2, add_labels=add_labels, label1=label1, label2=label2)
            
            # Apply scaling if requested
            if scale_factor != 1.0:
                combined_rgb = cv2.resize(combined_rgb, (width, height), interpolation=cv2.INTER_AREA)
            
            # Store for GIF if needed
            if output_gif and gif_frames is not None:
                gif_frames.append(combined_rgb)
            
            # Convert RGB to BGR for OpenCV
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
            
            # Write frame
            video_writer.write(combined_bgr)
        
        # Release video writer
        video_writer.release()
        
        print(f"\n✓ OpenCV video created")
        
        # Convert to H.264 if ffmpeg is available
        if temp_path and has_ffmpeg:
            if convert_to_h264(temp_path, output_path, crf):
                # Remove temporary file
                try:
                    temp_path.unlink()
                    print(f"  ✓ Temporary file removed")
                except Exception:
                    pass
                print(f"\n✓ Final H.264 video: {output_path}")
            else:
                # Keep temporary file as output if conversion failed
                print(f"\n⚠ Using OpenCV version: {temp_path}")
                print(f"  (Rename {temp_path.name} to {output_path.name} manually)")
        else:
            print(f"\n✓ Video output: {output_path}")
        
        print(f"  Duration: {len(matching_frames) / fps:.2f} seconds")
        print(f"  Total frames: {len(matching_frames)}")
        
        # Create GIF if requested
        if output_gif and gif_frames:
            gif_path = output_path.with_suffix('.gif')
            print(f"\nCreating additional GIF output...")
            if create_gif_from_frames(gif_frames, gif_path, fps=gif_fps, optimize=True):
                print(f"✓ GIF output: {gif_path}")
                print(f"  Duration: {len(matching_frames) / gif_fps:.2f} seconds")
        
        if has_ffmpeg:
            print(f"\n💬 Video is Slack-compatible (H.264 baseline profile)")
        else:
            print(f"\n⚠ Install ffmpeg for Slack-compatible videos:")
            print(f"   sudo apt install ffmpeg")
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create side-by-side comparison video of rendered images from two directories'
    )
    parser.add_argument(
        '--dir1',
        type=Path,
        default=Path('newton_rendered_images'),
        help='Path to first directory (default: newton_rendered_images)'
    )
    parser.add_argument(
        '--dir2',
        type=Path,
        default=Path('ovrtx_rendered_images'),
        help='Path to second directory (default: ovrtx_rendered_images)'
    )
    parser.add_argument(
        '--label1',
        type=str,
        default='Newton Warp',
        help='Label for first directory images (default: Newton Warp)'
    )
    parser.add_argument(
        '--label2',
        type=str,
        default='OVRTX',
        help='Label for second directory images (default: OVRTX)'
    )
    parser.add_argument(
        '--pattern1',
        type=str,
        default=None,
        help='Custom glob pattern for dir1 files (e.g., "frame_*_tiled.png"). Overrides --use-tiled and --env-id for dir1.'
    )
    parser.add_argument(
        '--pattern2',
        type=str,
        default=None,
        help='Custom glob pattern for dir2 files (e.g., "frame_*_tiled.png"). Overrides --use-tiled and --env-id for dir2.'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='(Deprecated: use --pattern1 and --pattern2) Custom glob pattern for both directories.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('comparison.mp4'),
        help='Output video file path (default: comparison.mp4). Use .gif extension for GIF-only output.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output video (default: 30)'
    )
    parser.add_argument(
        '--use-tiled',
        action='store_true',
        default=True,
        help='Use tiled images (default: True)'
    )
    parser.add_argument(
        '--no-tiled',
        dest='use_tiled',
        action='store_false',
        help='Use individual environment images instead of tiled'
    )
    parser.add_argument(
        '--env-id',
        type=int,
        default=0,
        help='Environment ID to use when not using tiled images (default: 0)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Disable text labels on images'
    )
    parser.add_argument(
        '--no-h264',
        action='store_true',
        help='Disable H.264 encoding (use OpenCV codec only)'
    )
    parser.add_argument(
        '--crf',
        type=int,
        default=23,
        help='H.264 quality setting (18-28, lower=better, default: 23)'
    )
    parser.add_argument(
        '--gif',
        action='store_true',
        help='Also output as animated GIF (in addition to video)'
    )
    parser.add_argument(
        '--gif-fps',
        type=int,
        default=10,
        help='Frames per second for GIF output (default: 10, lower reduces file size)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale output resolution by this factor (e.g., 0.5 = half size, default: 1.0)'
    )
    parser.add_argument(
        '--half-size',
        action='store_const',
        const=0.5,
        dest='scale',
        help='Shortcut for --scale 0.5 (reduces resolution and file size by ~4x)'
    )
    
    # Add deprecated arguments for backwards compatibility
    parser.add_argument(
        '--newton-dir',
        type=Path,
        dest='dir1',
        help='(Deprecated: use --dir1) Path to first directory'
    )
    parser.add_argument(
        '--ovrtx-dir',
        type=Path,
        dest='dir2',
        help='(Deprecated: use --dir2) Path to second directory'
    )
    
    args = parser.parse_args()
    
    # Handle backwards compatibility for --pattern
    pattern1 = args.pattern1
    pattern2 = args.pattern2
    if args.pattern and not pattern1 and not pattern2:
        # Use single pattern for both directories (deprecated behavior)
        pattern1 = args.pattern
        pattern2 = args.pattern
    
    # Validate CRF range
    if not 0 <= args.crf <= 51:
        print(f"⚠ CRF must be between 0-51 (got {args.crf}), using default 23")
        args.crf = 23
    
    # Validate GIF FPS
    if args.gif_fps < 1:
        print(f"⚠ GIF FPS must be at least 1 (got {args.gif_fps}), using default 10")
        args.gif_fps = 10
    
    # Validate scale factor
    if args.scale <= 0 or args.scale > 2.0:
        print(f"⚠ Scale must be between 0 and 2.0 (got {args.scale}), using default 1.0")
        args.scale = 1.0
    
    # Create comparison video
    create_comparison_video(
        dir1=args.dir1,
        dir2=args.dir2,
        output_path=args.output,
        fps=args.fps,
        use_tiled=args.use_tiled,
        env_id=args.env_id,
        add_labels=not args.no_labels,
        label1=args.label1,
        label2=args.label2,
        use_h264=not args.no_h264,
        crf=args.crf,
        output_gif=args.gif,
        gif_fps=args.gif_fps,
        scale_factor=args.scale,
        pattern1=pattern1,
        pattern2=pattern2,
    )


if __name__ == '__main__':
    main()
