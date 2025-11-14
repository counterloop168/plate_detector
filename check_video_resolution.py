"""
Video Statistics Checker
Check resolution, FPS, duration, and other properties of video files
"""

import sys
import os
import argparse
from datetime import timedelta
import cv2


def format_duration(seconds):
    """Format seconds into HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))


def get_video_stats(video_path):
    """
    Get comprehensive statistics about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video statistics
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Calculate duration
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    # Get codec name
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    # Get file size
    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Calculate bitrate
    bitrate_mbps = (file_size_bytes * 8) / (duration_seconds * 1000000) if duration_seconds > 0 else 0
    
    # Get aspect ratio
    gcd = lambda a, b: a if b == 0 else gcd(b, a % b)
    divisor = gcd(width, height)
    aspect_ratio = f"{width//divisor}:{height//divisor}"
    
    cap.release()
    
    stats = {
        'filename': os.path.basename(video_path),
        'filepath': video_path,
        'resolution': f"{width}x{height}",
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
        'duration_seconds': duration_seconds,
        'duration_formatted': format_duration(duration_seconds),
        'codec': fourcc,
        'file_size_mb': file_size_mb,
        'bitrate_mbps': bitrate_mbps,
        'aspect_ratio': aspect_ratio
    }
    
    return stats


def print_video_stats(stats):
    """
    Print video statistics in a formatted way
    
    Args:
        stats: Dictionary with video statistics
    """
    if not stats:
        return
    
    print("=" * 70)
    print("VIDEO STATISTICS")
    print("=" * 70)
    print(f"File Name:        {stats['filename']}")
    print(f"File Path:        {stats['filepath']}")
    print(f"File Size:        {stats['file_size_mb']:.2f} MB")
    print("-" * 70)
    print(f"Resolution:       {stats['resolution']}")
    print(f"Width:            {stats['width']} pixels")
    print(f"Height:           {stats['height']} pixels")
    print(f"Aspect Ratio:     {stats['aspect_ratio']}")
    print("-" * 70)
    print(f"Frame Rate:       {stats['fps']:.2f} FPS")
    print(f"Total Frames:     {stats['total_frames']:,}")
    print(f"Duration:         {stats['duration_formatted']} ({stats['duration_seconds']:.2f} seconds)")
    print("-" * 70)
    print(f"Codec:            {stats['codec']}")
    print(f"Bitrate:          {stats['bitrate_mbps']:.2f} Mbps")
    print("=" * 70)


def main():
    """Main function to check video statistics"""
    parser = argparse.ArgumentParser(description='Check video file statistics')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--json', action='store_true', help='Output as JSON format')
    
    args = parser.parse_args()
    
    # Get video statistics
    stats = get_video_stats(args.video_path)
    
    if stats is None:
        sys.exit(1)
    
    # Print or output as JSON
    if args.json:
        import json
        print(json.dumps(stats, indent=2))
    else:
        print_video_stats(stats)


if __name__ == "__main__":
    main()
