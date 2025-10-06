import argparse, os
from src.utils.video_io import extract_frames

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frames = extract_frames(args.video, args.out_dir, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out_dir}")
