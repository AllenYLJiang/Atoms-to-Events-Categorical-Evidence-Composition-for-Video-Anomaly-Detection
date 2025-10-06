import os, cv2, math, glob
from typing import List, Tuple

def extract_frames(video_path: str, out_dir: str, fps: float = 1.0) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    vfps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(vfps / fps))) if vfps > 0 else 1
    frames = []
    idx = 0
    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fn = os.path.join(out_dir, f"frame_{save_idx:06d}.jpg")
            cv2.imwrite(fn, frame)
            frames.append(fn)
            save_idx += 1
        idx += 1
    cap.release()
    return frames
