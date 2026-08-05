import os
import struct
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from gsplat.rendering import rasterization

class StandaloneRenderer:
    """Renderer for RGB Novel View Synthesis using Lucid camera intrinsics."""
    
    def __init__(self, ply_path, intrinsics, device="cuda"):
        self.device = torch.device(device)
        self.K = intrinsics.to(self.device)
        self.splats = self.load_ply(ply_path)
        
    def load_ply(self, path):
        """Loads the 14-float binary PLY format used in the midterm project[cite: 3]."""
        print(f"Loading splats from {path}...")
        with open(path, 'rb') as f:
            header = ""
            while "end_header" not in header:
                line = f.readline().decode('ascii')
                header += line
                if "element vertex" in line:
                    num_points = int(line.split()[-1])

            # point_format: 3 means, 3 scales, 4 quats, 1 opacity, 3 sh0[cite: 3]
            point_format = '14f'
            size = struct.calcsize(point_format)
            
            data = []
            for _ in range(num_points):
                data.append(struct.unpack(point_format, f.read(size)))
            
            data = torch.tensor(data, device=self.device)
            
        return nn.ParameterDict({
            "means": nn.Parameter(data[:, 0:3]),
            "scales": nn.Parameter(data[:, 3:6]),
            "quats": nn.Parameter(data[:, 6:10]),
            "opacities": nn.Parameter(data[:, 10:11]),
            "sh0": nn.Parameter(data[:, 11:14].unsqueeze(1)),
        })

    def render_rgb(self, camtoworld, width, height):
        """Standard GS rendering using SH DC components[cite: 3]."""
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)
        
        # Convert SH DC to RGB using standard constant[cite: 3]
        C0 = 0.28209479177387814
        rgb = self.splats["sh0"].squeeze(1) * C0 + 0.5
        
        viewmat = torch.linalg.inv(camtoworld)
        
        # Rasterization (packed=False for consistent 2D grids)[cite: 2]
        features, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=rgb,
            viewmats=viewmat.unsqueeze(0),
            Ks=self.K.unsqueeze(0),
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
        )

        image_rgb = (features[0] + (1.0 - alphas[0]) * 0.0).permute(2, 0, 1)
        return image_rgb

def load_intrinsics_from_json(json_path):
    """Loads camera intrinsics from JSON. Returns (K, width, height)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    fx, fy, cx, cy = data['fx'], data['fy'], data['cx'], data['cy']
    width, height = int(data['width']), int(data['height'])

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]).float()

    return K, width, height

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_num", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Track root directory (e.g. mid2dataset/Track1)")
    parser.add_argument("--ply_file", type=str, default=None)
    parser.add_argument("--pose_list", type=str, default=None)
    parser.add_argument("--frame_list", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    track_root = Path(args.data_dir) if args.data_dir else Path(f"mid2dataset/Track{args.track_num}")

    PLY_FILE = args.ply_file if args.ply_file is not None else str(track_root / "gaussian_reconstruction.ply")
    K_JSON = str(track_root / "result" / "camera_intrinsics.json")
    POSE_LIST_FILE = args.pose_list if args.pose_list is not None else str(track_root / "test_pose_list.txt")
    FRAME_LIST_FILE = args.frame_list if args.frame_list is not None else str(track_root / f"track{args.track_num}_test_frame_list.txt")
    OUTPUT_DIR = args.output_dir if args.output_dir is not None else str(track_root / "test_submission")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load official frame timestamps and poses
    with open(FRAME_LIST_FILE, 'r') as f:
        frame_ids = [line.strip().split('.')[0] for line in f if line.strip()]
    raw_poses = np.loadtxt(POSE_LIST_FILE)
    test_poses = [torch.from_numpy(p.reshape(4, 4)).float() for p in raw_poses]

    # 2. Load Lucid camera intrinsics[cite: 6]
    print(f"Loading camera parameters from {K_JSON}...")
    intrinsics, width, height = load_intrinsics_from_json(K_JSON)
    print(f"[Config] Track {args.track_num}: {width}x{height}")

    # 3. Initialize Renderer
    renderer = StandaloneRenderer(PLY_FILE, intrinsics)

    print(f"Rendering {len(frame_ids)} views...")
    for i, frame_id in enumerate(tqdm(frame_ids)):
        # Ensure the pose is on the same device as the renderer/splats
        current_pose = test_poses[i].to(renderer.device)
        rgb = renderer.render_rgb(current_pose, width, height)
        
        # Convert to 8-bit BGR for OpenCV
        rgb_np = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{frame_id}.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    print(f"\nDone! official renders saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()