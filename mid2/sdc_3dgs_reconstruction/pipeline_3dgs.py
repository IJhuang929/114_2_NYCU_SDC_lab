"""
Unified 3DGS pipeline.

Usage:
    python pipeline_3dgs.py --track1
    python pipeline_3dgs.py --track2
    python pipeline_3dgs.py --track3

Optional:
    python pipeline_3dgs.py --track1 --skip_train
    python pipeline_3dgs.py --track1 --skip_mask
    python pipeline_3dgs.py --track1 --epochs 30 --batch_size 2

Example individual commands (if not using pipeline_3dgs.py):
    python train.py --track_num 2
    python generate_test_pose.py --track_num 2
    python generate_submission.py --track_num 2
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_ROOT = SCRIPT_DIR / "mid2dataset"


def run(script_name: str, *args):
    script = SCRIPT_DIR / script_name

    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [sys.executable, str(script)] + [str(a) for a in args]

    print(f"\n{'=' * 70}")
    print("Running:")
    print(" ".join(cmd))
    print(f"{'=' * 70}")

    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def count_files(folder: Path, suffixes):
    if not folder.exists():
        return 0

    suffixes = tuple(s.lower() for s in suffixes)

    return len([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    ])


def check_data_dir(data_dir: Path, track_num: int):
    track_name = f"Track{track_num}"

    required_files = [
        data_dir / "result" / "camera_gt_pose.txt",
        data_dir / "result" / "camera_intrinsics.json",
        data_dir / "result" / f"{track_name}.pcd",
    ]

    required_dirs = [
        data_dir / "data" / "image",
    ]

    ok = True

    print("\n[Check dataset]")
    print(f"data_dir: {data_dir}")

    for p in required_files:
        if p.exists():
            print(f"  [OK] {p.name}")
        else:
            print(f"  [MISS] {p}")
            ok = False

    for p in required_dirs:
        if p.exists():
            print(f"  [OK] {p.name}/")
        else:
            print(f"  [MISS] {p}")
            ok = False

    image_count = count_files(data_dir / "data" / "image", [".jpg", ".jpeg", ".png"])
    mask_count = count_files(data_dir / "data" / "sky_masks", [".png", ".jpg", ".jpeg"])

    print(f"  images    : {image_count}")
    print(f"  sky_masks : {mask_count}")

    pose_file = data_dir / "result" / "camera_gt_pose.txt"
    if pose_file.exists():
        with open(pose_file, "r") as f:
            pose_count = sum(1 for _ in f)
        print(f"  poses     : {pose_count}")

        if pose_count != image_count:
            print(f"  [WARN] pose count != image count")
    else:
        pose_count = 0

    if image_count == 0:
        print("  [ERROR] No images found.")
        ok = False

    return ok


def make_zip(track_dir: Path):
    test_submission_dir = track_dir / "test_submission"
    submission_csv = track_dir / "submission.csv"
    zip_path = track_dir / "submission.zip"

    if not test_submission_dir.exists():
        raise FileNotFoundError(f"test_submission/ not found: {test_submission_dir}")

    if not submission_csv.exists():
        raise FileNotFoundError(f"submission.csv not found: {submission_csv}")

    png_files = sorted(test_submission_dir.glob("*.png"))

    if len(png_files) == 0:
        raise RuntimeError(f"No PNG files found in {test_submission_dir}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in png_files:
            zf.write(img, img.name)

        zf.write(submission_csv, "submission.csv")

    print(f"[OK] Created zip: {zip_path}")
    print(f"[OK] PNG count: {len(png_files)}")


def convert_camera_pose_csv(data_dir: Path):
    csv_path = data_dir / "result" / "camera_pose.csv"
    txt_path = data_dir / "result" / "camera_gt_pose.txt"

    if txt_path.exists():
        return

    if not csv_path.exists():
        raise FileNotFoundError(
            f"camera_gt_pose.txt not found and camera_pose.csv also missing:\n{csv_path}"
        )

    print(f"[Auto] Generating camera_gt_pose.txt from camera_pose.csv ...")

    with open(csv_path, "r") as f_in, open(txt_path, "w") as f_out:
        header = f_in.readline()  # skip header
        for line in f_in:
            parts = line.strip().split(",")
            f_out.write(",".join(parts[1:]) + "\n")  # drop timestamp column

    print(f"[OK] Written: {txt_path}")


def run_track(track_num: int, skip_train: bool, skip_mask: bool, epochs: int, batch_size: int):
    data_dir = DATA_ROOT / f"Track{track_num}"
    track_dir = data_dir
    output_dir = track_dir

    test_list = track_dir / f"track{track_num}_test_frame_list.txt"
    test_pose_list = track_dir / "test_pose_list.txt"
    test_submission_dir = track_dir / "test_submission"

    vis_ply = output_dir / "visualization.ply"
    trained_ply = output_dir / "gaussian_reconstruction.ply"

    track_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# Track {track_num}")
    print(f"# data_dir       : {data_dir}")
    print(f"# track_dir      : {track_dir}")
    print(f"# output_dir     : {output_dir}")
    print(f"# test_list      : {test_list}")
    print(f"# test_pose_list : {test_pose_list}")
    print(f"{'#' * 70}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    if not test_list.exists():
        raise FileNotFoundError(
            f"Test list not found: {test_list}\n"
            f"Expected file name: track{track_num}_test_frame_list.txt"
        )

    convert_camera_pose_csv(data_dir)

    if not check_data_dir(data_dir, track_num):
        raise RuntimeError(f"Dataset check failed: {data_dir}")

    # 1. 檢查 camera pose + point cloud
    run(
        "test_camera_poses.py",
        "--data_dir", data_dir,
        "--output", vis_ply,
    )

    # 2. 產 sky masks
    mask_dir = data_dir / "data" / "sky_masks"
    image_count = count_files(data_dir / "data" / "image", [".jpg", ".jpeg", ".png"])
    mask_count = count_files(mask_dir, [".png", ".jpg", ".jpeg"])

    if skip_mask:
        print("[Skip] sky mask generation")
    elif mask_count == image_count and mask_count > 0:
        print(f"[OK] sky masks already exist: {mask_dir}")
    else:
        run(
            "generate_sky_mask.py",
            "--data_dir", data_dir,
        )

    # 3. train
    if skip_train:
        print("[Skip] training")
        if not trained_ply.exists():
            raise FileNotFoundError(
                f"skip_train=True but trained model not found:\n{trained_ply}"
            )
    else:
        run(
            "train.py",
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--track_num", track_num,
            "--epochs", epochs,
            "--batch_size", batch_size,
        )

    # 4. 產生 test poses
    run(
        "generate_test_pose.py",
        "--data_dir", data_dir,
        "--test_list", test_list,
        "--output", test_pose_list,
        "--track_num", track_num,
    )

    # 5. render submission images
    run(
        "generate_submission.py",
        "--ply_file", trained_ply,
        "--data_dir", data_dir,
        "--pose_list", test_pose_list,
        "--frame_list", test_list,
        "--output_dir", test_submission_dir,
        "--track_num", track_num,
    )

    # 6. zip
    make_zip(track_dir)

    print(f"\n[Done] Track {track_num}")
    print(f"submission zip: {track_dir / 'submission.zip'}")


def main():
    parser = argparse.ArgumentParser(description="Unified 3DGS pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--track1", action="store_true", help="Run Track1")
    group.add_argument("--track2", action="store_true", help="Run Track2")
    group.add_argument("--track3", action="store_true", help="Run Track3")

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_mask", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    if args.track1:
        track_num = 1
    elif args.track2:
        track_num = 2
    elif args.track3:
        track_num = 3
    else:
        raise RuntimeError("No track selected")

    run_track(
        track_num=track_num,
        skip_train=args.skip_train,
        skip_mask=args.skip_mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
