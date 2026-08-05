import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    {
        "name": "LiDAR SLAM",
        "script": "main.py",
        "output": lambda track: f"mid2dataset/{track}/result/lidar_poses.csv",
    },
    {
        "name": "Camera Pose",
        "script": os.path.join("process", "camera_pose.py"),
        "output": lambda track: f"mid2dataset/{track}/result/camera_pose.csv",
    },
    {
        "name": "Color Mapping",
        "script": os.path.join("process", "color_mapping.py"),
        "output": lambda track: f"mid2dataset/{track}/result/{track}.pcd",
    },
]


def run_step(step, track):
    env = os.environ.copy()
    env["TARGET"] = track

    print(f"\n[{track}] >>> {step['name']}", flush=True)
    result = subprocess.run(
        [sys.executable, step["script"]],
        cwd=SCRIPT_DIR,
        env=env,
    )

    if result.returncode != 0:
        print(f"[{track}] FAILED: {step['name']} (exit code {result.returncode})", flush=True)
        return False

    output_path = os.path.join(SCRIPT_DIR, step["output"](track))
    if not os.path.exists(output_path):
        print(f"[{track}] FAILED: {step['name']} — output not found: {output_path}", flush=True)
        return False

    print(f"[{track}] OK: {step['name']} → {step['output'](track)}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="LiDAR SLAM full pipeline")
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["Track1", "Track2", "Track3"],
        choices=["Track1", "Track2", "Track3"],
        metavar="TRACK",
        help="Tracks to process (default: Track1 Track2 Track3)",
    )
    args = parser.parse_args()

    summary = {}

    for track in args.tracks:
        print(f"\n{'='*50}", flush=True)
        print(f" Processing {track}", flush=True)
        print(f"{'='*50}", flush=True)

        track_results = []
        for step in STEPS:
            ok = run_step(step, track)
            track_results.append((step["name"], ok))
            if not ok:
                remaining = STEPS[STEPS.index(step) + 1:]
                for skipped in remaining:
                    track_results.append((skipped["name"], None))
                break

        summary[track] = track_results

    print(f"\n{'='*50}", flush=True)
    print(" Summary", flush=True)
    print(f"{'='*50}", flush=True)
    for track, results in summary.items():
        print(f"\n  {track}:")
        for step_name, ok in results:
            status = "OK" if ok is True else ("SKIP" if ok is None else "FAIL")
            print(f"    [{status:4s}] {step_name}")


if __name__ == "__main__":
    main()
