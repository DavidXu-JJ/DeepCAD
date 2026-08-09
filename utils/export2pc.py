import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD


def point_normalize(points):
    """Normalize an N x 3 point cloud into a centered [-1, 1] cube."""
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    scale = np.max(max_vals - min_vals)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("point cloud has a zero or invalid bounding-box size")

    scaled_points = points / scale
    min_vals2 = np.min(scaled_points, axis=0)
    max_vals2 = np.max(scaled_points, axis=0)
    scaled_points = scaled_points * 2.0 - (min_vals2 + max_vals2)
    return scaled_points


def save_points(points, save_path, output_format):
    if output_format == "npy":
        np.save(save_path + ".npy", points)
    elif output_format == "npz":
        np.savez_compressed(save_path + ".npz", points=points)
    elif output_format == "ply":
        trimesh.PointCloud(points).export(save_path + ".ply")
    else:
        raise ValueError(f"unsupported output format: {output_format}")


def process_one(task):
    json_path, raw_data, save_root, normalize, output_format, n_points = task
    relative_id = os.path.splitext(os.path.relpath(json_path, raw_data))[0]
    save_path = os.path.join(save_root, relative_id)
    output_path = save_path + f".{output_format}"

    if os.path.exists(output_path):
        return "skipped", relative_id, None

    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)

        # CADsolid2pc uses this name for an intermediate STL file. Including
        # the PID prevents workers from colliding on the same temporary path.
        temp_name = f"{os.path.basename(relative_id)}_{os.getpid()}"
        points = CADsolid2pc(shape, n_points, temp_name)
        if normalize:
            points = point_normalize(points)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_points(points, save_path, output_format)
        return "exported", relative_id, None
    except Exception as error:
        return "failed", relative_id, f"{type(error).__name__}: {error}"


def iter_json_paths(raw_data):
    for root, dirnames, filenames in os.walk(raw_data):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".json"):
                yield os.path.join(root, filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalize each point cloud to a centered [-1, 1] cube",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="npy",
        choices=["npy", "npz", "ply"],
        help="output file format",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=100000,
        help="number of points sampled from each CAD surface",
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="start processing at this file index"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="number of files to process; -1 processes all remaining files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="number of worker processes (default: min(8, CPU count))",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="number of files dispatched to each worker at a time",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")
    if args.n_points < 1:
        raise ValueError("--n_points must be at least 1")
    if args.idx < 0 or args.num < -1:
        raise ValueError("--idx must be >= 0 and --num must be >= -1")

    data_root = os.path.abspath(args.data_root)
    raw_data = os.path.join(data_root, "json")
    save_root = os.path.join(data_root, "pcd")
    if not os.path.isdir(raw_data):
        raise FileNotFoundError(f"JSON input directory does not exist: {raw_data}")
    os.makedirs(save_root, exist_ok=True)

    input_paths = list(iter_json_paths(raw_data))
    if args.num == -1:
        input_paths = input_paths[args.idx :]
    else:
        input_paths = input_paths[args.idx : args.idx + args.num]

    tasks = (
        (
            path,
            raw_data,
            save_root,
            args.normalize,
            args.output_format,
            args.n_points,
        )
        for path in input_paths
    )
    counts = {"exported": 0, "skipped": 0, "failed": 0}
    total = len(input_paths)
    print(
        f"Sampling {total} CAD file(s) from {raw_data} into {save_root} "
        f"with {args.workers} worker(s), {args.n_points} points each"
    )

    # Spawn gives each process an independent OpenCascade native-library state.
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(
            process_one, tasks, chunksize=args.chunksize
        )
        for completed, (status, relative_id, error) in enumerate(results, start=1):
            counts[status] += 1
            if status == "failed":
                print(f"[failed] {relative_id}: {error}", file=sys.stderr)
            if completed % 100 == 0 or completed == total:
                print(
                    f"[{completed}/{total}] exported={counts['exported']} "
                    f"skipped={counts['skipped']} failed={counts['failed']}",
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
