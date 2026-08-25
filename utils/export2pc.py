import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
import trimesh
from OCC.Extend.DataExchange import write_stl_file
from trimesh.sample import sample_surface

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD


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


def sample_surface_with_normals(shape, n_points, temporary_stl_path):
    """Sample paired surface points and face normals from an OCC shape."""
    try:
        write_stl_file(shape, temporary_stl_path)
        mesh = trimesh.load(temporary_stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("the tessellated CAD shape does not contain a mesh")

        points, face_indices = sample_surface(mesh, n_points)
        normals = mesh.face_normals[face_indices]
        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)

        if points.shape != (n_points, 3):
            raise ValueError(f"unexpected sampled point shape: {points.shape}")
        if normals.shape != points.shape:
            raise ValueError(f"unexpected sampled normal shape: {normals.shape}")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
            raise ValueError("sampled surface contains non-finite values")
        return points, normals
    finally:
        if os.path.exists(temporary_stl_path):
            os.remove(temporary_stl_path)


def _has_training_surface_format(output_path, output_format):
    """Check whether an existing NumPy output contains points and normals."""
    try:
        if output_format == "npy":
            array = np.load(output_path, allow_pickle=True)
            if array.shape != () or array.dtype != object:
                return False
            sample = array.item()
            if not isinstance(sample, dict):
                return False
            points = sample.get("points")
            normals = sample.get("normals")
        elif output_format == "npz":
            with np.load(output_path) as sample:
                if "points" not in sample or "normals" not in sample:
                    return False
                points = sample["points"]
                normals = sample["normals"]
        else:
            return True
    except (EOFError, OSError, ValueError):
        return False

    try:
        return (
            isinstance(points, np.ndarray)
            and isinstance(normals, np.ndarray)
            and np.issubdtype(points.dtype, np.number)
            and np.issubdtype(normals.dtype, np.number)
            and points.ndim == 2
            and points.shape[1:] == (3,)
            and normals.shape == points.shape
            and points.shape[0] > 0
            and np.all(np.isfinite(points))
            and np.all(np.isfinite(normals))
        )
    except TypeError:
        return False


def save_surface(points, normals, output_path, output_format):
    """Atomically save a sampled surface in the requested format."""
    temporary_output_path = f"{output_path}.part.{os.getpid()}"
    try:
        if output_format == "npy":
            with open(temporary_output_path, "wb") as output_file:
                np.save(
                    output_file,
                    {"points": points, "normals": normals},
                    allow_pickle=True,
                )
        elif output_format == "npz":
            with open(temporary_output_path, "wb") as output_file:
                np.savez_compressed(output_file, points=points, normals=normals)
        elif output_format == "ply":
            trimesh.PointCloud(points).export(
                temporary_output_path, file_type="ply"
            )
        else:
            raise ValueError(f"unsupported output format: {output_format}")
        os.replace(temporary_output_path, output_path)
    finally:
        if os.path.exists(temporary_output_path):
            os.remove(temporary_output_path)


def process_one(task):
    json_path, raw_data, save_root, normalize, output_format, n_points = task
    relative_id = os.path.splitext(os.path.relpath(json_path, raw_data))[0]
    save_path = os.path.join(save_root, relative_id)
    output_path = save_path + f".{output_format}"
    output_exists = os.path.exists(output_path)

    if output_exists and _has_training_surface_format(output_path, output_format):
        return "skipped", relative_id, None

    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)

        temporary_stl_dir = os.path.join(save_root, ".export2pc_tmp")
        os.makedirs(temporary_stl_dir, exist_ok=True)
        temporary_stl_path = os.path.join(
            temporary_stl_dir,
            f"{os.path.basename(relative_id)}_{os.getpid()}.stl",
        )
        points, normals = sample_surface_with_normals(
            shape, n_points, temporary_stl_path
        )
        if normalize:
            points = point_normalize(points)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_surface(points, normals, output_path, output_format)
        status = "repaired" if output_exists else "exported"
        return status, relative_id, None
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
    counts = {"exported": 0, "repaired": 0, "skipped": 0, "failed": 0}
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
                    f"repaired={counts['repaired']} skipped={counts['skipped']} "
                    f"failed={counts['failed']}",
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

