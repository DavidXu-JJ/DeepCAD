import argparse
import glob
import json
import multiprocessing
import os
import sys

import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_ply_file, write_step_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD, vec2CADsolid
from file_utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="source folder")
    parser.add_argument(
        "--form",
        type=str,
        default="h5",
        choices=["h5", "json"],
        help="input file format",
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="export files starting from this index"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="number of shapes to export; -1 exports all shapes",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="skip shapes rejected by the OpenCascade validity analyzer",
    )
    parser.add_argument("-o", "--outputs", type=str, default=None, help="save folder")
    parser.add_argument(
        "--output_form",
        type=str,
        default="step",
        choices=["step", "ply"],
        help="output file format",
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


def export_one(task):
    path, src_dir, save_dir, input_form, output_form, check_validity = task
    relative_path = os.path.relpath(path, src_dir)
    save_path = os.path.join(
        save_dir, os.path.splitext(relative_path)[0] + f".{output_form}"
    )

    if os.path.exists(save_path):
        return "skipped", path, None

    try:
        if input_form == "h5":
            with h5py.File(path, "r") as fp:
                out_vec = fp["out_vec"][:].astype(np.float32)
            out_shape = vec2CADsolid(out_vec)
        else:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

        if check_validity and not BRepCheck_Analyzer(out_shape).IsValid():
            return "invalid", path, None

        ensure_dir(os.path.dirname(save_path))
        if output_form == "ply":
            write_ply_file(out_shape, save_path)
        else:
            write_step_file(out_shape, save_path)
        return "exported", path, None
    except Exception as error:
        return "failed", path, f"{type(error).__name__}: {error}"


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")

    src_dir = os.path.abspath(args.src)
    save_dir = (
        os.path.abspath(args.outputs)
        if args.outputs
        else src_dir + f"_{args.output_form}"
    )
    ensure_dir(save_dir)

    input_paths = sorted(
        glob.glob(os.path.join(src_dir, "**", f"*.{args.form}"), recursive=True)
    )
    if args.num != -1:
        input_paths = input_paths[args.idx : args.idx + args.num]
    elif args.idx:
        input_paths = input_paths[args.idx :]

    tasks = (
        (
            path,
            src_dir,
            save_dir,
            args.form,
            args.output_form,
            args.filter,
        )
        for path in input_paths
    )
    counts = {"exported": 0, "skipped": 0, "invalid": 0, "failed": 0}
    total = len(input_paths)
    print(
        f"Exporting {total} file(s) from {src_dir} to {save_dir} "
        f"with {args.workers} worker(s)"
    )

    # OpenCascade is safer with independently spawned interpreters than with
    # forked processes that inherit an initialized native-library state.
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(
            export_one, tasks, chunksize=args.chunksize
        )
        for completed, (status, path, error) in enumerate(results, start=1):
            counts[status] += 1
            if status == "failed":
                print(f"[failed] {path}: {error}", file=sys.stderr)
            elif status == "invalid":
                print(f"[invalid] {path}", file=sys.stderr)
            if completed % 100 == 0 or completed == total:
                print(
                    f"[{completed}/{total}] exported={counts['exported']} "
                    f"skipped={counts['skipped']} invalid={counts['invalid']} "
                    f"failed={counts['failed']}",
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
