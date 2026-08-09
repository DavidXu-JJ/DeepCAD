"""Convert DeepCAD JSON sequences to H5 CAD vectors.

Omni-CAD's ``raw_train_test_split.json`` stores base model IDs such as
``0022/00220825`` while ``json/`` contains one or more sequence variants such
as ``0022/00220825_00001.json``. This script expands each selected base ID to
all matching variants before applying the original DeepCAD vectorization.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed


DEEPCAD_ROOT = Path(__file__).resolve().parents[1]
if str(DEEPCAD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPCAD_ROOT))

from cadlib.extrude import CADSequence  # noqa: E402
from cadlib.macro import (  # noqa: E402
    MAX_N_CURVES,
    MAX_N_EXT,
    MAX_N_LOOPS,
    MAX_TOTAL_LEN,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        required=True,
        help="dataset root containing json/ and raw_train_test_split.json",
    )
    parser.add_argument(
        "--raw_dir",
        help="JSON input directory (default: <data_root>/json)",
    )
    parser.add_argument(
        "--split_file",
        help="split JSON path (default: <data_root>/raw_train_test_split.json)",
    )
    parser.add_argument(
        "--output_dir",
        help="H5 output directory (default: <data_root>/vec)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="start index in the selected list of base IDs",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="number of base IDs to process; -1 processes all",
    )
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="number of JSON files submitted to joblib per batch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing H5 vectors instead of skipping them",
    )
    return parser.parse_args()


def load_split_ids(split_path: Path, split_name: str):
    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    if not isinstance(split_data, dict):
        raise ValueError("Split file must contain a JSON object: {}".format(split_path))

    if split_name == "all":
        split_names = [
            name for name in ("train", "validation", "test") if name in split_data
        ]
    else:
        if split_name not in split_data:
            raise ValueError(
                "Split '{}' is not present in {}".format(split_name, split_path)
            )
        split_names = [split_name]

    selected_ids = []
    seen_ids = set()
    for name in split_names:
        ids = split_data[name]
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("Split '{}' must be a list of strings.".format(name))
        for item_id in ids:
            if item_id not in seen_ids:
                selected_ids.append(item_id)
                seen_ids.add(item_id)
    return selected_ids, split_names


def select_base_ids(base_ids, idx: int, num: int):
    if idx < 0:
        raise ValueError("--idx must be greater than or equal to zero")
    if num < -1:
        raise ValueError("--num must be -1 or greater than or equal to zero")
    if num == -1:
        return base_ids[idx:]
    return base_ids[idx : idx + num]


def collect_json_paths(raw_dir: Path, selected_ids):
    """Resolve full JSON sequence paths for complete IDs or base model IDs."""
    if len(selected_ids) <= 1000:
        json_paths = []
        for item_id in selected_ids:
            exact_path = raw_dir / (item_id + ".json")
            if exact_path.is_file():
                json_paths.append(exact_path)
            else:
                json_paths.extend(sorted(raw_dir.glob(item_id + "_*.json")))
        return sorted(set(json_paths))

    selected_id_set = set(selected_ids)
    json_paths = []
    for json_path in raw_dir.rglob("*.json"):
        sample_id = json_path.relative_to(raw_dir).with_suffix("").as_posix()
        base_id = sample_id.rsplit("_", 1)[0]
        if sample_id in selected_id_set or base_id in selected_id_set:
            json_paths.append(json_path)
    return sorted(json_paths)


def process_one(json_path_string, raw_dir_string, output_dir_string, overwrite):
    json_path = Path(json_path_string)
    raw_dir = Path(raw_dir_string)
    output_dir = Path(output_dir_string)
    sample_id = json_path.relative_to(raw_dir).with_suffix("").as_posix()
    save_path = output_dir / (sample_id + ".h5")

    if save_path.exists() and not overwrite:
        return "skipped", sample_id, None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        cad_seq.numericalize()
        cad_vec = cad_seq.to_vector(
            MAX_N_EXT,
            MAX_N_LOOPS,
            MAX_N_CURVES,
            MAX_TOTAL_LEN,
            pad=False,
        )

        if cad_vec is None:
            return "failed", sample_id, "vectorization returned None"
        if cad_vec.shape[0] > MAX_TOTAL_LEN:
            return "too_long", sample_id, int(cad_vec.shape[0])

        save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = save_path.with_name(
            "{}.tmp.{}".format(save_path.name, os.getpid())
        )
        try:
            with h5py.File(temporary_path, "w") as h5_file:
                h5_file.create_dataset("vec", data=cad_vec, dtype=np.int64)
            os.replace(temporary_path, save_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return "written", sample_id, tuple(cad_vec.shape)
    except Exception as error:
        return "failed", sample_id, repr(error)


def main():
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero")

    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = (
        Path(args.raw_dir).expanduser().resolve()
        if args.raw_dir
        else data_root / "json"
    )
    split_path = (
        Path(args.split_file).expanduser().resolve()
        if args.split_file
        else data_root / "raw_train_test_split.json"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_root / "vec"
    )

    if not raw_dir.is_dir():
        raise FileNotFoundError("JSON input directory does not exist: {}".format(raw_dir))

    base_ids, split_names = load_split_ids(split_path, args.split)
    selected_ids = select_base_ids(base_ids, args.idx, args.num)
    json_paths = collect_json_paths(raw_dir, selected_ids)

    print("[!] split file: {}".format(split_path))
    print("[!] selected splits: {}".format(", ".join(split_names)))
    print("[!] selected base/complete IDs: {}".format(len(selected_ids)))
    print("[!] resolved JSON sequence files: {}".format(len(json_paths)))
    print("[!] output directory: {}".format(output_dir))

    status_counts = {"written": 0, "skipped": 0, "too_long": 0, "failed": 0}
    failure_examples = []
    total_paths = len(json_paths)

    for batch_start in range(0, total_paths, args.batch_size):
        batch_paths = json_paths[batch_start : batch_start + args.batch_size]
        if args.jobs == 1:
            results = [
                process_one(str(path), str(raw_dir), str(output_dir), args.overwrite)
                for path in batch_paths
            ]
        else:
            results = Parallel(n_jobs=args.jobs)(
                delayed(process_one)(
                    str(path), str(raw_dir), str(output_dir), args.overwrite
                )
                for path in batch_paths
            )

        for status, sample_id, detail in results:
            status_counts[status] += 1
            if status in ("failed", "too_long") and len(failure_examples) < 20:
                failure_examples.append((status, sample_id, detail))

        processed_count = min(batch_start + len(batch_paths), total_paths)
        print("[!] processed {}/{} JSON files".format(processed_count, total_paths))

    print("[!] result counts: {}".format(status_counts))
    if failure_examples:
        print("[!] first conversion failures:")
        for example in failure_examples:
            print("    {}".format(example))


if __name__ == "__main__":
    main()
