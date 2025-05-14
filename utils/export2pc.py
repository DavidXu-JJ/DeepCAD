import os
import json
import numpy as np
import argparse
import sys
import trimesh

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD

def point_normalize(points):  # [N, 3]
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    scaled_points = points / np.max((max_vals - min_vals), axis=0)

    min_vals2 = np.min(scaled_points, axis=0)
    max_vals2 = np.max(scaled_points, axis=0)
    scaled_points = scaled_points * 2.0 - (min_vals2 + max_vals2)

    add = (min_vals2 + max_vals2)
    scale = np.max((max_vals - min_vals), axis=0)
    return scaled_points, add, scale

def save_points(points, save_path, output_format="npy"):
    if output_format == "npy":
        np.save(save_path + ".npy", points)
    elif output_format == "npz":
        np.savez_compressed(save_path + ".npz", points=points)
    elif output_format == "ply":
        cloud = trimesh.PointCloud(points)
        cloud.export(save_path + ".ply")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def process_one(root, data_id, save_root, normalize, output_format, n_points=100000):
    new_directory_path = os.path.join(save_root, data_id[:4])
    os.makedirs(new_directory_path, exist_ok=True)
    save_path = os.path.join(new_directory_path, data_id)

    if os.path.exists(save_path + f".{output_format}"):
        print("skip {}: file already exists".format(data_id))
        return

    json_path = os.path.join(root, data_id + ".json")
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        print("create_CAD failed:", data_id, e)
        return

    try:
        out_pc = CADsolid2pc(shape, n_points, data_id)
    except Exception as e:
        print("convert point cloud failed:", data_id, e)
        return

    if normalize:
        out_pc, _, _ = point_normalize(out_pc)

    save_points(out_pc, save_path, output_format)

def main(args):
    data_root = args.data_root
    raw_data = os.path.join(data_root, "json")
    save_root = os.path.join(data_root, "pcd")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for index in range(100):
        folder = os.path.join(raw_data, "{:04d}".format(index))
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    data_id = file[:-5]  # remove ".json"
                    process_one(
                        root=root,
                        data_id=data_id,
                        save_root=save_root,
                        normalize=args.normalize,
                        output_format=args.output_format,
                        n_points=args.n_points
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of the dataset")
    parser.add_argument('--normalize', action='store_true', help="Whether to normalize the point cloud to [-1, 1]")
    parser.add_argument('--output_format', type=str, default='npy', choices=['npy', 'npz', 'ply'], help="Output file format")
    parser.add_argument('--n_points', type=int, default=100000, help="Number of points in sampled point cloud")

    args = parser.parse_args()
    main(args)
