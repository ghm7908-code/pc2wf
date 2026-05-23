import argparse
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


SPLIT_ALIASES = {
    "train": ["train"],
    "validation": ["validation", "val"],
    "test": ["test"],
}
GT_DIR_CANDIDATES = ("gt", "wireframe")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        array = np.expand_dims(array, 0)
    return array


def _resolve_split_name(data_root, canonical_split):
    data_root = Path(data_root)
    for split_name in SPLIT_ALIASES[canonical_split]:
        xyz_dir = data_root / split_name / "xyz"
        if xyz_dir.exists():
            return split_name
    return None


def _find_gt_dir(split_root):
    split_root = Path(split_root)
    for gt_dir_name in GT_DIR_CANDIDATES:
        if (split_root / gt_dir_name).exists():
            return gt_dir_name
    return None


def add_noise(clean_point_path, dst_noisy_file, ori_gt_f=None, dst_noise_gt_f=None, sigma=0.01, clip=0.02):
    clean_pts = _ensure_2d(np.loadtxt(clean_point_path))
    row = clean_pts.shape[0]

    centroid = np.mean(clean_pts[:, :3], axis=0)
    centered_pts = clean_pts[:, :3] - centroid
    max_dist = np.max(np.linalg.norm(centered_pts, axis=1))
    if max_dist == 0:
        max_dist = 1.0

    noise = np.clip(sigma * np.random.randn(row, 3), -clip, clip)
    noisy_pts = clean_pts.copy()
    noisy_pts[:, :3] = centered_pts + noise

    norm_noisy_pts = noisy_pts.copy()
    norm_noisy_pts[:, :3] = np.clip(noisy_pts[:, :3] / max_dist, -1.0, 1.0)

    dst_noisy_file = Path(dst_noisy_file)
    dst_noisy_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst_noisy_file, norm_noisy_pts, fmt="%.4f")

    if ori_gt_f is None or dst_noise_gt_f is None:
        return
    if not Path(ori_gt_f).exists():
        return

    with open(ori_gt_f, "r", encoding="utf-8") as ori_f:
        vertices = []
        lines = []
        for line in ori_f:
            if line.startswith("v "):
                line_split = line.split()
                vertices.append([float(line_split[1]), float(line_split[2]), float(line_split[3])])
            elif line.startswith("l "):
                lines.append(line)

    dst_noise_gt_f = Path(dst_noise_gt_f)
    dst_noise_gt_f.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_noise_gt_f, "w", encoding="utf-8") as dst_f:
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.size > 0:
            vertices = np.clip((vertices - centroid) / max_dist, -1.0, 1.0)
            for vertex in vertices:
                dst_f.write("v {:.4f} {:.4f} {:.4f}\n".format(vertex[0], vertex[1], vertex[2]))

        for obj_line in lines:
            dst_f.write(obj_line)

        dst_f.write("# centroid: {} {} {}\n".format(centroid[0], centroid[1], centroid[2]))
        dst_f.write("# max_dist: {}\n".format(max_dist))


def _add_noise_one_file(args):
    clean_point_path, dst_noise_root, output_split, gt_dir_name, sigma, clip = args
    clean_point_path = Path(clean_point_path)
    split_root = clean_point_path.parent.parent
    file_name = clean_point_path.name

    dst_noisy_file = Path(dst_noise_root) / output_split / "xyz" / file_name
    ori_gt_f = None
    dst_noise_gt_f = None
    if gt_dir_name is not None:
        ori_gt_f = split_root / gt_dir_name / file_name.replace(".xyz", ".obj")
        dst_noise_gt_f = Path(dst_noise_root) / output_split / "gt" / file_name.replace(".xyz", ".obj")

    try:
        add_noise(
            clean_point_path=clean_point_path,
            dst_noisy_file=dst_noisy_file,
            ori_gt_f=ori_gt_f,
            dst_noise_gt_f=dst_noise_gt_f,
            sigma=sigma,
            clip=clip,
        )
        return True, file_name, ""
    except Exception as exc:
        return False, file_name, str(exc)


def process_split(data_root, dst_noise_root, canonical_split, sigma, clip, num_workers):
    source_split = _resolve_split_name(data_root, canonical_split)
    if source_split is None:
        print(f"Skip {canonical_split}: {data_root}/{canonical_split}/xyz not found.")
        return

    split_root = Path(data_root) / source_split
    xyz_dir = split_root / "xyz"
    gt_dir_name = _find_gt_dir(split_root)
    file_list = sorted(xyz_dir.glob("*.xyz"))

    print(f"Processing {canonical_split}: found {len(file_list)} files from split '{source_split}'.")
    if gt_dir_name is None and canonical_split != "test":
        print(f"Warning: GT directory not found for split '{source_split}'. Expected one of {GT_DIR_CANDIDATES}.")

    (Path(dst_noise_root) / canonical_split / "xyz").mkdir(parents=True, exist_ok=True)
    (Path(dst_noise_root) / canonical_split / "gt").mkdir(parents=True, exist_ok=True)

    task_args = [
        (str(file_path), str(dst_noise_root), canonical_split, gt_dir_name, sigma, clip)
        for file_path in file_list
    ]

    if num_workers <= 1:
        results = [_add_noise_one_file(task) for task in tqdm(task_args, total=len(task_args))]
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_add_noise_one_file, task_args), total=len(task_args)))

    failures = [result for result in results if not result[0]]
    for _, file_name, error_message in failures:
        print(f"Error processing {canonical_split}/{file_name}: {error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Gaussian noise and normalize point clouds to [-1, 1].")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Clean dataset root.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Gaussian noise sigma.")
    parser.add_argument("--clip", type=float, default=0.01, help="Gaussian noise clip.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the old noise directory before regenerating.")
    args = parser.parse_args()

    dst_noise_root = Path(args.data_root) / f"noise_sigma{args.sigma}clip{args.clip}"
    if args.rebuild and dst_noise_root.exists():
        shutil.rmtree(dst_noise_root)

    for split_name in ("train", "validation", "test"):
        process_split(
            data_root=args.data_root,
            dst_noise_root=dst_noise_root,
            canonical_split=split_name,
            sigma=args.sigma,
            clip=args.clip,
            num_workers=args.num_workers,
        )
