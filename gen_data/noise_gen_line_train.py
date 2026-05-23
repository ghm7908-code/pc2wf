import itertools
from pathlib import Path

import MinkowskiEngine.utils as ME_utils
import numpy as np


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        array = np.expand_dims(array, 0)
    return array


def _replace_suffix(path, new_suffix):
    return str(Path(path).with_suffix(new_suffix))


def _trace_line(pointcloud_down, start_idx, end_idx, point_num_in_line, dist_threshold):
    if start_idx == end_idx:
        return None

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    start = pointcloud_down[start_idx]
    end = pointcloud_down[end_idx]

    row = [start_idx]
    for inter_point in range(1, point_num_in_line + 1):
        ratio = float(inter_point) / (point_num_in_line + 1)
        target = (1.0 - ratio) * start + ratio * end
        dists = np.linalg.norm(pointcloud_down - target, axis=1)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] > dist_threshold:
            return None
        row.append(nearest_idx)

    row.append(end_idx)
    return row


def _build_line_labels(pointcloud_down, edge_gt, point_num_in_line=30, dist_threshold=0.03):
    edge_gt = np.asarray(edge_gt, dtype=np.float32)
    if edge_gt.size == 0:
        return np.empty((0, point_num_in_line + 3), dtype=np.int32)
    if edge_gt.ndim == 1:
        edge_gt = edge_gt.reshape(1, -1)

    positive_rows = []
    positive_pairs = set()
    valid_vertex_ids = set()

    for edge in edge_gt:
        if edge.shape[0] < 6:
            continue

        p1, p2 = edge[:3], edge[3:6]
        e1 = int(np.argmin(np.linalg.norm(pointcloud_down - p1, axis=1)))
        e2 = int(np.argmin(np.linalg.norm(pointcloud_down - p2, axis=1)))

        if np.linalg.norm(pointcloud_down[e1] - p1) > dist_threshold:
            continue
        if np.linalg.norm(pointcloud_down[e2] - p2) > dist_threshold:
            continue

        row = _trace_line(pointcloud_down, e1, e2, point_num_in_line, dist_threshold)
        if row is None:
            continue

        positive_rows.append(row + [1])
        positive_pairs.add(tuple(sorted((e1, e2))))
        valid_vertex_ids.add(e1)
        valid_vertex_ids.add(e2)

    negative_rows = []
    candidate_pairs = list(itertools.combinations(sorted(valid_vertex_ids), 2))
    rng = np.random.default_rng(0)
    rng.shuffle(candidate_pairs)
    target_negative = max(len(positive_rows), 1)

    for e1, e2 in candidate_pairs:
        if tuple(sorted((e1, e2))) in positive_pairs:
            continue

        row = _trace_line(pointcloud_down, e1, e2, point_num_in_line, dist_threshold)
        if row is None:
            continue

        negative_rows.append(row + [0])
        if len(negative_rows) >= target_negative:
            break

    all_rows = positive_rows + negative_rows
    if not all_rows:
        return np.empty((0, point_num_in_line + 3), dtype=np.int32)

    return np.asarray(all_rows, dtype=np.int32)


def _infer_save_root_path(point_path, patch_size, sigma, clip, train_val_test):
    point_path = Path(point_path)
    file_name = point_path.name.replace(".norm.tmp", "")
    xyz_dir = point_path.parent
    split_dir = xyz_dir.parent
    noise_root = split_dir.parent
    data_root = noise_root.parent
    base_dir = data_root / f"patches_{patch_size}_noise_sigma{sigma}clip{clip}" / train_val_test
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / file_name)


def gen_line(
    point_path,
    edge_gt,
    vert_gt_list=None,
    index=0,
    rotate_angle=None,
    random_rotate=False,
    patch_size=50,
    clean_noise="noise",
    sigma=0.01,
    clip=0.01,
    train_val_test="test",
    save_root_path=None,
    pointcloud_down=None,
    coords=None,
    point_num_in_line=30,
    dist_threshold=0.03,
):
    del vert_gt_list, index, rotate_angle, random_rotate, clean_noise

    if pointcloud_down is None:
        pointcloud = _ensure_2d(np.loadtxt(point_path))
        pointcloud_coords = pointcloud[:, :3].astype(np.float32)
        quantization_size = 0.05
        quantized_coords = np.floor(pointcloud_coords / quantization_size).astype(np.int32)
        _, sel = ME_utils.sparse_quantize(quantized_coords, return_index=True)
        sel = np.asarray(sel).astype(np.int64).reshape(-1)
        pointcloud_down = pointcloud_coords[sel]
        coords = np.hstack([quantized_coords[sel], np.zeros((len(sel), 1), dtype=np.int32)])

    pointcloud_down = _ensure_2d(np.asarray(pointcloud_down, dtype=np.float32))
    if coords is not None:
        coords = _ensure_2d(np.asarray(coords, dtype=np.int32))

    if save_root_path is None:
        if point_path is None:
            raise ValueError("Either save_root_path or point_path must be provided.")
        save_root_path = _infer_save_root_path(point_path, patch_size, sigma, clip, train_val_test)

    save_root_path = Path(save_root_path)
    save_root_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(_replace_suffix(save_root_path, ".down"), pointcloud_down, fmt="%.6f")
    np.savetxt(_replace_suffix(save_root_path, ".feats"), np.ones((len(pointcloud_down), 1), dtype=np.int32), fmt="%d")
    if coords is not None:
        np.savetxt(_replace_suffix(save_root_path, ".coords"), coords, fmt="%d")

    mini_lines = _build_line_labels(
        pointcloud_down=pointcloud_down,
        edge_gt=edge_gt,
        point_num_in_line=point_num_in_line,
        dist_threshold=dist_threshold,
    )
    np.savetxt(_replace_suffix(save_root_path, ".mini_line"), mini_lines, fmt="%d")
    return mini_lines
