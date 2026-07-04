import argparse
import collections
import json
import multiprocessing
import shutil
from pathlib import Path

import MinkowskiEngine.utils as ME_utils
import numpy as np
from scipy import spatial
from tqdm import tqdm

from noise_gen_line_train import gen_line


SPLIT_ALIASES = {
    "train": ["train"],
    "validation": ["validation", "val"],
    "test": ["test"],
}
GT_DIR_CANDIDATES = ("gt", "wireframe")
DEFAULT_GRAPH_RADIUS = 0.05
DEFAULT_QUANTIZATION_SIZE = 0.03
DEFAULT_PATCH_VERTEX_THRESHOLD = 0.01
DEFAULT_LINE_DIST_THRESHOLD = 0.03
MAX_TEST_SEEDS = 10000
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        array = np.expand_dims(array, 0)
    return array


def _replace_suffix(path, new_suffix):
    return str(Path(path).with_suffix(new_suffix))


def _resolve_split_name(root, canonical_split):
    root = Path(root)
    for split_name in SPLIT_ALIASES[canonical_split]:
        xyz_dir = root / split_name / "xyz"
        if xyz_dir.exists():
            return split_name
    return None


def _find_gt_path_for_noisy_xyz(point_path):
    point_path = Path(point_path)
    split_root = point_path.parent.parent
    file_name = point_path.name.replace(".xyz", ".obj")
    for gt_dir_name in GT_DIR_CANDIDATES:
        candidate = split_root / gt_dir_name / file_name
        if candidate.exists():
            return candidate
    return split_root / "gt" / file_name


class PatchGenerator:
    def __init__(
        self,
        point_path,
        save_root_path,
        patch_size=50,
        sigma=0.01,
        clip=0.01,
        split="train",
        graph_radius=DEFAULT_GRAPH_RADIUS,
        quantization_size=DEFAULT_QUANTIZATION_SIZE,
        patch_vertex_threshold=DEFAULT_PATCH_VERTEX_THRESHOLD,
        line_dist_threshold=DEFAULT_LINE_DIST_THRESHOLD,
    ):
        self.point_path = Path(point_path)
        self.save_root_path = Path(save_root_path)
        self.patch_size = patch_size
        self.sigma = sigma
        self.clip = clip
        self.split = split
        self.graph_radius = float(graph_radius)
        self.quantization_size = float(quantization_size)
        self.patch_vertex_threshold = float(patch_vertex_threshold)
        self.line_dist_threshold = float(line_dist_threshold)
        self.gt_path = _find_gt_path_for_noisy_xyz(point_path)

        self.save_root_path.parent.mkdir(parents=True, exist_ok=True)

        raw_data = _ensure_2d(np.loadtxt(self.point_path))
        self.pointcloud = raw_data[:, :3].astype(np.float32)
        self._save_metadata()

        self.pointcloud_down, self.coords = self._down_sample()
        self._build_graph()

    def _save_metadata(self):
        center = np.zeros(3, dtype=np.float32)
        max_dist = 1.0

        if self.gt_path.exists():
            with open(self.gt_path, "r", encoding="utf-8") as gt_file:
                for line in gt_file:
                    if line.startswith("# centroid:"):
                        parts = line.strip().split()
                        center = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
                    elif line.startswith("# max_dist:"):
                        max_dist = float(line.strip().split()[-1])

        meta_data = {
            "center": center.tolist(),
            "max_dist": float(max_dist),
            "original_path": str(self.point_path),
            "graph_radius": float(self.graph_radius),
            "quantization_size": float(self.quantization_size),
        }
        with open(_replace_suffix(self.save_root_path, ".json"), "w", encoding="utf-8") as meta_file:
            json.dump(meta_data, meta_file)

    def _down_sample(self):
        feats = np.ones((self.pointcloud.shape[0], 1), dtype=np.float32)
        offset = np.mean(self.pointcloud, axis=0)
        coords = np.floor((self.pointcloud - offset) / self.quantization_size).astype(np.int32)

        _, inds = ME_utils.sparse_quantize(coords, return_index=True)
        inds = np.asarray(inds).astype(np.int64).reshape(-1)

        coords = np.hstack([coords[inds], np.zeros((len(inds), 1), dtype=np.int32)])
        pointcloud_down = self.pointcloud[inds]
        feats = feats[inds]

        np.savetxt(_replace_suffix(self.save_root_path, ".down"), pointcloud_down, fmt="%.6f")
        np.savetxt(_replace_suffix(self.save_root_path, ".feats"), feats, fmt="%.1f")
        np.savetxt(_replace_suffix(self.save_root_path, ".coords"), coords, fmt="%d")
        return pointcloud_down, coords

    def _build_graph(self):
        if len(self.pointcloud_down) == 0:
            self.nbrs = None
            self.graph = []
            return

        self.nbrs = spatial.cKDTree(self.pointcloud_down)
        # Build a denser local graph than the original implementation so BFS patches
        # can still reach enough neighbors on complex real-world geometry.
        k = min(max(self.patch_size * 2, 32), len(self.pointcloud_down))
        dists, idxs = self.nbrs.query(self.pointcloud_down, k=k)
        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

        self.graph = []
        for item, dist in zip(idxs, dists):
            valid_item = np.atleast_1d(item)[np.atleast_1d(dist) < self.graph_radius]
            self.graph.append(set(valid_item.astype(np.int32).tolist()))

    def _load_gt(self):
        if not self.gt_path.exists():
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 6), dtype=np.float32)

        vertices = []
        edges = []
        with open(self.gt_path, "r", encoding="utf-8") as gt_file:
            for line in gt_file:
                item = line.strip().split()
                if not item:
                    continue
                if item[0] == "v":
                    vertices.append([float(x) for x in item[1:4]])
                elif item[0] == "l" and len(item) >= 3:
                    start_idx = int(item[1]) - 1
                    end_idx = int(item[2]) - 1
                    if start_idx < len(vertices) and end_idx < len(vertices):
                        edges.append(vertices[start_idx] + vertices[end_idx])

        vertices = np.asarray(vertices, dtype=np.float32)
        edges = np.asarray(edges, dtype=np.float32)

        if edges.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 6), dtype=np.float32)

        vert_with_line = []
        seen = set()
        for edge in edges:
            for endpoint in (tuple(edge[:3]), tuple(edge[3:6])):
                if endpoint in seen:
                    continue
                seen.add(endpoint)
                vert_with_line.append(endpoint)

        return np.asarray(vert_with_line, dtype=np.float32), edges

    def _count_vertices_in_patch(self, patch_index, gt_vertices):
        if len(gt_vertices) == 0 or len(patch_index) == 0:
            return 0
        patch_points = self.pointcloud_down[patch_index]
        dists = np.linalg.norm(gt_vertices[:, None, :] - patch_points[None, :, :], axis=2).min(axis=1)
        return int(np.sum(dists < self.patch_vertex_threshold))

    def bfs_knn(self, seed=0, k=10):
        if not self.graph:
            return np.empty((0,), dtype=np.int32)

        q = collections.deque([int(seed)])
        visited = set()
        result = []

        while len(visited) < k and q:
            vertex = q.popleft()
            if vertex in visited:
                continue
            visited.add(vertex)
            result.append(vertex)
            if len(q) < k * 5:
                q.extend(self.graph[vertex] - visited)

        return np.asarray(result, dtype=np.int32)

    def _knn_patch(self, query_point, k):
        if self.nbrs is None or len(self.pointcloud_down) == 0:
            return np.empty((0,), dtype=np.int32)

        k = min(int(k), len(self.pointcloud_down))
        if k <= 0:
            return np.empty((0,), dtype=np.int32)

        _, idxs = self.nbrs.query(query_point, k=k)
        idxs = np.atleast_1d(idxs).astype(np.int32)
        return idxs

    def _patch_from_seed(self, seed):
        patch_index = self.bfs_knn(seed=int(seed), k=self.patch_size)
        if len(patch_index) == self.patch_size:
            return patch_index
        return self._knn_patch(self.pointcloud_down[int(seed)], self.patch_size)

    def _candidate_patches_for_vertex(self, vertex):
        if self.nbrs is None or len(self.pointcloud_down) == 0:
            return []

        candidates = []
        seen = set()

        # First try a direct local KNN patch centered at the GT vertex.
        direct_patch = self._knn_patch(vertex, self.patch_size)
        if len(direct_patch) == self.patch_size:
            patch_key = tuple(direct_patch.tolist())
            seen.add(patch_key)
            candidates.append(direct_patch)

        seed_k = min(max(self.patch_size, 32), len(self.pointcloud_down))
        dists, seed_idx = self.nbrs.query(vertex, k=seed_k)
        dists = np.atleast_1d(dists)
        seed_idx = np.atleast_1d(seed_idx)

        # Search a bit wider than graph_radius to avoid missing valid seeds after
        # down-sampling around thin structures.
        seed_radius = max(self.graph_radius, self.quantization_size * 2.0, self.patch_vertex_threshold * 4.0)
        candidate_seeds = seed_idx[dists < seed_radius]
        if len(candidate_seeds) == 0:
            candidate_seeds = seed_idx[: min(8, len(seed_idx))]

        candidate_seeds = candidate_seeds.copy()
        np.random.shuffle(candidate_seeds)
        for seed in candidate_seeds:
            patch_index = self._patch_from_seed(seed=int(seed))
            if len(patch_index) != self.patch_size:
                continue
            patch_key = tuple(patch_index.tolist())
            if patch_key in seen:
                continue
            seen.add(patch_key)
            candidates.append(patch_index)
        return candidates

    def farthest_sample(self, k):
        count = len(self.pointcloud_down)
        if count == 0:
            return np.empty((0,), dtype=np.int32)
        if count <= k:
            return np.arange(count, dtype=np.int32)

        farthest_pts_index = np.zeros((k,), dtype=np.int32)
        farthest_pts_index[0] = np.random.randint(count)
        distances = np.sum((self.pointcloud_down - self.pointcloud_down[farthest_pts_index[0]]) ** 2, axis=1)

        for i in range(1, k):
            farthest_pts_index[i] = int(np.argmax(distances))
            new_dist = np.sum((self.pointcloud_down - self.pointcloud_down[farthest_pts_index[i]]) ** 2, axis=1)
            distances = np.minimum(distances, new_dist)

        return farthest_pts_index

    def _generate_positive_patches(self, gt_vertices):
        if len(gt_vertices) == 0:
            return np.empty((0, self.patch_size), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

        patches = []
        vert_gts = []
        seen = set()

        for vertex in gt_vertices:
            for patch_index in self._candidate_patches_for_vertex(vertex):
                if self._count_vertices_in_patch(patch_index, gt_vertices) != 1:
                    continue
                if np.min(np.linalg.norm(self.pointcloud_down[patch_index] - vertex, axis=1)) > self.patch_vertex_threshold:
                    continue

                patch_key = tuple(patch_index.tolist())
                if patch_key in seen:
                    continue

                seen.add(patch_key)
                patches.append(patch_index)
                vert_gts.append(vertex)
                break

        if not patches:
            return np.empty((0, self.patch_size), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

        return np.asarray(patches, dtype=np.int32), np.asarray(vert_gts, dtype=np.float32)

    def _generate_negative_patches(self, gt_vertices, target_count, seen_positive_patches):
        if target_count <= 0:
            return np.empty((0, self.patch_size), dtype=np.int32)

        seen = set(seen_positive_patches)
        patches = []
        seed_points = self.farthest_sample(min(MAX_TEST_SEEDS, len(self.pointcloud_down)))

        for seed in seed_points:
            if len(patches) >= target_count:
                break

            if len(gt_vertices) > 0:
                if np.min(np.linalg.norm(gt_vertices - self.pointcloud_down[seed], axis=1)) < self.graph_radius:
                    continue

            patch_index = self._patch_from_seed(seed=int(seed))
            if len(patch_index) != self.patch_size:
                continue
            if self._count_vertices_in_patch(patch_index, gt_vertices) != 0:
                continue

            patch_key = tuple(patch_index.tolist())
            if patch_key in seen:
                continue

            seen.add(patch_key)
            patches.append(patch_index)

        if not patches:
            return np.empty((0, self.patch_size), dtype=np.int32)

        return np.asarray(patches, dtype=np.int32)

    def _generate_test_patches(self):
        patches = []
        seen = set()
        seed_points = self.farthest_sample(min(MAX_TEST_SEEDS, len(self.pointcloud_down)))

        for seed in seed_points:
            patch_index = self._patch_from_seed(seed=int(seed))
            if len(patch_index) != self.patch_size:
                continue

            patch_key = tuple(patch_index.tolist())
            if patch_key in seen:
                continue

            seen.add(patch_key)
            patches.append(patch_index)

        if not patches:
            return np.empty((0, self.patch_size), dtype=np.int32)
        return np.asarray(patches, dtype=np.int32)

    def process(self):
        if self.split == "test":
            patch_index = self._generate_test_patches()
            np.savetxt(_replace_suffix(self.save_root_path, ".patch_index"), patch_index, fmt="%d")
            return {
                "positive_patches": 0,
                "negative_patches": 0,
                "mini_lines": 0,
            }

        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {self.gt_path}")

        vert_gt, edge_gt = self._load_gt()
        if len(vert_gt) == 0 or len(edge_gt) == 0:
            raise ValueError(f"GT file has no usable vertices/edges: {self.gt_path}")

        positive_patches, positive_vert_gt = self._generate_positive_patches(vert_gt)
        seen_positive = {tuple(patch.tolist()) for patch in positive_patches}
        negative_patches = self._generate_negative_patches(
            gt_vertices=vert_gt,
            target_count=len(positive_patches),
            seen_positive_patches=seen_positive,
        )

        np.savetxt(_replace_suffix(self.save_root_path, ".vert_index"), positive_patches, fmt="%d")
        np.savetxt(_replace_suffix(self.save_root_path, ".vert_gt"), positive_vert_gt, fmt="%.6f")
        np.savetxt(_replace_suffix(self.save_root_path, ".other_index"), negative_patches, fmt="%d")

        mini_lines = gen_line(
            point_path=None,
            edge_gt=edge_gt,
            patch_size=self.patch_size,
            sigma=self.sigma,
            clip=self.clip,
            train_val_test=self.split,
            save_root_path=str(self.save_root_path),
            pointcloud_down=self.pointcloud_down,
            coords=self.coords,
            dist_threshold=self.line_dist_threshold,
        )

        return {
            "positive_patches": len(positive_patches),
            "negative_patches": len(negative_patches),
            "mini_lines": len(mini_lines),
        }


def _run_task(args):
    (
        point_path,
        save_root_path,
        patch_size,
        sigma,
        clip,
        split,
        graph_radius,
        quantization_size,
        patch_vertex_threshold,
        line_dist_threshold,
    ) = args
    try:
        stats = PatchGenerator(
            point_path=point_path,
            save_root_path=save_root_path,
            patch_size=patch_size,
            sigma=sigma,
            clip=clip,
            split=split,
            graph_radius=graph_radius,
            quantization_size=quantization_size,
            patch_vertex_threshold=patch_vertex_threshold,
            line_dist_threshold=line_dist_threshold,
        ).process()
        return True, Path(point_path).name, stats
    except Exception as exc:
        return False, Path(point_path).name, str(exc)


def _has_completed_outputs(save_root_path, split):
    required_suffixes = [".down", ".feats", ".coords", ".json"]
    if split == "test":
        required_suffixes.append(".patch_index")
    else:
        required_suffixes.extend([".mini_line", ".other_index", ".vert_index", ".vert_gt"])

    return all(Path(_replace_suffix(save_root_path, suffix)).exists() for suffix in required_suffixes)


def process_split(
    noise_root,
    output_root,
    split,
    patch_size,
    sigma,
    clip,
    num_workers,
    graph_radius,
    quantization_size,
    patch_vertex_threshold,
    line_dist_threshold,
    rebuild=False,
    names_file=None,
):
    actual_split = _resolve_split_name(noise_root, split)
    if actual_split is None:
        print(f"Skip {split}: {noise_root}/{split}/xyz not found.")
        return

    xyz_dir = Path(noise_root) / actual_split / "xyz"
    file_list = sorted(xyz_dir.glob("*.xyz"))

    if names_file:
        with open(names_file, 'r', encoding='utf-8') as f:
            names = {line.strip() for line in f if line.strip()}
        file_list = [fp for fp in file_list if fp.stem in names]
        print(f"Filtered by names_file: {len(names)} names loaded, {len(file_list)} files matched.")

    split_output_dir = Path(output_root) / split

    if rebuild and split_output_dir.exists():
        shutil.rmtree(split_output_dir)
    split_output_dir.mkdir(parents=True, exist_ok=True)

    task_args = []
    for file_path in file_list:
        save_root_path = split_output_dir / file_path.name
        if not rebuild and _has_completed_outputs(save_root_path, split):
            continue
        task_args.append(
            (
                str(file_path),
                str(save_root_path),
                patch_size,
                sigma,
                clip,
                split,
                graph_radius,
                quantization_size,
                patch_vertex_threshold,
                line_dist_threshold,
            )
        )

    print(
        f"Processing {split}: found {len(file_list)} files in split '{actual_split}', "
        f"need to build {len(task_args)} files."
    )

    if not task_args:
        return

    if num_workers <= 1:
        results = [_run_task(task) for task in tqdm(task_args, total=len(task_args))]
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_run_task, task_args), total=len(task_args)))

    failures = [result for result in results if not result[0]]
    for _, file_name, error_message in failures:
        print(f"Error on {split}/{file_name}: {error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training patches and line labels for PC2WF.")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Dataset root that contains noise_sigma*/.")
    parser.add_argument("--noise_root", type=str, default=None, help="Optional noisy dataset root.")
    parser.add_argument("--output_root", type=str, default=None, help="Optional patches output root.")
    parser.add_argument("--patch_size", type=int, default=50, help="Number of points per patch.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Noise sigma used by the noisy dataset.")
    parser.add_argument("--clip", type=float, default=0.01, help="Noise clip used by the noisy dataset.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--graph_radius", type=float, default=DEFAULT_GRAPH_RADIUS, help="Radius for quantization and local graph construction.")
    parser.add_argument("--quantization_size", type=float, default=DEFAULT_QUANTIZATION_SIZE, help="Voxel size used for sparse quantization before graph construction.")
    parser.add_argument(
        "--patch_vertex_threshold",
        type=float,
        default=DEFAULT_PATCH_VERTEX_THRESHOLD,
        help="Max distance from a GT vertex to a positive patch point.",
    )
    parser.add_argument(
        "--line_dist_threshold",
        type=float,
        default=DEFAULT_LINE_DIST_THRESHOLD,
        help="Max distance for matching GT line endpoints and interpolated line samples.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete old patch outputs before regenerating.")
    parser.add_argument("--names_file", type=str, default="", help="Path to a file listing cloud names (one per line) to filter which clouds to process.")
    parser.add_argument("--split", type=str, default="", help="Only process this split (train/validation/test). If empty, process all.")
    args = parser.parse_args()

    noise_root = Path(args.noise_root) if args.noise_root else Path(args.data_root) / f"noise_sigma{args.sigma}clip{args.clip}"
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(args.data_root) / f"patches_{args.patch_size}_noise_sigma{args.sigma}clip{args.clip}"
    )

    names_file = args.names_file if args.names_file else None
    splits_to_run = [args.split] if args.split else ["train", "validation", "test"]

    for split_name in splits_to_run:
        process_split(
            noise_root=noise_root,
            output_root=output_root,
            split=split_name,
            patch_size=args.patch_size,
            sigma=args.sigma,
            clip=args.clip,
            num_workers=args.num_workers,
            graph_radius=args.graph_radius,
            quantization_size=args.quantization_size,
            patch_vertex_threshold=args.patch_vertex_threshold,
            line_dist_threshold=args.line_dist_threshold,
            rebuild=args.rebuild,
            names_file=names_file,
        )
