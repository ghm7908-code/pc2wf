import argparse
import json
from pathlib import Path

import numpy as np


GT_DIR_CANDIDATES = ("gt", "wireframe")


def _parse_obj(path):
    vertices = []
    edges = []
    if (not path.exists()) or path.stat().st_size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    with open(path, "r", encoding="utf-8") as obj_file:
        for raw_line in obj_file:
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue
            items = line.split()
            if items[0] == "v" and len(items) >= 4:
                vertices.append([float(items[1]), float(items[2]), float(items[3])])
            elif items[0] == "l" and len(items) >= 3:
                idxs = []
                for item in items[1:]:
                    idx = item.split("/")[0]
                    if idx:
                        idxs.append(int(idx) - 1)
                for start, end in zip(idxs[:-1], idxs[1:]):
                    if start != end:
                        edges.append([start, end])

    return np.asarray(vertices, dtype=np.float32), np.asarray(edges, dtype=np.int32)


def _pairwise_dist(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)), dtype=np.float32)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _greedy_match(candidates, pred_count, gt_count):
    matched_pred = set()
    matched_gt = set()
    matches = []
    for cost, pred_idx, gt_idx in sorted(candidates, key=lambda x: x[0]):
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, float(cost)))
    return matches


def match_vertices(pred_vertices, gt_vertices, threshold):
    if len(pred_vertices) == 0 or len(gt_vertices) == 0:
        return []

    dists = _pairwise_dist(pred_vertices, gt_vertices)
    pred_ids, gt_ids = np.where(dists <= threshold)
    candidates = [(float(dists[i, j]), int(i), int(j)) for i, j in zip(pred_ids, gt_ids)]
    return _greedy_match(candidates, len(pred_vertices), len(gt_vertices))


def _edge_match_cost(pred_vertices, pred_edge, gt_vertices, gt_edge):
    p0 = pred_vertices[pred_edge[0]]
    p1 = pred_vertices[pred_edge[1]]
    g0 = gt_vertices[gt_edge[0]]
    g1 = gt_vertices[gt_edge[1]]

    d00 = np.linalg.norm(p0 - g0)
    d11 = np.linalg.norm(p1 - g1)
    d01 = np.linalg.norm(p0 - g1)
    d10 = np.linalg.norm(p1 - g0)

    direct_max = max(d00, d11)
    swap_max = max(d01, d10)
    if direct_max <= swap_max:
        return direct_max, d00 + d11
    return swap_max, d01 + d10


def match_edges(pred_vertices, pred_edges, gt_vertices, gt_edges, threshold):
    if len(pred_edges) == 0 or len(gt_edges) == 0 or len(pred_vertices) == 0 or len(gt_vertices) == 0:
        return []

    candidates = []
    for pred_idx, pred_edge in enumerate(pred_edges):
        if pred_edge.max(initial=-1) >= len(pred_vertices) or pred_edge.min(initial=0) < 0:
            continue
        for gt_idx, gt_edge in enumerate(gt_edges):
            if gt_edge.max(initial=-1) >= len(gt_vertices) or gt_edge.min(initial=0) < 0:
                continue
            max_cost, sum_cost = _edge_match_cost(pred_vertices, pred_edge, gt_vertices, gt_edge)
            if max_cost <= threshold:
                candidates.append((float(sum_cost), int(pred_idx), int(gt_idx)))
    return _greedy_match(candidates, len(pred_edges), len(gt_edges))


def _f1(precision, recall):
    return 2.0 * precision * recall / (precision + recall + 1e-12)


def _safe_ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def _find_gt_dir(data_root, split):
    split_root = Path(data_root) / split
    for gt_dir_name in GT_DIR_CANDIDATES:
        gt_dir = split_root / gt_dir_name
        if gt_dir.exists():
            return gt_dir
    return split_root / "gt"


def evaluate_dataset(data_root, split, obj_dir, vertex_th, edge_th):
    gt_dir = _find_gt_dir(data_root, split)
    gt_files = sorted(gt_dir.glob("*.obj"))
    if not gt_files:
        raise FileNotFoundError(f"No GT OBJ files found under {gt_dir}")

    totals = {
        "samples": len(gt_files),
        "pred_obj_present": 0,
        "pred_obj_nonempty": 0,
        "gt_vertices": 0,
        "pred_vertices": 0,
        "matched_vertices": 0,
        "gt_edges": 0,
        "pred_edges": 0,
        "matched_edges": 0,
    }
    per_sample = []

    obj_dir = Path(obj_dir)
    for gt_file in gt_files:
        stem = gt_file.stem
        pred_file = obj_dir / f"{stem}_pred.obj"
        gt_vertices, gt_edges = _parse_obj(gt_file)
        pred_vertices, pred_edges = _parse_obj(pred_file)

        if pred_file.exists():
            totals["pred_obj_present"] += 1
            if pred_file.stat().st_size > 0:
                totals["pred_obj_nonempty"] += 1

        vertex_matches = match_vertices(pred_vertices, gt_vertices, vertex_th)
        edge_matches = match_edges(pred_vertices, pred_edges, gt_vertices, gt_edges, edge_th)

        totals["gt_vertices"] += len(gt_vertices)
        totals["pred_vertices"] += len(pred_vertices)
        totals["matched_vertices"] += len(vertex_matches)
        totals["gt_edges"] += len(gt_edges)
        totals["pred_edges"] += len(pred_edges)
        totals["matched_edges"] += len(edge_matches)

        vertex_precision = _safe_ratio(len(vertex_matches), len(pred_vertices))
        vertex_recall = _safe_ratio(len(vertex_matches), len(gt_vertices))
        edge_precision = _safe_ratio(len(edge_matches), len(pred_edges))
        edge_recall = _safe_ratio(len(edge_matches), len(gt_edges))

        per_sample.append(
            {
                "name": stem,
                "pred_obj_exists": pred_file.exists(),
                "gt_vertices": int(len(gt_vertices)),
                "pred_vertices": int(len(pred_vertices)),
                "matched_vertices": int(len(vertex_matches)),
                "vertex_precision": vertex_precision,
                "vertex_recall": vertex_recall,
                "vertex_f1": _f1(vertex_precision, vertex_recall),
                "gt_edges": int(len(gt_edges)),
                "pred_edges": int(len(pred_edges)),
                "matched_edges": int(len(edge_matches)),
                "edge_precision": edge_precision,
                "edge_recall": edge_recall,
                "edge_f1": _f1(edge_precision, edge_recall),
            }
        )

    vertex_precision = _safe_ratio(totals["matched_vertices"], totals["pred_vertices"])
    vertex_recall = _safe_ratio(totals["matched_vertices"], totals["gt_vertices"])
    edge_precision = _safe_ratio(totals["matched_edges"], totals["pred_edges"])
    edge_recall = _safe_ratio(totals["matched_edges"], totals["gt_edges"])

    summary = {
        "split": split,
        "data_root": str(data_root),
        "obj_dir": str(obj_dir),
        "vertex_threshold": float(vertex_th),
        "edge_threshold": float(edge_th),
        "samples": totals["samples"],
        "pred_obj_present": totals["pred_obj_present"],
        "pred_obj_nonempty": totals["pred_obj_nonempty"],
        "vertex_precision": vertex_precision,
        "vertex_recall": vertex_recall,
        "vertex_f1": _f1(vertex_precision, vertex_recall),
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": _f1(edge_precision, edge_recall),
        "gt_vertices": totals["gt_vertices"],
        "pred_vertices": totals["pred_vertices"],
        "matched_vertices": totals["matched_vertices"],
        "gt_edges": totals["gt_edges"],
        "pred_edges": totals["pred_edges"],
        "matched_edges": totals["matched_edges"],
    }
    return summary, per_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted wireframe OBJ files against GT OBJ files.")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root that contains split/gt or split/wireframe.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--obj_dir", type=str, required=True, help="Directory containing *_pred.obj files.")
    parser.add_argument("--vertex_th", type=float, default=0.03, help="Matching threshold for vertices.")
    parser.add_argument("--edge_th", type=float, default=0.05, help="Endpoint matching threshold for edges.")
    parser.add_argument("--report_path", type=str, default="", help="Optional JSON report output path.")
    parser.add_argument("--top_k_worst", type=int, default=10, help="How many worst samples to print by edge F1.")
    args = parser.parse_args()

    summary, per_sample = evaluate_dataset(
        data_root=args.data_root,
        split=args.split,
        obj_dir=args.obj_dir,
        vertex_th=args.vertex_th,
        edge_th=args.edge_th,
    )

    print(f"Split: {summary['split']}")
    print(f"OBJ dir: {summary['obj_dir']}")
    print(f"Samples: {summary['samples']}")
    print(f"Pred OBJ present: {summary['pred_obj_present']}/{summary['samples']}")
    print(f"Pred OBJ non-empty: {summary['pred_obj_nonempty']}/{summary['samples']}")
    print(
        "Vertex metrics: "
        f"P={summary['vertex_precision']:.4f} "
        f"R={summary['vertex_recall']:.4f} "
        f"F1={summary['vertex_f1']:.4f} "
        f"(matched {summary['matched_vertices']}, pred {summary['pred_vertices']}, gt {summary['gt_vertices']})"
    )
    print(
        "Edge metrics: "
        f"P={summary['edge_precision']:.4f} "
        f"R={summary['edge_recall']:.4f} "
        f"F1={summary['edge_f1']:.4f} "
        f"(matched {summary['matched_edges']}, pred {summary['pred_edges']}, gt {summary['gt_edges']})"
    )

    worst_samples = sorted(per_sample, key=lambda item: (item["edge_f1"], item["vertex_f1"], item["name"]))[: args.top_k_worst]
    if worst_samples:
        print("Worst samples by edge F1:")
        for item in worst_samples:
            print(
                f"  {item['name']}: "
                f"edge_f1={item['edge_f1']:.4f}, "
                f"vertex_f1={item['vertex_f1']:.4f}, "
                f"pred_edges={item['pred_edges']}, gt_edges={item['gt_edges']}"
            )

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump({"summary": summary, "per_sample": per_sample}, report_file, indent=2)
        print(f"Saved report to: {report_path}")
