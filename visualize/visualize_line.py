import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from glob import glob
import argparse
import json
import csv
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
# from cal_prec_line import *


def _file_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def filter_prob_vertex(vertex_pred, vertex_probs, line_pred, line_probs, prob_th=0.5):
    '''filter vertex with probability'''
    dropped_vertex_index = []
    for vertex_i in range(len(vertex_probs)):
        if vertex_probs[vertex_i] < prob_th:
            dropped_vertex_index.append(vertex_i)
    dropped_vertex_index = [i+1 for i in dropped_vertex_index]
    keep_line_index = []
    for line_i in range(len(line_pred)):
        if (line_pred[line_i][0] not in dropped_vertex_index) and (line_pred[line_i][1] not in dropped_vertex_index):
            keep_line_index.append(line_i)
    line_pred = line_pred[keep_line_index]
    line_probs = line_probs[keep_line_index]
    return line_pred, line_probs

def filter_nms_vertex(vertex_pred, vertex_probs, line_pred, line_probs, nms_th=0.02):
    '''filter vertex with NMS'''
    dropped_vertex_index = []
    for vertex_i in range(len(vertex_probs)):
        if vertex_i in dropped_vertex_index:
            continue
        dist_all = np.linalg.norm(vertex_pred-vertex_pred[vertex_i], axis=1)
        same_region_indexes = (dist_all < nms_th).nonzero()
        for same_region_i in same_region_indexes[0]:
            if same_region_i == vertex_i:
                continue
            if vertex_probs[same_region_i] <= vertex_probs[vertex_i]:
                dropped_vertex_index.append(same_region_i)
            else:
                dropped_vertex_index.append(vertex_i)
    dropped_vertex_index = [i+1 for i in dropped_vertex_index]
    keep_line_index = []
    for line_i in range(len(line_pred)):
        if (line_pred[line_i][0] not in dropped_vertex_index) and (line_pred[line_i][1] not in dropped_vertex_index):
            keep_line_index.append(line_i)
    line_pred = line_pred[keep_line_index]
    line_probs = line_probs[keep_line_index]
    return line_pred, line_probs

def merge_vertex(vertex_pred, vertex_probs, merge_th=0.02):
    '''merge vertex that close to each other'''
    to_merge_index = [] # vertex that to be merged
    merge_to_index = [] # which vertex merge to
    for vertex_i in range(len(vertex_probs)):
        dist_all = np.linalg.norm(vertex_pred-vertex_pred[vertex_i], axis=1)
        same_region_indexes = (dist_all < merge_th).nonzero()
        for same_region_i in same_region_indexes[0]:
            if same_region_i == vertex_i:
                continue
            if vertex_probs[same_region_i] <= vertex_probs[vertex_i]:
                to_merge_index.append(same_region_i)
                merge_to_index.append(vertex_i)
            else:
                to_merge_index.append(vertex_i)
                merge_to_index.append(same_region_i)
    
    for merge_i in range(len(to_merge_index)):
        vertex_pred[to_merge_index[merge_i]] = vertex_pred[merge_to_index[merge_i]]
        vertex_probs[to_merge_index[merge_i]] = vertex_probs[merge_to_index[merge_i]]
    return vertex_pred, vertex_probs


def filter_prob_line(line_pred, line_probs, prob_th=0.5):
    '''filter line with probability'''
    filter_line = []
    filter_probs = []
    for line_i in range(len(line_probs)):
        if line_probs[line_i] >= prob_th:
            filter_line.append(line_pred[line_i])
            filter_probs.append(line_probs[line_i])
    return np.array(filter_line), np.array(filter_probs)

def filter_short_line(vertex_pred, line_pred, line_probs, len_th=0.01):
    '''filter short lines'''
    filter_line = []
    filter_probs = []
    for line_i in range(len(line_probs)):
        l0, l1 = vertex_pred[line_pred[line_i][0]-1], vertex_pred[line_pred[line_i][1]-1]
        if np.linalg.norm(l0-l1) > len_th:
            filter_line.append(line_pred[line_i])
            filter_probs.append(line_probs[line_i])
    return np.array(filter_line), np.array(filter_probs)


def filter_nms_line(vertex_pred, line_pred, line_probs, nms_th=0.05):
    '''filter lines with nms, sum of two endpoints <= nms_th'''
    dropped_line_index = []
    line_pred = line_pred.tolist()
    for line_i in range(len(line_probs)):
        if line_i in dropped_line_index:
            continue
        dist_l0 = np.linalg.norm(vertex_pred-vertex_pred[line_pred[line_i][0]-1], axis=1)
        dist_l1 = np.linalg.norm(vertex_pred-vertex_pred[line_pred[line_i][1]-1], axis=1)
        same_region_indexes_0 = (dist_l0 < nms_th).nonzero()[0]
        same_region_indexes_1 = (dist_l1 < nms_th).nonzero()[0]
        for region_i_0 in same_region_indexes_0:
            for region_i_1 in same_region_indexes_1:
                if ([region_i_0+1, region_i_1+1] == line_pred[line_i]) or ([region_i_1+1, region_i_0+1] == line_pred[line_i]):
                    continue
                if (dist_l0[region_i_0]+dist_l1[region_i_1])>nms_th:
                    continue
                close_line_index = -1
                if ([region_i_0+1, region_i_1+1] in line_pred):
                    close_line_index = line_pred.index([region_i_0+1, region_i_1+1])
                elif ([region_i_1+1, region_i_0+1] in line_pred):
                    close_line_index = line_pred.index([region_i_1+1, region_i_0+1])
                if close_line_index != -1:
                    if line_probs[close_line_index] <= line_probs[line_i]:
                        dropped_line_index.append(close_line_index)
                    else:
                        dropped_line_index.append(line_i)

    keep_line_index = [i for i in range(len(line_pred)) if i not in dropped_line_index]
    filter_line = np.array(line_pred)[keep_line_index]
    filter_probs = line_probs[keep_line_index]
    return np.array(filter_line), np.array(filter_probs)


def remove_extra_vertex(vertex_pred, line_pred):
    vertex_pred = vertex_pred.tolist()
    new_vertex_pred = []
    new_line_pred = []
    for v_i, v in enumerate(vertex_pred):
        if v not in new_vertex_pred and ((v_i+1) in line_pred):
            new_vertex_pred.append(v)
    
    for line in line_pred:
        line0, line1 = new_vertex_pred.index(vertex_pred[line[0]-1])+1, new_vertex_pred.index(vertex_pred[line[1]-1])+1
        if ([line0, line1] not in new_line_pred) and ([line1, line0] not in new_line_pred):
            new_line_pred.append([line0, line1])
    return np.array(new_vertex_pred), np.array(new_line_pred)

def merge_vertex(vertex_pred, vertex_probs, merge_th=0.02):
    '''merge vertex that close to each other'''
    to_merge_index = [] # vertex that to be merged
    merge_to_index = [] # which vertex merge to
    for vertex_i in range(len(vertex_probs)):
        dist_all = np.linalg.norm(vertex_pred-vertex_pred[vertex_i], axis=1)
        same_region_indexes = (dist_all < merge_th).nonzero()
        for same_region_i in same_region_indexes[0]:
            if same_region_i == vertex_i:
                continue
            if vertex_probs[same_region_i] <= vertex_probs[vertex_i]:
                to_merge_index.append(same_region_i)
                merge_to_index.append(vertex_i)
            else:
                to_merge_index.append(vertex_i)
                merge_to_index.append(same_region_i)
    
    for merge_i in range(len(to_merge_index)):
        vertex_pred[to_merge_index[merge_i]] = vertex_pred[merge_to_index[merge_i]]
        vertex_probs[to_merge_index[merge_i]] = vertex_probs[merge_to_index[merge_i]]
    return vertex_pred, vertex_probs

def line_to_obj(vertex_pred, line_pred, save_to_path):
    with open(save_to_path, 'w') as f:
        for v in vertex_pred:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for l in line_pred:
            f.write(f'l {l[0]} {l[1]}\n')


# ============================================================
# APCalculator precision evaluation (from evaluate_wireframe.py)
# ============================================================

def parse_obj_index(token, num_vertices):
    raw = int(token.split("/")[0])
    if raw > 0:
        return raw - 1
    return num_vertices + raw


def load_wireframe_obj(obj_path):
    vertices = []
    edges = []
    with open(obj_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                vertices.append([float(v) for v in parts[1:4]])
                continue
            if tag == "l":
                ids = [parse_obj_index(token, len(vertices)) for token in parts[1:]]
                for start, end in zip(ids[:-1], ids[1:]):
                    if start != end:
                        edges.append(sorted((start, end)))
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(edges) == 0:
        edges = np.zeros((0, 2), dtype=np.int32)
    else:
        edges = np.unique(np.asarray(edges, dtype=np.int32), axis=0)
    if len(edges) == 0 or len(vertices) == 0:
        edge_vertices = np.zeros((0, 2, 3), dtype=np.float64)
    else:
        edge_vertices = vertices[edges]
    return {
        "vertices": vertices.reshape(-1, 3),
        "edges": edges.reshape(-1, 2),
        "edge_vertices": edge_vertices.reshape(-1, 2, 3),
    }


def hausdorff_distance_line(p_line, t_line, sample_points=20):
    n_pred, n_gt = p_line.shape[0], t_line.shape[0]
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)
    all_lines = np.concatenate((p_line, t_line), axis=0)
    weights = np.linspace(0.0, 1.0, sample_points, dtype=np.float64).reshape(1, sample_points, 1)
    all_points = all_lines[:, 0, :][:, np.newaxis, :] + weights * (
        all_lines[:, 1, :][:, np.newaxis, :] - all_lines[:, 0, :][:, np.newaxis, :]
    )
    distance_matrix = cdist(
        all_points[:n_pred, :, :].reshape(-1, 3),
        all_points[n_pred:n_pred + n_gt, :, :].reshape(-1, 3),
        "euclidean",
    )
    distance_matrix = distance_matrix.reshape(n_pred, sample_points, n_gt, sample_points)
    distance_matrix = np.transpose(distance_matrix, axes=(0, 2, 1, 3))
    h_pt_value = distance_matrix.min(-1).max(-1, keepdims=True)
    h_tp_value = distance_matrix.min(-2).max(-1, keepdims=True)
    hausdorff_matrix = np.concatenate((h_pt_value, h_tp_value), axis=-1)
    hausdorff_matrix = hausdorff_matrix.max(-1)
    return hausdorff_matrix


def graph_edit_distance(pd_vertices, pd_edges, gt_vertices, gt_edges, wed_v):
    wed_e = 0.0
    if len(pd_vertices) > 0:
        distances = cdist(pd_vertices, gt_vertices)
        wed_v += float(np.min(distances, axis=1).sum())
        min_indices = np.argmin(distances, axis=1)
        pd_vertices = pd_vertices.copy()
        for i, index in enumerate(min_indices):
            pd_vertices[i] = gt_vertices[index]
        unique_pd_vertices = np.unique(pd_vertices, axis=0)
        renew_pd_edges = pd_edges.copy()
        for i, point in enumerate(unique_pd_vertices):
            v_indices = np.where((pd_vertices == point).all(axis=1))[0]
            for v_index in v_indices:
                renew_pd_edges[pd_edges == v_index] = i
        renew_pd_edges = np.unique(renew_pd_edges, axis=0)
        gt_edges_copy = gt_edges.copy()
        for edge in renew_pd_edges:
            e1_index = np.where((gt_vertices == unique_pd_vertices[edge[0]]).all(axis=1))[0]
            e2_index = np.where((gt_vertices == unique_pd_vertices[edge[1]]).all(axis=1))[0]
            if len(e1_index) == 0 or len(e2_index) == 0:
                wed_e += np.linalg.norm(unique_pd_vertices[edge[0]] - unique_pd_vertices[edge[1]])
                continue
            matched_edge = np.array(sorted([e1_index[0], e2_index[0]]))
            exists = np.where((gt_edges == matched_edge).all(axis=1))[0]
            if len(exists):
                mask = np.any(gt_edges_copy != matched_edge, axis=1)
                gt_edges_copy = gt_edges_copy[mask]
            else:
                wed_e += np.linalg.norm(unique_pd_vertices[edge[0]] - unique_pd_vertices[edge[1]])
    else:
        gt_edges_copy = gt_edges.copy()
        wed_v = 0.0
    for edge in gt_edges_copy:
        wed_e += np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]])
    sum_distance = 0.0
    for edge in gt_edges:
        sum_distance += np.linalg.norm(gt_vertices[edge[0]] - gt_vertices[edge[1]])
    if sum_distance <= 1e-12:
        return 0.0
    return float((wed_e + wed_v) / sum_distance)


def computer_edges(edges, vertices):
    index = []
    for edge in edges:
        indices = []
        for point in edge:
            matching_indices = np.where((vertices == point).all(axis=1))[0]
            indices.append(matching_indices[0] if len(matching_indices) > 0 else -1)
        index.append(indices)
    if len(index) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.sort(np.asarray(index, dtype=np.int32), axis=-1)


def remove_corners(corner_a, corner_b):
    if len(corner_a) == 0:
        return corner_a.reshape(0, 3)
    if len(corner_b) == 0:
        return corner_a.copy()
    corner_a_view = corner_a.view([("", corner_a.dtype)] * corner_a.shape[1])
    corner_b_view = corner_b.view([("", corner_b.dtype)] * corner_b.shape[1])
    corner = np.setdiff1d(corner_a_view, corner_b_view).view(corner_a.dtype).reshape(-1, corner_a.shape[1])
    return corner


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


class APCalculator:
    def __init__(self, distance_thresh=0.1, confidence_thresh=0.7):
        self.distance_thresh = distance_thresh
        self.confidence_thresh = confidence_thresh
        self.sample_count = 0
        self.reset()

    def compute_metrics(self, batch):
        batch_size = len(batch["predicted_vertices"])
        self.sample_count += batch_size

        batch_predicted_corners = batch["predicted_vertices"]
        batch_predicted_edges = batch["predicted_edges"]
        batch_pred_edges_vertices = batch["pred_edges_vertices"]
        batch_label_corners = batch["wf_vertices"]
        batch_label_edges = batch["wf_edges"]
        batch_label_edges_vertices = batch["wf_edges_vertices"]

        for b in range(batch_size):
            predicted_corners = np.asarray(batch_predicted_corners[b], dtype=np.float64).reshape(-1, 3)
            predicted_edges = np.asarray(batch_predicted_edges[b], dtype=np.int32).reshape(-1, 2)
            pred_edges_vertices = np.asarray(batch_pred_edges_vertices[b], dtype=np.float64).reshape(-1, 2, 3)
            label_corners = np.asarray(batch_label_corners[b], dtype=np.float64).reshape(-1, 3)
            label_edges = np.asarray(batch_label_edges[b], dtype=np.int32).reshape(-1, 2)
            label_edges_vertices = np.asarray(batch_label_edges_vertices[b], dtype=np.float64).reshape(-1, 2, 3)

            tp_edges = 0
            tp_fp_edges = len(predicted_edges)
            tp_fn_edges = len(label_edges)
            distances = 0.0

            pr_corners = np.zeros((0, 2, 3), dtype=np.float64)
            gt_corners = np.zeros((0, 2, 3), dtype=np.float64)
            matched_pred_edge_indices = np.zeros((0,), dtype=np.int32)
            matched_gt_edge_indices = np.zeros((0,), dtype=np.int32)

            if len(predicted_edges) != 0 and len(label_edges_vertices) != 0:
                edge_distance = hausdorff_distance_line(pred_edges_vertices, label_edges_vertices)
                predict_indices, label_indices = linear_sum_assignment(edge_distance)
                edge_mask = edge_distance[predict_indices, label_indices] <= self.distance_thresh
                matched_pred_edge_indices = predict_indices[edge_mask]
                matched_gt_edge_indices = label_indices[edge_mask]
                pr_corners = pred_edges_vertices[matched_pred_edge_indices]
                gt_corners = label_edges_vertices[matched_gt_edge_indices]
                tp_edges = int(edge_mask.sum())

                un_match_pr_corners = remove_corners(
                    predicted_corners,
                    np.unique(pr_corners.reshape(-1, 3), axis=0) if len(pr_corners) else np.zeros((0, 3))
                )
                un_match_gt_corners = remove_corners(
                    label_corners,
                    np.unique(gt_corners.reshape(-1, 3), axis=0) if len(gt_corners) else np.zeros((0, 3))
                )

                additional_corner_matches = 0
                if len(un_match_pr_corners) > 0 and len(un_match_gt_corners) > 0:
                    distance_matrix = cdist(un_match_pr_corners, un_match_gt_corners)
                    un_match_predict_indices, un_match_label_indices = linear_sum_assignment(distance_matrix)
                    un_match_mask = (
                        distance_matrix[un_match_predict_indices, un_match_label_indices] <= self.distance_thresh
                    )
                    distances += float(
                        distance_matrix[
                            un_match_predict_indices[un_match_mask],
                            un_match_label_indices[un_match_mask],
                        ].sum()
                    )
                    additional_corner_matches = int(un_match_mask.sum())

                matched_pred_vertices = (
                    np.unique(pr_corners.reshape(-1, 3), axis=0) if len(pr_corners) else np.zeros((0, 3), dtype=np.float64)
                )
                matched_gt_vertices = (
                    np.unique(gt_corners.reshape(-1, 3), axis=0) if len(gt_corners) else np.zeros((0, 3), dtype=np.float64)
                )

                tp_corners = len(matched_pred_vertices) + additional_corner_matches
                tp_fp_corners = len(predicted_corners)
                tp_fn_corners = len(label_corners)

                if len(matched_pred_vertices) > 0 and len(matched_gt_vertices) > 0:
                    distance_matrix = cdist(matched_pred_vertices, matched_gt_vertices)
                    distances += float(np.min(distance_matrix, axis=1).sum())

                if len(matched_pred_edge_indices) > 0:
                    adjusted_pred_edges_vertices = pred_edges_vertices.copy()
                    adjusted_pred_edges_vertices[matched_pred_edge_indices] = label_edges_vertices[matched_gt_edge_indices]
                    predicted_corners_for_wed = np.unique(label_edges_vertices.reshape(-1, 3), axis=0)
                    submission_edges = computer_edges(label_edges_vertices, predicted_corners_for_wed)
                    wed = graph_edit_distance(
                        predicted_corners_for_wed,
                        submission_edges.copy(),
                        label_corners.copy(),
                        label_edges.copy(),
                        distances,
                    )
                else:
                    wed = graph_edit_distance(
                        np.zeros((0, 3), dtype=np.float64),
                        np.zeros((0, 2), dtype=np.int32),
                        label_corners.copy(),
                        label_edges.copy(),
                        distances,
                    )
            else:
                if len(predicted_corners) > 0 and len(label_corners) > 0:
                    distance_matrix = cdist(predicted_corners, label_corners)
                    predict_indices, label_indices = linear_sum_assignment(distance_matrix)
                    mask = distance_matrix[predict_indices, label_indices] <= self.distance_thresh
                    distances = float(distance_matrix[predict_indices[mask], label_indices[mask]].sum())
                    tp_corners = int(mask.sum())
                else:
                    distances = 0.0
                    tp_corners = 0
                tp_fp_corners = len(predicted_corners)
                tp_fn_corners = len(label_corners)
                tp_edges = 0
                tp_fp_edges = 0
                tp_fn_edges = len(label_edges)
                wed = 1.0

            self.ap_dict["tp_corners"] += tp_corners
            self.ap_dict["tp_fp_corners"] += tp_fp_corners
            self.ap_dict["tp_fn_corners"] += tp_fn_corners
            self.ap_dict["distance"] += distances
            self.ap_dict["wed"] += wed
            self.ap_dict["tp_edges"] += tp_edges
            self.ap_dict["tp_fp_edges"] += tp_fp_edges
            self.ap_dict["tp_fn_edges"] += tp_fn_edges

    def output_accuracy(self):
        self.ap_dict["average_corner_offset"] = safe_divide(self.ap_dict["distance"], self.ap_dict["tp_corners"])
        self.ap_dict["average_wed"] = safe_divide(self.ap_dict["wed"], self.sample_count)
        self.ap_dict["corners_precision"] = safe_divide(self.ap_dict["tp_corners"], self.ap_dict["tp_fp_corners"])
        self.ap_dict["corners_recall"] = safe_divide(self.ap_dict["tp_corners"], self.ap_dict["tp_fn_corners"])
        cp = self.ap_dict["corners_precision"]
        cr = self.ap_dict["corners_recall"]
        self.ap_dict["corners_f1"] = safe_divide(2 * cp * cr, cp + cr)
        self.ap_dict["edges_precision"] = safe_divide(self.ap_dict["tp_edges"], self.ap_dict["tp_fp_edges"])
        self.ap_dict["edges_recall"] = safe_divide(self.ap_dict["tp_edges"], self.ap_dict["tp_fn_edges"])
        ep = self.ap_dict["edges_precision"]
        er = self.ap_dict["edges_recall"]
        self.ap_dict["edges_f1"] = safe_divide(2 * ep * er, ep + er)
        return {
            "ACO": self.ap_dict["average_corner_offset"],
            "WED": self.ap_dict["average_wed"],
            "CP": self.ap_dict["corners_precision"],
            "CR": self.ap_dict["corners_recall"],
            "CF1": self.ap_dict["corners_f1"],
            "EP": self.ap_dict["edges_precision"],
            "ER": self.ap_dict["edges_recall"],
            "EF1": self.ap_dict["edges_f1"],
            "support_samples": self.sample_count,
            "tp_corners": self.ap_dict["tp_corners"],
            "tp_fp_corners": self.ap_dict["tp_fp_corners"],
            "tp_fn_corners": self.ap_dict["tp_fn_corners"],
            "tp_edges": self.ap_dict["tp_edges"],
            "tp_fp_edges": self.ap_dict["tp_fp_edges"],
            "tp_fn_edges": self.ap_dict["tp_fn_edges"],
        }

    def reset(self):
        self.ap_dict = {
            "tp_corners": 0,
            "tp_fp_corners": 0,
            "tp_fn_corners": 0,
            "distance": 0.0,
            "tp_edges": 0,
            "wed": 0.0,
            "tp_fp_edges": 0,
            "tp_fn_edges": 0,
            "average_corner_offset": 0.0,
            "corners_precision": 0.0,
            "corners_recall": 0.0,
            "corners_f1": 0.0,
            "edges_precision": 0.0,
            "edges_recall": 0.0,
            "edges_f1": 0.0,
        }
        self.sample_count = 0


def build_batch(pred_obj, gt_obj):
    pred = load_wireframe_obj(pred_obj)
    gt = load_wireframe_obj(gt_obj)
    return {
        "predicted_vertices": [pred["vertices"]],
        "predicted_edges": [pred["edges"]],
        "pred_edges_vertices": [pred["edge_vertices"]],
        "wf_vertices": [gt["vertices"]],
        "wf_edges": [gt["edges"]],
        "wf_edges_vertices": [gt["edge_vertices"]],
    }


def resolve_pairs(pred_dir, gt_dir, pred_ext, gt_ext, names_file):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    # Pred files are named like "foo_pred.obj"; GT files are "foo.obj".
    # Strip the _pred suffix from pred stems so they match GT stems.
    pred_suffix = pred_ext.lstrip("*")
    _pred_tag = "_pred"
    raw_pred_map = {}
    for path in pred_dir.glob(f"*{pred_ext}"):
        stem = path.stem
        if stem.endswith(_pred_tag):
            stem = stem[:-len(_pred_tag)]
        raw_pred_map[stem] = path
    raw_gt_map = {path.stem: path for path in gt_dir.glob(f"*{gt_ext}")}
    if names_file:
        with open(names_file, "r", encoding="utf-8") as handle:
            selected = {line.strip() for line in handle if line.strip()}
        pred_map = {name: path for name, path in raw_pred_map.items() if name in selected}
        gt_map = {name: path for name, path in raw_gt_map.items() if name in selected}
    else:
        pred_map = raw_pred_map
        gt_map = raw_gt_map
    common_names = sorted(pred_map.keys() & gt_map.keys())
    missing_pred = sorted(gt_map.keys() - pred_map.keys())
    missing_gt = sorted(pred_map.keys() - gt_map.keys())
    return [(name, pred_map[name], gt_map[name]) for name in common_names], missing_pred, missing_gt


def _is_patch_name(stem, base_name):
    """Check if stem is a patch of base_name: base_name + _N (digits only)."""
    if stem == base_name:
        return True
    suffix = stem[len(base_name):]
    if suffix.startswith("_") and suffix[1:].isdigit():
        return True
    return False


def _strip_patch_suffix(stem):
    """Strip trailing _N (digits) to get base point cloud name. Returns (base, is_patch)."""
    import re
    m = re.match(r"^(.+)_(\d+)$", stem)
    if m:
        return m.group(1), True
    return stem, False


def merge_patch_preds_to_pointcloud(pred_dir, merge_th=0.02):
    """Merge per-patch _pred.obj files into per-point-cloud _pred.obj files.

    Groups files like 'building1_0_pred.obj', 'building1_1_pred.obj'
    into a single 'building1_pred.obj' by merging vertices and edges.
    Returns the number of merged point clouds.
    """
    import re
    from collections import defaultdict

    pred_dir = Path(pred_dir)
    pred_files = sorted(pred_dir.glob("*_pred.obj"))
    if not pred_files:
        return 0

    # Group by base point cloud name
    groups = defaultdict(list)
    for f in pred_files:
        stem = f.stem  # e.g. "building1_0_pred"
        # Strip _pred suffix
        if stem.endswith("_pred"):
            stem = stem[:-len("_pred")]
        # Try to extract base name
        base, is_patch = _strip_patch_suffix(stem)
        groups[base].append(f)

    # Only merge groups that have multiple patches; single-patch groups are already fine
    merged_count = 0
    for base_name, files in groups.items():
        if len(files) == 1:
            # Single patch: rename to base_name_pred.obj if needed
            expected_name = pred_dir / f"{base_name}_pred.obj"
            if files[0] != expected_name:
                try:
                    files[0].rename(expected_name)
                    merged_count += 1
                except OSError:
                    pass
            continue

        # Merge multiple patches
        all_vertices = []
        all_edges = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts[0] == "v":
                        all_vertices.append([float(v) for v in parts[1:4]])
                    elif parts[0] == "l":
                        idxs = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                        for s, e in zip(idxs[:-1], idxs[1:]):
                            if s != e:
                                all_edges.append(tuple(sorted((s, e))))

        if not all_vertices:
            continue

        all_vertices = np.array(all_vertices, dtype=np.float64)

        # Remap edge indices to be contiguous 0-based
        # First, merge close vertices
        n = len(all_vertices)
        merged_vertices = []
        vertex_map = {}  # old_idx -> new_idx
        for i in range(n):
            if i in vertex_map:
                continue
            # Find all close vertices
            for j in range(i + 1, n):
                if j in vertex_map:
                    continue
                if np.linalg.norm(all_vertices[i] - all_vertices[j]) < merge_th:
                    vertex_map[j] = i
            vertex_map[i] = i

        # Build new vertex list (only keep representative vertices)
        old_to_new = {}
        new_idx = 0
        for i in range(n):
            rep = vertex_map.get(i, i)
            if rep not in old_to_new:
                old_to_new[rep] = new_idx
                merged_vertices.append(all_vertices[rep])
                new_idx += 1
            old_to_new[i] = old_to_new[rep]

        merged_vertices = np.array(merged_vertices, dtype=np.float64)

        # Remap edges
        new_edges = set()
        for edge in all_edges:
            s, e = edge
            if s >= n or e >= n:
                continue
            ns, ne = old_to_new.get(s, s), old_to_new.get(e, e)
            if ns != ne:
                new_edges.add(tuple(sorted((ns, ne))))

        if not new_edges or len(merged_vertices) == 0:
            continue

        # Export merged OBJ
        merged_path = pred_dir / f"{base_name}_pred.obj"
        with open(merged_path, "w", encoding="utf-8") as fh:
            for v in merged_vertices:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for s, e in sorted(new_edges):
                fh.write(f"l {s + 1} {e + 1}\n")

        # Remove old per-patch OBJ files
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass

        merged_count += 1

    return merged_count


def write_eval_csv(output_csv, results):
    fieldnames = [
        "name", "ACO", "WED", "CP", "CR", "CF1", "EP", "ER", "EF1",
        "support_samples", "tp_corners", "tp_fp_corners", "tp_fn_corners",
        "tp_edges", "tp_fp_edges", "tp_fn_edges",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_ap_evaluation(pred_dir, gt_dir, output_json, output_csv, distance_thresh, names_file):
    # First, merge per-patch _pred.obj into per-point-cloud _pred.obj
    merged = merge_patch_preds_to_pointcloud(pred_dir)
    if merged > 0:
        print(f"Merged {merged} point clouds from patch-level OBJ files.")

    pairs, missing_pred, missing_gt = resolve_pairs(
        pred_dir, gt_dir, "_pred.obj", ".obj", names_file
    )
    if not pairs:
        print("Warning: No matched prediction/ground-truth OBJ pairs found for APCalculator evaluation.")
        return

    overall_calculator = APCalculator(distance_thresh=distance_thresh)
    per_file_results = []

    for index, (name, pred_path, gt_path) in enumerate(pairs, start=1):
        batch = build_batch(pred_path, gt_path)
        overall_calculator.compute_metrics(batch)
        single_calculator = APCalculator(distance_thresh=distance_thresh)
        single_calculator.compute_metrics(batch)
        result = {"name": name}
        result.update(single_calculator.output_accuracy())
        per_file_results.append(result)

    summary = overall_calculator.output_accuracy()

    print("\n=== APCalculator Precision Metrics ===")
    print(f"Pairs evaluated: {len(pairs)}")
    print(f"ACO (Avg Corner Offset): {summary['ACO']:.6f}")
    print(f"CP (Corner Precision):  {summary['CP']:.4f}")
    print(f"CR (Corner Recall):     {summary['CR']:.4f}")
    print(f"CF1 (Corner F1):        {summary['CF1']:.4f}")
    print(f"EP (Edge Precision):    {summary['EP']:.4f}")
    print(f"ER (Edge Recall):       {summary['ER']:.4f}")
    print(f"EF1 (Edge F1):          {summary['EF1']:.4f}")
    print(f"WED (Wireframe Edit Dist): {summary['WED']:.6f}")
    print("=====================================\n")

    if missing_pred:
        print(f"Warning: {len(missing_pred)} GT files have no matching prediction.")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} prediction files have no matching GT.")

    if output_csv:
        write_eval_csv(output_csv, per_file_results)
        print(f"Wrote per-file AP metrics to: {output_csv}")

    if output_json:
        payload = {
            "summary": summary,
            "missing_predictions": missing_pred,
            "missing_ground_truth": missing_gt,
            "per_file": per_file_results,
        }
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote AP metrics JSON to: {output_json}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process predicted lines and export wireframe OBJ files.')
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--vertex_prob_th', type=float, default=0.5)
    parser.add_argument('--vertex_nms_th', type=float, default=0.01)
    parser.add_argument('--line_prob_th', type=float, default=0.5)
    parser.add_argument('--line_len_th', type=float, default=0.02)
    parser.add_argument('--line_nms_th', type=float, default=0.05)
    parser.add_argument('--merge_th', type=float, default=0.02)
    parser.add_argument('--test_list', type=str, default='', help='Path to test_list.txt containing point cloud names to filter.')
    parser.add_argument('--gt_dir', type=str, default='', help='GT OBJ directory for APCalculator evaluation.')
    parser.add_argument('--eval_json', type=str, default='', help='Output JSON path for APCalculator evaluation results.')
    parser.add_argument('--eval_csv', type=str, default='', help='Output CSV path for per-file APCalculator results.')
    parser.add_argument('--eval_distance_thresh', type=float, default=0.1, help='Distance threshold for APCalculator matching.')
    args = parser.parse_args()

    patch_size = args.patch_size
    sigma = args.sigma
    clip = args.clip

    result_dir = args.result_dir or os.path.join(curr_dir, f'run_test_result/patch{patch_size}sigma{sigma}clip{clip}')
    save_to_dir = args.save_dir or os.path.join(curr_dir, f'visualize_line/patch{patch_size}sigma{sigma}clip{clip}')
    os.makedirs(save_to_dir, exist_ok=True)

    vertex_pred_list = glob(os.path.join(result_dir, '*_vertex.txt'))
    vertex_pred_list.sort()

    if args.test_list:
        import re
        with open(args.test_list, 'r', encoding='utf-8') as f:
            filter_names = {line.strip() for line in f if line.strip()}
        print(f'Loaded {len(filter_names)} names from test_list: {args.test_list}')
        patterns = {name: re.compile(r'^' + re.escape(name) + r'(?:_\d+)?$') for name in filter_names}
        filtered_list = []
        for fpath in vertex_pred_list:
            stem = os.path.basename(fpath).replace('_vertex.txt', '')
            for name, pat in patterns.items():
                if pat.match(stem):
                    filtered_list.append(fpath)
                    break
        print(f'Filtered vertex files: {len(vertex_pred_list)} -> {len(filtered_list)}')
        vertex_pred_list = filtered_list
    exported = 0
    for vertex_pred_f in vertex_pred_list:
        vertex_prob_f = vertex_pred_f.replace('_vertex.txt', '_vprobs.txt')
        line_pred_f = vertex_pred_f.replace('_vertex.txt', '_line.txt')
        line_prob_f = vertex_pred_f.replace('_vertex.txt', '_lprobs.txt')
        required_files = [vertex_pred_f, vertex_prob_f, line_pred_f, line_prob_f]
        if not all(_file_nonempty(p) for p in required_files):
            continue

        # load data
        vertex_pred = np.loadtxt(vertex_pred_f)
        vertex_probs = np.loadtxt(vertex_prob_f)
        line_pred = np.loadtxt(line_pred_f, dtype=np.int32)
        line_probs = np.loadtxt(line_prob_f)
        if np.size(vertex_pred) == 0 or np.size(line_pred) == 0:
            continue
        if np.ndim(vertex_pred) == 1:
            vertex_pred = np.expand_dims(vertex_pred, 0)
        if np.ndim(vertex_probs) == 0:
            vertex_probs = np.expand_dims(vertex_probs, 0)
        if len(line_pred.shape) == 1:
            line_pred = np.expand_dims(line_pred, 0)
            line_probs = np.expand_dims(line_probs, 0)
        
        # post-processing
        line_pred, line_probs = filter_prob_vertex(vertex_pred, vertex_probs, line_pred, line_probs, prob_th=args.vertex_prob_th)
        line_pred, line_probs = filter_nms_vertex(vertex_pred, vertex_probs, line_pred, line_probs, nms_th=args.vertex_nms_th)
        # vertex_pred, vertex_probs = merge_vertex(vertex_pred, vertex_probs, merge_th=0.04)
        line_pred, line_probs = filter_prob_line(line_pred, line_probs, prob_th=args.line_prob_th)
        line_pred, line_probs = filter_short_line(vertex_pred, line_pred, line_probs, len_th=args.line_len_th)
        line_pred, line_probs = filter_nms_line(vertex_pred, line_pred, line_probs, nms_th=args.line_nms_th)
        vertex_pred, vertex_probs = merge_vertex(vertex_pred, vertex_probs, merge_th=args.merge_th)
        if np.size(line_pred) == 0:
            continue

        vertex_pred, line_pred = remove_extra_vertex(vertex_pred, line_pred)
        if np.size(vertex_pred) == 0 or np.size(line_pred) == 0:
            continue

        save_to_path = os.path.join(save_to_dir, os.path.basename(vertex_pred_f).replace('_vertex.txt', '_pred.obj'))
        line_to_obj(vertex_pred, line_pred, save_to_path)
        exported += 1

    print(f'Done. Exported {exported} OBJ files to {save_to_dir}')

    # Run APCalculator evaluation if GT directory is provided
    if args.gt_dir:
        print(f'Running APCalculator evaluation against GT dir: {args.gt_dir}')
        eval_names_file = args.test_list if args.test_list else ''
        run_ap_evaluation(
            pred_dir=save_to_dir,
            gt_dir=args.gt_dir,
            output_json=args.eval_json if args.eval_json else os.path.join(save_to_dir, 'ap_metrics.json'),
            output_csv=args.eval_csv if args.eval_csv else os.path.join(save_to_dir, 'ap_metrics.csv'),
            distance_thresh=args.eval_distance_thresh,
            names_file=eval_names_file,
        )
