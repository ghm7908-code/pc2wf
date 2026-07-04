import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.abspath(os.path.join(curr_dir, '..')))
from tqdm import tqdm
import random
import numpy as np
from glob import glob
import argparse
from scipy.spatial import cKDTree


def _safe_loadtxt(path, dtype, empty_shape=None):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        if empty_shape is None:
            return np.empty((0,), dtype=dtype)
        return np.empty(empty_shape, dtype=dtype)
    array = np.loadtxt(path, dtype=dtype)
    array = np.asarray(array, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _ensure_row_matrix(array, cols, dtype):
    array = np.asarray(array, dtype=dtype)
    if array.size == 0:
        return np.empty((0, cols), dtype=dtype)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _pick_latest_checkpoint(checkpoint_dir, prefix, patch_size):
    pattern = os.path.join(checkpoint_dir, f'{prefix}_patchSize{patch_size}_*_Val.pth')
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found for pattern: {pattern}')
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def load_models(backbone_pth, patch_pth, vertex_pth, line_pth, device):
    import torch
    from train_end2end import patchNet, vertexNet, lineNet
    from model.resunet import ResUNetBN2C

    backbone_net = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
    backbone_net.load_state_dict(torch.load(backbone_pth, map_location=device))
    backbone_net = backbone_net.to(device)
    backbone_net.eval()

    patch_net = patchNet()
    patch_net.load_state_dict(torch.load(patch_pth, map_location=device))
    patch_net = patch_net.to(device)
    patch_net.eval()

    vertex_net = vertexNet()
    vertex_net.load_state_dict(torch.load(vertex_pth, map_location=device))
    vertex_net = vertex_net.to(device)
    vertex_net.eval()

    line_net = lineNet()
    line_net.load_state_dict(torch.load(line_pth, map_location=device))
    line_net = line_net.to(device)
    line_net.eval()

    return backbone_net, patch_net, vertex_net, line_net


def predict(
    test_file_path,
    backbone_net,
    patch_net,
    vertex_net,
    line_net,
    device,
    patch_prob_threshold=0.85,
    line_prob_threshold=0.5,
    vertex_nms_threshold=0.01,
    line_dist_threshold=0.03,
):
    import torch
    import MinkowskiEngine as ME
    from model.me_compat import sparse_tensor

    '''first, load test_data and feed into backbone_net'''
    pc_down = _ensure_row_matrix(
        _safe_loadtxt(test_file_path, dtype=np.float32, empty_shape=(0, 3)),
        cols=3,
        dtype=np.float32,
    )
    feats_raw = _safe_loadtxt(test_file_path.replace('.down', '.feats'), dtype=np.float32, empty_shape=(0, 1))
    feats_raw = np.asarray(feats_raw, dtype=np.float32)
    if feats_raw.size == 0:
        feats = np.empty((0, 1), dtype=np.float32)
    elif feats_raw.ndim == 1:
        feats = np.expand_dims(feats_raw, 1)
    else:
        feats = feats_raw
    coords = _ensure_row_matrix(
        _safe_loadtxt(test_file_path.replace('.down', '.coords'), dtype=np.int32, empty_shape=(0, 4)),
        cols=4,
        dtype=np.int32,
    )
    patch_index = _safe_loadtxt(test_file_path.replace('.down', '.patch_index'), dtype=np.int32, empty_shape=(0, 0))
    patch_index = np.asarray(patch_index, dtype=np.int32)
    if patch_index.size == 0:
        patch_index = np.empty((0, 0), dtype=np.int32)
    elif patch_index.ndim == 1:
        patch_index = patch_index.reshape(1, -1)

    if len(pc_down) == 0 or len(feats) == 0 or len(coords) == 0 or len(patch_index) == 0:
        return np.empty((0, 3)), np.empty((0,)), np.empty((0, 2), dtype=np.int32), np.empty((0,))

    # Build KD-tree once for fast batched nearest-neighbour queries
    kdtree = cKDTree(pc_down.astype(np.float64))

    with torch.inference_mode():
        stensor = sparse_tensor(
            ME,
            torch.from_numpy(feats).float(),
            coordinates=torch.from_numpy(coords),
            device=device,
        )
        features = backbone_net(stensor).F

        '''second, feed into patch_net to find patches with vertex'''
        patch_index = torch.from_numpy(patch_index).long()
        if len(patch_index.shape) == 1:
            patch_index = patch_index.unsqueeze(-1)
        valid_rows = torch.all((patch_index >= 0) & (patch_index < features.shape[0]), dim=1)
        patch_index = patch_index[valid_rows]

        if len(patch_index) == 0:
            return np.empty((0, 3)), np.empty((0,)), np.empty((0, 2), dtype=np.int32), np.empty((0,))

        patch_index = patch_index.to(device)
        pc_down_t = torch.from_numpy(pc_down).to(device)
        batch_features = features[patch_index]
        batch_coords = pc_down_t[patch_index]
        batch_input_patch = torch.cat([batch_coords, batch_features], 2).transpose(1, 2)
        batch_output_patch = patch_net(batch_input_patch)

        # select patches with positive vertex
        predicted_patch_prob = torch.sigmoid(batch_output_patch.squeeze())
        if predicted_patch_prob.ndim == 0:
            predicted_patch_prob = predicted_patch_prob.unsqueeze(0)
        positive_mask = predicted_patch_prob > patch_prob_threshold
        if not bool(torch.any(positive_mask)):
            return np.empty((0, 3)), np.empty((0,)), np.empty((0, 2), dtype=np.int32), np.empty((0,))
        batch_input_vertex = batch_input_patch[positive_mask]
        batch_input_vertex_prob = predicted_patch_prob[positive_mask].detach().cpu().numpy().astype(np.float32)

        '''third, feed into vertex_net to produce new vertex'''
        batch_output_vertex = vertex_net(batch_input_vertex)
        batch_output_vertex_coord = batch_output_vertex
        predicted_vertex_list_all = batch_output_vertex_coord.detach().cpu().numpy()

        # NMS to select vertex (fixed: use set for O(1) membership)
        nms_threshhold = vertex_nms_threshold
        dropped_vertex_index = set()
        for i in range(len(predicted_vertex_list_all)):
            if i in dropped_vertex_index:
                continue
            dist_all = np.linalg.norm(predicted_vertex_list_all - predicted_vertex_list_all[i], axis=1)
            same_region_indexes = (dist_all < nms_threshhold).nonzero()
            for same_region_i in same_region_indexes[0]:
                if same_region_i == i:
                    continue
                if batch_input_vertex_prob[same_region_i] <= batch_input_vertex_prob[i]:
                    dropped_vertex_index.add(same_region_i)
                else:
                    dropped_vertex_index.add(i)
        selected_vertex_index = [i for i in range(len(predicted_vertex_list_all)) if i not in dropped_vertex_index]
        batch_output_vertex_coord = batch_output_vertex_coord[selected_vertex_index]
        batch_input_vertex_prob = np.array(batch_input_vertex_prob, dtype=np.float32)[selected_vertex_index]

        predicted_vertex_list = batch_output_vertex_coord.detach().cpu().numpy()
        predicted_vertex_probs = np.array(batch_input_vertex_prob)

        n_vertices = len(predicted_vertex_list)
        if n_vertices == 0:
            return np.empty((0, 3)), np.empty((0,)), np.empty((0, 2), dtype=np.int32), np.empty((0,))

        # Find nearest point in pc_down for each predicted vertex (batched)
        _, nn_indices = kdtree.query(predicted_vertex_list, k=1)
        nn_indices = np.atleast_1d(nn_indices)
        predicted_vertex_features = [features[idx] for idx in nn_indices]

        '''fourth, feed into line_net to predict lines — optimized with batched KD-tree'''
        point_num_in_line = 30
        input_line_features = predicted_vertex_features

        # Generate all unique vertex pairs
        if n_vertices < 2:
            return predicted_vertex_list, predicted_vertex_probs, np.empty((0, 2), dtype=np.int32), np.empty((0,))

        triu_rows, triu_cols = np.triu_indices(n_vertices, k=1)
        n_pairs = len(triu_rows)

        # ── Step A: batch midpoint check via KD-tree ──
        v_np = predicted_vertex_list.astype(np.float64)
        all_midpoints = (v_np[triu_rows] + v_np[triu_cols]) * 0.5
        mid_dists, _ = kdtree.query(all_midpoints, k=1)
        mid_valid = mid_dists < line_dist_threshold

        if not np.any(mid_valid):
            return predicted_vertex_list, predicted_vertex_probs, np.empty((0, 2), dtype=np.int32), np.empty((0,))

        valid_pairs_i = triu_rows[mid_valid]
        valid_pairs_j = triu_cols[mid_valid]
        n_valid = len(valid_pairs_i)

        # ── Step B: batch interpolation check via KD-tree ──
        # Generate all interpolation points for all valid pairs at once
        t_vals = (np.arange(1, point_num_in_line + 1, dtype=np.float64) / (point_num_in_line + 1))
        t_vals = t_vals.reshape(1, point_num_in_line, 1)  # (1, 30, 1)

        v_i = v_np[valid_pairs_i]  # (n_valid, 3)
        v_j = v_np[valid_pairs_j]  # (n_valid, 3)

        # interp_points[k, p, :] = t[p] * v_i[k] + (1-t[p]) * v_j[k]
        interp_points = (
            v_i[:, np.newaxis, :] * t_vals
            + v_j[:, np.newaxis, :] * (1.0 - t_vals)
        )  # (n_valid, 30, 3)

        # Flatten to (n_valid * point_num_in_line, 3) for single KD-tree query
        interp_flat = interp_points.reshape(-1, 3)
        interp_dists, interp_indices = kdtree.query(interp_flat, k=1)
        interp_dists = interp_dists.reshape(n_valid, point_num_in_line)
        interp_indices = interp_indices.reshape(n_valid, point_num_in_line)

        # A pair is valid only if ALL interpolation points are within threshold
        pair_valid = np.all(interp_dists < line_dist_threshold, axis=1)

        if not np.any(pair_valid):
            return predicted_vertex_list, predicted_vertex_probs, np.empty((0, 2), dtype=np.int32), np.empty((0,))

        final_i = valid_pairs_i[pair_valid]
        final_j = valid_pairs_j[pair_valid]
        final_interp_idx = interp_indices[pair_valid]  # (n_final, point_num_in_line)
        final_interp_dist = interp_dists[pair_valid]    # (n_final, point_num_in_line)
        n_final = len(final_i)

        # ── Step C: build line_net inputs ──
        batch_input_line = []
        batch_index_line = []
        batch_index_dist = []
        for k in range(n_final):
            tmp = [input_line_features[final_i[k]]]
            for p in range(point_num_in_line):
                tmp.append(features[final_interp_idx[k, p]])
            tmp.append(input_line_features[final_j[k]])
            batch_input_line.append(torch.stack(tmp))
            batch_index_line.append([final_i[k] + 1, final_j[k] + 1])
            batch_index_dist.append(float(np.mean(final_interp_dist[k])))

        if len(batch_input_line) == 0:
            return predicted_vertex_list, predicted_vertex_probs, np.empty((0, 2), dtype=np.int32), np.empty((0,))

        batch_input_line = torch.stack(batch_input_line).transpose(1, 2)
        batch_output_line = line_net(batch_input_line)
        predicted_line_index = torch.sigmoid(batch_output_line.squeeze())
        predicted_line_list = []
        predicted_line_probs = []
        if len(predicted_line_index.shape) == 0:
            predicted_line_index = predicted_line_index.unsqueeze(0)
        for i, predicted_index in enumerate(predicted_line_index):
            predicted_prob = float(predicted_index.detach().cpu().item())
            if predicted_prob > line_prob_threshold:
                predicted_line_list.append(batch_index_line[i])
                predicted_line_probs.append(predicted_prob)
        return np.array(predicted_vertex_list), np.array(predicted_vertex_probs), np.array(predicted_line_list), np.array(predicted_line_probs, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict wireframe vertices and lines on patch test set.')
    parser.add_argument('--data_root', type=str, default=os.path.join(curr_dir, '..', 'abc_data'))
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--backbone_pth', type=str, default='')
    parser.add_argument('--patch_pth', type=str, default='')
    parser.add_argument('--vertex_pth', type=str, default='')
    parser.add_argument('--line_pth', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--patch_prob_th', type=float, default=0.5)
    parser.add_argument('--line_prob_th', type=float, default=0.5)
    parser.add_argument('--vertex_nms_th', type=float, default=0.01)
    parser.add_argument('--line_dist_th', type=float, default=0.03)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--test_list', type=str, default='', help='Path to test_list.txt containing point cloud names to filter.')
    args = parser.parse_args()

    sigma = args.sigma
    clip = args.clip
    patch_size = args.patch_size

    checkpoint_dir = args.checkpoint_dir or os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}')
    backbone_pth = args.backbone_pth or _pick_latest_checkpoint(checkpoint_dir, 'backbone', patch_size)
    patch_pth = args.patch_pth or _pick_latest_checkpoint(checkpoint_dir, 'patchnet', patch_size)
    vertex_pth = args.vertex_pth or _pick_latest_checkpoint(checkpoint_dir, 'vertexnet', patch_size)
    line_pth = args.line_pth or _pick_latest_checkpoint(checkpoint_dir, 'linenet', patch_size)

    save_to_folder = args.save_dir or os.path.join(curr_dir, 'run_test_result', f'patch{patch_size}sigma{sigma}clip{clip}')
    os.makedirs(save_to_folder, exist_ok=True)

    test_glob = os.path.join(args.data_root, f'patches_{patch_size}_noise_sigma{sigma}clip{clip}/{args.split}/*.down')
    print(test_glob)
    print(f'Using checkpoints from: {checkpoint_dir}')
    print(f'backbone: {backbone_pth}')
    print(f'patchnet: {patch_pth}')
    print(f'vertexnet: {vertex_pth}')
    print(f'linenet: {line_pth}')
    print(f'save_to: {save_to_folder}')

    test_file_list = glob(test_glob)
    test_file_list.sort()

    if args.test_list:
        import re
        with open(args.test_list, 'r', encoding='utf-8') as f:
            filter_names = {line.strip() for line in f if line.strip()}
        print(f'Loaded {len(filter_names)} names from test_list: {args.test_list}')
        # Naming: {cloud_name}{patch_digits} with NO separator (e.g. "1001" + "0" = "10010")
        # Match if filename stem starts with a test_list name and remainder is all digits.
        # Use longest match to avoid ambiguity (e.g. "10010" should match "10010", not "1001").
        patterns = {name: re.compile(r'^' + re.escape(name) + r'(\d*)$') for name in filter_names}
        filtered_list = []
        for fpath in test_file_list:
            stem = os.path.basename(fpath).replace('.down', '')
            best_name = None
            best_len = 0
            for name, pat in patterns.items():
                if pat.match(stem) and len(name) > best_len:
                    best_name = name
                    best_len = len(name)
            if best_name:
                filtered_list.append(fpath)
        print(f'Filtered test files: {len(test_file_list)} -> {len(filtered_list)}')
        test_file_list = filtered_list

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_net, patch_net, vertex_net, line_net = load_models(
        backbone_pth,
        patch_pth,
        vertex_pth,
        line_pth,
        device,
    )

    processed = 0
    for test_file in tqdm(test_file_list):
        basename = os.path.basename(test_file).replace('.down', '')
        line_out = os.path.join(save_to_folder, f'{basename}_line.txt')
        if (not args.overwrite) and os.path.exists(line_out):
            continue

        predicted_vertex_list, predicted_vertex_probs, predicted_line_list, predicted_line_probs = predict(
            test_file,
            backbone_net,
            patch_net,
            vertex_net,
            line_net,
            device,
            patch_prob_threshold=args.patch_prob_th,
            line_prob_threshold=args.line_prob_th,
            vertex_nms_threshold=args.vertex_nms_th,
            line_dist_threshold=args.line_dist_th,
        )

        np.savetxt(os.path.join(save_to_folder, f'{basename}_vertex.txt'), predicted_vertex_list)
        np.savetxt(os.path.join(save_to_folder, f'{basename}_vprobs.txt'), predicted_vertex_probs)
        np.savetxt(os.path.join(save_to_folder, f'{basename}_line.txt'), predicted_line_list, fmt='%d')
        np.savetxt(os.path.join(save_to_folder, f'{basename}_lprobs.txt'), predicted_line_probs)
        processed += 1

    print(f'Done. Processed {processed} point clouds.')
