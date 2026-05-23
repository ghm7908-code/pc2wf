import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from glob import glob
import argparse
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
    args = parser.parse_args()

    patch_size = args.patch_size
    sigma = args.sigma
    clip = args.clip

    result_dir = args.result_dir or os.path.join(curr_dir, f'run_test_result/patch{patch_size}sigma{sigma}clip{clip}')
    save_to_dir = args.save_dir or os.path.join(curr_dir, f'visualize_line/patch{patch_size}sigma{sigma}clip{clip}')
    os.makedirs(save_to_dir, exist_ok=True)

    vertex_pred_list = glob(os.path.join(result_dir, '*_vertex.txt'))
    vertex_pred_list.sort()
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
