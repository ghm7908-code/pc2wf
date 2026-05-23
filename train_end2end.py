import os
import time
import numpy as np
from glob import glob
import logging
import argparse
import random
from tqdm import tqdm
from model.resunet import NewResUNet2, NewnewResUNet2, ResUNetBN2C

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import MinkowskiEngine.utils as ME_utils
from datetime import datetime
from tqdm import tqdm
import argparse
from model.me_compat import sparse_tensor


class patchNet(nn.Module):
  def __init__(self, dropout_rate=0.2):
    super(patchNet, self).__init__()
    self.conv1 = nn.Conv1d(35, 35, 1)
    self.batch1 = nn.BatchNorm1d(35)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.conv2 = nn.Conv1d(35, 35, 1)
    self.batch2 = nn.BatchNorm1d(35)
    self.dropout2 = nn.Dropout(dropout_rate)
    self.conv3 = nn.Conv1d(35, 1, 1)

  def forward(self, x):
    x = self.dropout1(F.relu(self.batch1(self.conv1(x))))
    x = self.dropout2(self.batch2(self.conv2(x)))
    x = F.max_pool2d(x, kernel_size=(1, x.size(-1)))
    x = self.conv3(x)
    return x

class vertexNet(nn.Module):
  def __init__(self, dropout_rate=0.1):
    super(vertexNet, self).__init__()
    self.conv1 = nn.Conv1d(35, 35, 1)
    self.batch1 = nn.BatchNorm1d(35)
    self.dropout = nn.Dropout(dropout_rate)
    self.conv2 = nn.Conv1d(35, 1, 1)
    self.weight = nn.Softmax(-1)
    
  def forward(self, x):
    _x = self.dropout(F.relu(self.batch1(self.conv1(x))))
    _x = self.conv2(_x)
    weight = self.weight(_x)
    new_vertex = torch.sum(weight * x[:, :3], -1)
    return new_vertex

class lineNet(nn.Module):
  def __init__(self, dropout_rate=0.3):
    super(lineNet, self).__init__()
    self.f0 = nn.Flatten()
    self.f1 = nn.Linear(8*32, 128)
    self.dropout = nn.Dropout(dropout_rate)
    self.f2 = nn.Linear(128, 1)

  def forward(self, x):
    x = F.max_pool2d(x, kernel_size=(1, 4))
    x = self.f0(x)
    x = self.dropout(F.relu(self.f1(x)))
    x = self.f2(x)
    x = x.unsqueeze(-1)
    return x


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


def _ensure_row_matrix(array, empty_shape):
  array = np.asarray(array)
  if array.size == 0:
    return np.empty(empty_shape, dtype=array.dtype if array.dtype != np.dtype("O") else np.float32)
  if array.ndim == 1:
    return array.reshape(1, -1)
  return array


def _ensure_feature_matrix(array):
  array = np.asarray(array)
  if array.size == 0:
    return np.empty((0, 1), dtype=array.dtype if array.dtype != np.dtype("O") else np.float32)
  if array.ndim == 1:
    return np.expand_dims(array, 1)
  return array


class PointcloudDataset(Dataset):
  def __init__(self, dataset_path, dataset_type='train'):
    self.pointcloud_root = dataset_path
    self.dataset_type = dataset_type
    if dataset_type == 'train':
      data_num = 100000
    else:
      data_num = 100000
    minilinefiles = glob('{}/*.mini_line'.format(self.pointcloud_root))[:data_num]
    self.samples = []
    complete_files = 0
    nonempty_files = 0
    for minilinefile in minilinefiles:
      pointcloudfile = minilinefile.replace('mini_line', 'down')
      featfile = pointcloudfile.replace('down', 'feats')
      coordfile = pointcloudfile.replace('down', 'coords')
      otherindexfile = pointcloudfile.replace('down', 'other_index')
      vertindexfile = pointcloudfile.replace('down', 'vert_index')
      vertgtfile = pointcloudfile.replace('down', 'vert_gt')
      required_files = [pointcloudfile, featfile, coordfile, otherindexfile, vertindexfile, vertgtfile, minilinefile]
      if all(os.path.exists(required_file) for required_file in required_files):
        complete_files += 1
        if all(os.path.getsize(required_file) > 0 for required_file in required_files):
          nonempty_files += 1
          self.samples.append((pointcloudfile, featfile, coordfile, otherindexfile, vertindexfile, vertgtfile, minilinefile))

    print('Dataset {}: kept {}/{} samples ({} complete-file candidates, {} non-empty training-ready candidates).'.format(
      self.pointcloud_root, len(self.samples), len(minilinefiles), complete_files, nonempty_files
    ))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    pointcloudfile, featfile, coordfile, otherindexfile, vertindexfile, vertgtfile, minilinefile = self.samples[index]
    # point cloud after down sampling
    pc_down = _ensure_row_matrix(_safe_loadtxt(pointcloudfile, dtype=np.float32, empty_shape=(0, 3)), (0, 3)) # Ndx3

    # initial features
    feats = _ensure_feature_matrix(_safe_loadtxt(featfile, dtype=np.float32, empty_shape=(0, 1))) # Ndx1

    # coords of pc_down
    coords = _ensure_row_matrix(_safe_loadtxt(coordfile, dtype=np.int32, empty_shape=(0, 4)), (0, 4)) # Ndx4

    patch_other_index = _ensure_row_matrix(_safe_loadtxt(otherindexfile, dtype=np.int32, empty_shape=(0, 0)), (0, 0))
    patch_vert_index = _ensure_row_matrix(_safe_loadtxt(vertindexfile, dtype=np.int32, empty_shape=(0, 0)), (0, 0))
    patch_vert_gt = _ensure_row_matrix(_safe_loadtxt(vertgtfile, dtype=np.float32, empty_shape=(0, 3)), (0, 3))

    # line and label for lines
    mini_line = _ensure_row_matrix(_safe_loadtxt(minilinefile, dtype=np.int32, empty_shape=(0, 0)), (0, 0))

    return pc_down, feats, coords, patch_other_index, patch_vert_index, patch_vert_gt, mini_line
  

def train(
  data_path,
  patch_size=50,
  mini_batch=512,
  nms_th=0.05,
  line_positive_th=0.05,
  line_negative_th=0.10,
  loss_weight=[1.0, 1.0, 1.0],
  sigma=0.01,
  clip=0.02,
  n_epoch=20,
  backbone_lr=5e-5,
  head_lr=3e-4,
  weight_decay=5e-4,
  patch_dropout=0.2,
  vertex_dropout=0.1,
  line_dropout=0.3,
  lr_decay_factor=0.5,
  lr_decay_patience=2,
  min_lr=1e-5,
  early_stop_patience=6,
  early_stop_min_delta=1e-4,
):
  recover_from_last_train = False

  if not os.path.exists(f'./checkpoint_sigma{sigma}clip{clip}'):
    os.mkdir(f'./checkpoint_sigma{sigma}clip{clip}')

  if not os.path.exists('./logs'):
    os.mkdir('./logs')

  log_f = open('./logs/log_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}_sigma{}clip{}_epochs{}_blr{}_hlr{}_wd{}_{}.txt'.format(
    patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2], sigma, clip, n_epoch, backbone_lr, head_lr, weight_decay, datetime.now().strftime("%Y%m%d_%H%M%S")), 'w')

  checkpoint = torch.load('ResUNetBN2C-32feat.pth')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # initialize backbone_net
  backbone_net = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
  if recover_from_last_train:
    backbone_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/backbone_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
  else:
    backbone_net.load_state_dict(checkpoint['state_dict'])
  backbone_net = backbone_net.to(device)
  backbone_net.train()
  
  patch_net = patchNet(dropout_rate=patch_dropout).to(device)
  if recover_from_last_train:
    patch_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/patchnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
  vertex_net = vertexNet(dropout_rate=vertex_dropout).to(device)
  if recover_from_last_train:
    vertex_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/vertexnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
  line_net = lineNet(dropout_rate=line_dropout).to(device)
  if recover_from_last_train:
    line_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/linenet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))

  # loss functions
  patch_weight = [1.0, 2.0]
#   criterion_patch = nn.CrossEntropyLoss(weight=torch.Tensor(np.array(patch_weight))).to(device)
  criterion_patch = nn.BCEWithLogitsLoss().to(device)
  criterion_vertex = nn.MSELoss(reduction='sum').to(device)
#   criterion_line = nn.CrossEntropyLoss().to(device)
  criterion_line = nn.BCEWithLogitsLoss().to(device)

  # train parameters
  optimizer = optim.Adam(
    [
      {'params': backbone_net.parameters(), 'lr': backbone_lr},
      {'params': list(patch_net.parameters()) + list(vertex_net.parameters()) + list(line_net.parameters()), 'lr': head_lr},
    ],
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=lr_decay_factor,
    patience=lr_decay_patience,
    min_lr=min_lr,
  )

  # train_loader
  train_dataset = PointcloudDataset(os.path.join(data_path, f'patches_{patch_size}_noise_sigma{sigma}clip{clip}', 'train'))
  if len(train_dataset) == 0:
    raise RuntimeError('No valid training samples found. Please regenerate patches and check .mini_line/.vert_index/.other_index files.')
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

  # val_loader
  val_dataset = PointcloudDataset(os.path.join(data_path, f'patches_{patch_size}_noise_sigma{sigma}clip{clip}', 'validation'))
  if len(val_dataset) == 0:
    raise RuntimeError('No valid validation samples found. Please regenerate patches for the validation split.')
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

  best_val_loss = 100.0
  best_val_score = -1.0
  epochs_without_improvement = 0
  for epoch in range(n_epoch):      
    backbone_current_lr = optimizer.param_groups[0]['lr']
    head_current_lr = optimizer.param_groups[1]['lr']
    

    ''' ---begin: training---'''
    total_loss = 0.0
    total_loss_patch = 0.0
    total_acc_patch = 0.0
    total_precision_patch = 0.0
    total_recall_patch = 0.0
    total_loss_vertex = 0.0
    total_loss_line = 0.0
    total_acc_line = 0.0
    total_precision_line = 0.0
    total_recall_line = 0.0
    total = 0
    for train_loader_i, data in enumerate(tqdm(train_loader)):
      # load train data
      pc_down, feats, coords, patch_other_index, patch_vert_index, patch_vert_gt, mini_line = data
      pc_down, feats, coords, patch_other_index, patch_vert_index, patch_vert_gt, mini_line = pc_down[0], feats[0], coords[0], patch_other_index[0], patch_vert_index[0], patch_vert_gt[0], mini_line[0]
      pc_down = pc_down.to(device)
      try:
        if len(patch_other_index) == 0 or len(patch_vert_index) == 0:
          continue
      except:
        continue
      # extract features from backbone_net
      stensor = sparse_tensor(ME, feats, coordinates=coords, device=device)
      features = backbone_net(stensor).F

      # mini_features: features of each patch, of size num_patches x points_per_patch x 32, e.g., 20 x 32.
      # mini_coords: coords of each patch, of size num_patches x points_per_patch x 3, e.g., 20 x 3.
      # mini_labels: labels of each patch, of size num_patches x points_per_patch x 1, e.g., 20 x 1.
      # mini_verts: vertex index of each positive patch, of size num_positive_patches x 1, note that num_positive_patches+num_negative_patches=num_patches
      # mini_verts_gt: vertex gt coord of each patch, of size num_patches x 3, note that num_positive_patches+num_negative_patches=num_patches
      mini_features = []
      mini_coords = []
      mini_labels = []
      mini_verts_gt = []
      for i, index in enumerate(patch_vert_index):
        mini_features.append(features[index.long()])
        mini_coords.append(pc_down[index.long()])
        
        mini_labels.append(torch.ones((1,)).long())
        mini_verts_gt.append(patch_vert_gt[i])

      for i, index in enumerate(patch_other_index):
        mini_features.append(features[index.long()])
        curr_coords = pc_down[index.long()]
        mini_coords.append(curr_coords)
        
        mini_labels.append(torch.zeros((1,)).long())
        mini_verts_gt.append(torch.zeros((3,)))

      
      # static_positive_line_*: static and positive lines
      # static_negative_line_*: static and negative lines
      line_features = []
      line_labels = []
      static_positive_line_coords = [] # coords of two vertices of a line
      static_positive_line_patches = [] # which patches does a line belong to ?
      static_positive_num = 0
      static_negative_line_coords = [] # coords of two vertices of a line
      static_negative_line_patches = [] # which patches does a line belong to ?
      static_negative_num = 0
      for i_line, edge in enumerate(mini_line):
        tmp_line_feature = []
        for l in edge[:-1]:
          tmp_line_feature.append(features[l])
        line_features.append(torch.stack(tmp_line_feature))
        if edge[-1] == 1:
          line_labels.append(torch.ones((1,)).long())
          tmp_edge_0_patches = [i_patch for i_patch in range(len(patch_vert_index)) if edge[0] in patch_vert_index[i_patch]]
          tmp_edge_1_patches = [i_patch for i_patch in range(len(patch_vert_index)) if edge[-2] in patch_vert_index[i_patch]]
          random.shuffle(tmp_edge_0_patches)
          random.shuffle(tmp_edge_1_patches)
          for tmp_0_patch in tmp_edge_0_patches:
            for tmp_1_patch in tmp_edge_1_patches:
                static_positive_num += 1
                static_positive_line_patches.append([tmp_0_patch, tmp_1_patch])
                static_positive_line_coords.append([pc_down[edge[0].long()], pc_down[edge[-2].long()]])
        else:
          line_labels.append(torch.zeros((1,)).long())
          tmp_edge_0_patches = [i_patch for i_patch in range(len(patch_vert_index)) if edge[0] in patch_vert_index[i_patch]]
          tmp_edge_1_patches = [i_patch for i_patch in range(len(patch_vert_index)) if edge[-2] in patch_vert_index[i_patch]]
          random.shuffle(tmp_edge_0_patches)
          random.shuffle(tmp_edge_1_patches)
          for tmp_0_patch in tmp_edge_0_patches:
            for tmp_1_patch in tmp_edge_1_patches:
                static_negative_num += 1
                static_negative_line_patches.append([tmp_0_patch, tmp_1_patch])
                static_negative_line_coords.append([pc_down[edge[0].long()], pc_down[edge[-2].long()]])
        

      '''train patches from one point cloud'''
      mini_loss = 0.0
      mini_loss_patch = 0.0
      mini_loss_vertex = 0.0
      mini_loss_line = 0.0
      mini_acc_patch = 0
      mini_TP_patch = 0
      mini_TN_patch = 0
      mini_FP_patch = 0
      mini_FN_patch = 0
      mini_acc_line = 0
      mini_TP_line = 0
      mini_TN_line = 0
      mini_FP_line = 0
      mini_FN_line = 0

      # store correctly predicted vertex
      predicted_vertex_coords = []
      predicted_vertex_probs = []
      # store index of vertex gt of predicted vertex
      true_positive_ids = []

      all_ids = list(range(0, len(mini_features)))
      random.shuffle(all_ids)
      for batch_id_start in range(0, len(all_ids), mini_batch):
        # select batch
        batch_ids = all_ids[batch_id_start : batch_id_start+mini_batch]
        # features of selected batch, of size batch_size x points_per_patch x 32
        batch_features = torch.cat([mini_features[i].unsqueeze(0) for i in batch_ids], 0).to(device)
        # coords of selected batch, of size batch_size x points_per_patch x 3
        batch_coords = torch.cat([mini_coords[i].unsqueeze(0) for i in batch_ids], 0).to(device)

        # vert_gt_delta coords of selected batch, of size batch_size x 3
        batch_vert_gt = torch.cat([mini_verts_gt[i].unsqueeze(0) for i in batch_ids], 0).to(device)

        '''for patch_net'''
        # input of patch_net, of size batch_size x 35 x points_per_patch, e.g., 2048x35x20
        batch_input_patch = torch.cat([batch_coords, batch_features], 2).transpose(1, 2)
        # labels of selected batch, of size batch_size x 1
        batch_label_patch = torch.cat([mini_labels[i].unsqueeze(0) for i in batch_ids], 0).float().squeeze().to(device)

        batch_output_patch = patch_net(batch_input_patch)

        mini_loss_patch += criterion_patch(batch_output_patch.squeeze(), batch_label_patch)

        # acc, TP, TN, FP, FN
        batch_label_patch = batch_label_patch.long()
        predicted = (torch.sigmoid(batch_output_patch.squeeze())>=0.5).long()
        mini_acc_patch += int((predicted == batch_label_patch).sum().item())

        predicted, batch_label = predicted.cpu().numpy(), batch_label_patch.cpu().numpy()
        mini_TP_patch += int((predicted & batch_label).sum())
        mini_TN_patch += int(((~predicted+2) & (~batch_label+2)).sum())
        mini_FP_patch += int(((predicted) & (~batch_label+2)).sum())
        mini_FN_patch += int(((~predicted+2) & (batch_label)).sum())

        '''for vertex_net'''
        # index of true_positive patches
        batch_output_label_patch = (torch.sigmoid(batch_output_patch.squeeze())>=0.5).long()
        batch_output_prob_patch = torch.sigmoid(batch_output_patch.squeeze())
        true_positive_patches = (batch_output_label_patch & batch_label_patch).data.cpu().numpy()==1
        # input of vertex_net, of size #true_positive_patches x 35 x points_per_patch, e.g., 40x35x20
        batch_input_vertex = torch.cat([batch_coords[true_positive_patches], batch_features[true_positive_patches]], 2).transpose(1, 2)
        # labels
        batch_label_vertex = batch_vert_gt[true_positive_patches]
        batch_prob_vertex = batch_output_prob_patch[true_positive_patches]
        # Keep training behavior aligned with validation/inference: one positive
        # patch is enough to supervise vertex regression.
        if len(batch_input_vertex) == 0:
          continue
        
        batch_output_vertex = vertex_net(batch_input_vertex)
        mini_loss_vertex += criterion_vertex(batch_output_vertex, batch_label_vertex)

        # results of vertexNet, used in lineNet
        batch_output_vertex_coord = batch_output_vertex
        predicted_vertex_coords.extend(batch_output_vertex_coord)
        predicted_vertex_probs.extend(batch_prob_vertex)
        true_positive_ids.extend(np.array(batch_ids)[true_positive_patches])

      '''for line_net'''
      # NMS to filter vertices that close
      nms_threshhold = nms_th
      dropped_vertex_index = []
      predicted_vertex_coords = torch.stack(predicted_vertex_coords) if len(predicted_vertex_coords) != 0 else torch.Tensor([])
      for i in range(len(predicted_vertex_coords)):
          if i in dropped_vertex_index:
              continue
          dist_all = torch.norm(predicted_vertex_coords-predicted_vertex_coords[i], dim=1)
          same_region_indexes = (dist_all < nms_threshhold).nonzero()
          for same_region_i in same_region_indexes[0]:
              if same_region_i == i:
                  continue
              if predicted_vertex_probs[same_region_i] <= predicted_vertex_probs[i]:
                  dropped_vertex_index.append(same_region_i)
              else:
                  dropped_vertex_index.append(i)
      selected_vertex_index = [i for i in range(len(predicted_vertex_coords)) if i not in dropped_vertex_index]
      predicted_vertex_coords = predicted_vertex_coords[selected_vertex_index]
      true_positive_ids = np.array(true_positive_ids)[selected_vertex_index].tolist()

      # dynamic line samples
      dynamic_positive_num = 0
      dynamic_negative_num = 0
      predicted_vertex_features = []
      for coord in predicted_vertex_coords:
        pred_vertex_index = torch.argmin(torch.norm(pc_down - coord, dim=1))
        predicted_vertex_features.append(features[pred_vertex_index])
      # add dynamic samples, th_p for positive threshhold, th_n for negative threshhold
      point_num_in_line = 30
      th_p = line_positive_th
      th_n = line_negative_th
      for i, positive_patches in enumerate(static_positive_line_patches):
        if (positive_patches[0] in true_positive_ids) and (positive_patches[1] in true_positive_ids):
          dynamic_positive_line_feature = []
          predicted_e1, predicted_e2 = predicted_vertex_coords[true_positive_ids.index(positive_patches[0])], predicted_vertex_coords[true_positive_ids.index(positive_patches[1])]
          gt_e1, gt_e2 = static_positive_line_coords[i]
          d1, d2 = torch.norm(predicted_e1-gt_e1), torch.norm(predicted_e2-gt_e2)
          if d1 <= th_p and d2 <= th_p:
              # dynamic positive sample
              line_labels.append(torch.ones((1,)).long())
              dynamic_positive_num += 1
              e1_coord, e2_coord = predicted_e1, predicted_e2
              e1_feature, e2_feature = predicted_vertex_features[true_positive_ids.index(positive_patches[0])], predicted_vertex_features[true_positive_ids.index(positive_patches[1])]
          elif d1 >= th_n or d2 >= th_n:
              # dynamic negative sample
              line_labels.append(torch.zeros((1,)).long())
              dynamic_negative_num += 1
              e1_coord, e2_coord = predicted_e1, predicted_e2
              e1_feature, e2_feature = predicted_vertex_features[true_positive_ids.index(positive_patches[0])], predicted_vertex_features[true_positive_ids.index(positive_patches[1])]
          else:
              continue
          dynamic_positive_line_feature.append(e1_feature)
          for inter_point in range(1, point_num_in_line+1):
              inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1_coord + (1-float(inter_point)/(point_num_in_line+1))*e2_coord)
              inter_point_index = torch.argmin(torch.norm(pc_down - inter_point_coord, dim=1))
              dynamic_positive_line_feature.append(features[inter_point_index])
          dynamic_positive_line_feature.append(e2_feature)
          line_features.append(torch.stack(dynamic_positive_line_feature))
      # add dynamic negative samples
      point_num_in_line = 30
      for i, negative_patches in enumerate(static_negative_line_patches):
        if (negative_patches[0] in true_positive_ids) and (negative_patches[1] in true_positive_ids):
          dynamic_negative_line_feature = []
          e1_coord, e2_coord = predicted_vertex_features[true_positive_ids.index(negative_patches[0])][:3], predicted_vertex_features[true_positive_ids.index(negative_patches[1])][:3]
          dynamic_negative_line_feature.append(predicted_vertex_features[true_positive_ids.index(negative_patches[0])])
          for inter_point in range(1, point_num_in_line+1):
              inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1_coord + (1-float(inter_point)/(point_num_in_line+1))*e2_coord)
              inter_point_index = torch.argmin(torch.norm(pc_down - inter_point_coord, dim=1))
              dynamic_negative_line_feature.append(features[inter_point_index])
          dynamic_negative_line_feature.append(predicted_vertex_features[true_positive_ids.index(negative_patches[1])])
          line_features.append(torch.stack(dynamic_negative_line_feature))
          line_labels.append(torch.zeros((1,)).long())
          dynamic_negative_num += 1
      
      if len(line_features) == 0:
        continue

      # train lineNet
      line_input = torch.stack(line_features).transpose(1, 2)
      line_labels = torch.stack(line_labels).to(device)

      all_line_ids = list(range(0, len(line_labels)))
      random.shuffle(all_line_ids)
      for batch_id_start in range(0, len(all_line_ids), mini_batch):
        batch_ids_line = all_line_ids[batch_id_start : batch_id_start+mini_batch]
        batch_input_line = line_input[batch_ids_line]
        batch_output_line = line_net(batch_input_line)
        batch_labels_line = line_labels[batch_ids_line].squeeze().float()

        mini_loss_line += criterion_line(batch_output_line.squeeze(), batch_labels_line)

        # acc, TP, TN, FP, FN
        batch_labels_line = batch_labels_line.long()
        predicted = (torch.sigmoid(batch_output_line.squeeze())>=0.5).long()
        mini_acc_line += int((predicted == batch_labels_line).sum().item())

        predicted, batch_labels = predicted.cpu().numpy(), batch_labels_line.cpu().numpy()
        mini_TP_line += int((predicted & batch_labels).sum())
        mini_TN_line += int(((~predicted+2) & (~batch_labels+2)).sum())
        mini_FP_line += int(((predicted) & (~batch_labels+2)).sum())
        mini_FN_line += int(((~predicted+2) & (batch_labels)).sum())

      '''for updating'''
      loss = loss_weight[0]*mini_loss_patch + loss_weight[1]*mini_loss_vertex + loss_weight[2]*mini_loss_line

      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()

      mini_acc_patch = mini_acc_patch / len(all_ids)
      mini_precision_patch = mini_TP_patch / (mini_TP_patch + mini_FP_patch + 1e-12)
      mini_recall_patch = mini_TP_patch / (mini_TP_patch + mini_FN_patch + 1e-12)

      mini_acc_line = mini_acc_line / len(line_labels)
      mini_precision_line = mini_TP_line / (mini_TP_line + mini_FP_line + 1e-12)
      mini_recall_line = mini_TP_line / (mini_TP_line + mini_FN_line + 1e-12)

      total_loss += float(loss)
      total_loss_patch += float(mini_loss_patch)
      total_acc_patch += float(mini_acc_patch)
      total_precision_patch += float(mini_precision_patch)
      total_recall_patch += float(mini_recall_patch)
      total_loss_vertex += float(mini_loss_vertex)
      total_loss_line += float(mini_loss_line)
      total_acc_line += float(mini_acc_line)
      total_precision_line += float(mini_precision_line)
      total_recall_line += float(mini_recall_line)
      total += 1
      print('Epoch %d: Obj: %d loss: %f loss_patch: %f loss_vertex: %f loss_line: %f acc_patch: %f precision_patch: %f recall_patch: %f acc_line: %f precision_line: %f recall_line: %f lr: %f' % (
        epoch, train_loader_i, total_loss/total, total_loss_patch/total, total_loss_vertex/total, total_loss_line/total,
        total_acc_patch/total, total_precision_patch/total, total_recall_patch/total,
        total_acc_line/total, total_precision_line/total, total_recall_line/total, head_current_lr))
      print('lr_backbone: %f lr_head: %f' % (backbone_current_lr, head_current_lr))
      print('static_positive_num: {}, static_negative_num: {}, dynamic_positive_num: {}, dynamic_negative_num: {}'.format(
        static_positive_num, static_negative_num, dynamic_positive_num, dynamic_negative_num
      ))


    if total == 0:
      raise RuntimeError('No valid training batches were produced. Check whether generated patches contain positive vertices and line labels.')

    ''' ---end: training---'''

    torch.save(backbone_net.state_dict(), './checkpoint_sigma{}clip{}/backbone_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    ))
    torch.save(patch_net.state_dict(), './checkpoint_sigma{}clip{}/patchnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    ))
    torch.save(vertex_net.state_dict(), './checkpoint_sigma{}clip{}/vertexnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    ))
    torch.save(line_net.state_dict(), './checkpoint_sigma{}clip{}/linenet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    ))

    log_f.write('--train-- Epoch: %d loss: %f loss_patch: %f loss_vertex: %f loss_line: %f acc_patch: %f precision_patch: %f recall_patch: %f acc_line: %f precision_line: %f recall_line: %f lr_backbone: %f lr_head: %f\t' % (
        epoch, total_loss/total, total_loss_patch/total, total_loss_vertex/total, total_loss_line/total,
        total_acc_patch/total, total_precision_patch/total, total_recall_patch/total,
        total_acc_line/total, total_precision_line/total, total_recall_line/total, backbone_current_lr, head_current_lr))
    del total_loss
    del total_loss_patch
    del total_loss_vertex
    del total_loss_line
    torch.cuda.empty_cache() 

    '''validation'''
    with torch.no_grad():
      val_loss, val_loss_patch, val_loss_vertex, val_loss_line, val_acc_patch, val_precision_patch, val_recall_patch, val_acc_line, val_precision_line, val_recall_line = evaluate(
        val_loader, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, patch_weight, loss_weight, sigma, clip)
    scheduler.step(val_loss)

    val_patch_f1 = 2 * val_precision_patch * val_recall_patch / (val_precision_patch + val_recall_patch + 1e-12)
    val_line_f1 = 2 * val_precision_line * val_recall_line / (val_precision_line + val_recall_line + 1e-12)
    # Use a wireframe-oriented score for checkpoint selection instead of raw loss alone.
    val_score = 0.35 * val_patch_f1 + 0.65 * val_line_f1

    if val_loss < best_val_loss:
      best_val_loss = val_loss

    if val_score > best_val_score + early_stop_min_delta:
      best_val_score = val_score
      epochs_without_improvement = 0
      torch.save(backbone_net.state_dict(), './checkpoint_sigma{}clip{}/backbone_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}_Val.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
      ))
      torch.save(patch_net.state_dict(), './checkpoint_sigma{}clip{}/patchnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}_Val.pth'.format(
        sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
      ))
      torch.save(vertex_net.state_dict(), './checkpoint_sigma{}clip{}/vertexnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}_Val.pth'.format(
        sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
      ))
      torch.save(line_net.state_dict(), './checkpoint_sigma{}clip{}/linenet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}_Val.pth'.format(
        sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
      ))
    else:
      epochs_without_improvement += 1


    log_f.write('--valid-- Epoch: %d loss: %f loss_patch: %f loss_vertex: %f loss_line: %f acc_patch: %f precision_patch: %f recall_patch: %f f1_patch: %f acc_line: %f precision_line: %f recall_line: %f f1_line: %f score: %f\n' % (
      epoch, val_loss, val_loss_patch, val_loss_vertex, val_loss_line, val_acc_patch, val_precision_patch, val_recall_patch, val_patch_f1, val_acc_line, val_precision_line, val_recall_line, val_line_f1, val_score
    ))
    log_f.flush()

    torch.cuda.empty_cache() 
    if epochs_without_improvement >= early_stop_patience:
      stop_msg = 'Early stopping triggered at epoch {} after {} validation plateaus. Best val loss: {:.6f}, best val score: {:.6f}\n'.format(
        epoch, epochs_without_improvement, best_val_loss, best_val_score
      )
      print(stop_msg.strip())
      log_f.write(stop_msg)
      log_f.flush()
      break
  log_f.close()



def evaluate(dataset_loader, patch_size=50, mini_batch=512, nms_th=0.05, line_positive_th=0.05, line_negative_th=0.10, patch_weight=[1.0, 2.0], loss_weight=[1.0, 1.0, 1.0], sigma=0.01, clip=0.02):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load backbone_net
    backbone_net = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
    backbone_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/backbone_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
    backbone_net = backbone_net.to(device)
    backbone_net.eval()
    # load patch_net
    patch_net = patchNet()
    patch_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/patchnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
    patch_net = patch_net.to(device)
    patch_net.eval()
    # load vertex_net
    vertex_net = vertexNet()
    vertex_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/vertexnet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
    vertex_net = vertex_net.to(device)
    vertex_net.eval()
    # load line_net
    line_net = lineNet()
    line_net.load_state_dict(torch.load('./checkpoint_sigma{}clip{}/linenet_patchSize{}_miniBatch{}_nmsTh{}_linePosTh{}_lineNegTh{}_lossweightP{}V{}L{}.pth'.format(
      sigma, clip, patch_size, mini_batch, nms_th, line_positive_th, line_negative_th, loss_weight[0], loss_weight[1], loss_weight[2]
    )))
    line_net = line_net.to(device)
    line_net.eval()

    # criterion_patch = nn.CrossEntropyLoss(weight=torch.Tensor(np.array(patch_weight))).to(device)
    criterion_patch = nn.BCEWithLogitsLoss().to(device)
    criterion_vertex = nn.MSELoss(reduction='sum').to(device)
    # criterion_line = nn.CrossEntropyLoss().to(device)
    criterion_line = nn.BCEWithLogitsLoss().to(device)


    ''' ---begin: evaluating---'''
    total_loss = 0.0
    total_acc_patch = 0.0
    total_precision_patch = 0.0
    total_recall_patch = 0.0
    total_loss_patch = 0.0
    total_loss_vertex = 0.0
    total_loss_line = 0.0
    total_acc_line = 0.0
    total_precision_line = 0.0
    total_recall_line = 0.0
    total = 0
    for val_loader_i, data in enumerate(tqdm(dataset_loader)):
      # load train data
      pc_down, feats, coords, patch_other_index, patch_vert_index, patch_vert_gt, mini_line = data
      pc_down, feats, coords, patch_other_index, patch_vert_index, patch_vert_gt, mini_line = pc_down[0], feats[0], coords[0], patch_other_index[0], patch_vert_index[0], patch_vert_gt[0], mini_line[0]
      pc_down = pc_down.to(device)
      try:
        if len(patch_other_index) == 0 or len(patch_vert_index) == 0:
          continue
      except:
        continue

      # extract features from backbone_net
      stensor = sparse_tensor(ME, feats, coordinates=coords, device=device)
      features = backbone_net(stensor).F

      # mini_features: features of each patch, of size num_patches x points_per_patch x 32, e.g., 20 x 32.
      # mini_coords: coords of each patch, of size num_patches x points_per_patch x 3, e.g., 20 x 3.
      # mini_coords_center: center of each patch, of size num_patches x points_per_patch x 3, e.g., 20 x 3.
      # mini_coords_lwh: length, width, height of each patch, of size num_patches x points_per_patch x 3, e.g., 20 x 3.
      # mini_labels: labels of each patch, of size num_patches x points_per_patch x 1, e.g., 20 x 1.
      # mini_verts: vertex index of each positive patch, of size num_positive_patches x 1, note that num_positive_patches+num_negative_patches=num_patches
      # mini_verts_gt: vertex gt coord of each patch, of size num_patches x 3, note that num_positive_patches+num_negative_patches=num_patches
      mini_features = []
      mini_coords = []
      mini_labels = []
      mini_verts_gt = []
      for i, index in enumerate(patch_vert_index):
        mini_features.append(features[index.long()])
        mini_coords.append(pc_down[index.long()])
        
        mini_labels.append(torch.ones((1,)).long())
        mini_verts_gt.append(patch_vert_gt[i])

      for i, index in enumerate(patch_other_index):
        mini_features.append(features[index.long()])
        curr_coords = pc_down[index.long()]
        mini_coords.append(curr_coords)
        
        mini_labels.append(torch.zeros((1,)).long())
        mini_verts_gt.append(torch.zeros((3,)))

      
      line_features = []
      line_labels = []
      static_positive_line_coords = [] # coords of two vertices of a line
      static_positive_line_patches = [] # which patches does a line belong to ?
      static_negative_line_coords = [] # coords of two vertices of a line
      static_negative_line_patches = [] # which patches does a line belong to ?
      for i, edge in enumerate(mini_line):
        tmp_line_feature = []
        for l in edge[:-1]:
          tmp_line_feature.append(features[l])
        line_features.append(torch.stack(tmp_line_feature))
        if edge[-1] == 1:
          line_labels.append(torch.ones((1,)).long())
          tmp_edge_0_patches = [i for i in range(len(patch_vert_index)) if edge[0] in patch_vert_index[i]]
          tmp_edge_1_patches = [i for i in range(len(patch_vert_index)) if edge[-2] in patch_vert_index[i]]
          for tmp_0_patch in tmp_edge_0_patches:
            for tmp_1_patch in tmp_edge_1_patches:
                static_positive_line_patches.append([tmp_0_patch, tmp_1_patch])
                static_positive_line_coords.append([pc_down[edge[0].long()], pc_down[edge[-2].long()]])
        else:
          line_labels.append(torch.zeros((1,)).long())
          tmp_edge_0_patches = [i for i in range(len(patch_vert_index)) if edge[0] in patch_vert_index[i]]
          tmp_edge_1_patches = [i for i in range(len(patch_vert_index)) if edge[-2] in patch_vert_index[i]]
          for tmp_0_patch in tmp_edge_0_patches[:2]:
            for tmp_1_patch in tmp_edge_1_patches[:2]:
                static_negative_line_patches.append([tmp_0_patch, tmp_1_patch])
                static_negative_line_coords.append([pc_down[edge[0].long()], pc_down[edge[-2].long()]])
        

      '''evaluate patches from one point cloud'''
      mini_loss = 0.0
      mini_loss_patch = 0.0
      mini_loss_vertex = 0.0
      mini_loss_line = 0.0
      mini_acc_patch = 0
      mini_TP_patch = 0
      mini_TN_patch = 0
      mini_FP_patch = 0
      mini_FN_patch = 0
      mini_acc_line = 0
      mini_TP_line = 0
      mini_TN_line = 0
      mini_FP_line = 0
      mini_FN_line = 0

      # store correctly predicted vertex
      predicted_vertex_coords = []
      predicted_vertex_probs = []
      # store index of vertex gt of predicted vertex
      true_positive_ids = []

      all_ids = list(range(0, len(mini_features)))
      for batch_id_start in range(0, len(all_ids), mini_batch):
        # select batch
        batch_ids = all_ids[batch_id_start : batch_id_start+mini_batch]
        # features of selected batch, of size batch_size x points_per_patch x 32
        batch_features = torch.cat([mini_features[i].unsqueeze(0) for i in batch_ids], 0).to(device)
        # coords of selected batch, of size batch_size x points_per_patch x 3
        batch_coords = torch.cat([mini_coords[i].unsqueeze(0) for i in batch_ids], 0).to(device)

        # vert_gt_delta coords of selected batch, of size batch_size x 3
        batch_vert_gt = torch.cat([mini_verts_gt[i].unsqueeze(0) for i in batch_ids], 0).to(device)

        '''for patch_net'''
        # input of patch_net, of size batch_size x 35 x points_per_patch, e.g., 2048x35x20
        batch_input_patch = torch.cat([batch_coords, batch_features], 2).transpose(1, 2)
        # labels of selected batch, of size batch_size x 1
        batch_label_patch = torch.cat([mini_labels[i].unsqueeze(0) for i in batch_ids], 0).float().squeeze().to(device)

        batch_output_patch = patch_net(batch_input_patch)

        mini_loss_patch += criterion_patch(batch_output_patch.squeeze(), batch_label_patch)

        # acc, TP, TN, FP, FN
        batch_label_patch = batch_label_patch.long()
        predicted = (torch.sigmoid(batch_output_patch.squeeze())>=0.5).long()
        mini_acc_patch += int((predicted == batch_label_patch).sum().item())

        predicted, batch_label = predicted.cpu().numpy(), batch_label_patch.cpu().numpy()
        mini_TP_patch += int((predicted & batch_label).sum())
        mini_TN_patch += int(((~predicted+2) & (~batch_label+2)).sum())
        mini_FP_patch += int(((predicted) & (~batch_label+2)).sum())
        mini_FN_patch += int(((~predicted+2) & (batch_label)).sum())

        '''for vertex_net'''
        # index of true_positive patches
        batch_output_label_patch = (torch.sigmoid(batch_output_patch.squeeze())>=0.5).long()
        batch_output_prob_patch = torch.sigmoid(batch_output_patch.squeeze())
        true_positive_patches = (batch_output_label_patch & batch_label_patch).data.cpu().numpy()==1
        # input of vertex_net, of size #true_positive_patches x 35 x points_per_patch, e.g., 40x35x20
        batch_input_vertex = torch.cat([batch_coords[true_positive_patches], batch_features[true_positive_patches]], 2).transpose(1, 2)
        # labels
        batch_label_vertex = batch_vert_gt[true_positive_patches]
        batch_prob_vertex = batch_output_prob_patch[true_positive_patches]
        if len(batch_input_vertex) == 0:
          continue
        
        batch_output_vertex = vertex_net(batch_input_vertex)
        mini_loss_vertex += criterion_vertex(batch_output_vertex, batch_label_vertex)

        # results of vertexNet, used in lineNet
        batch_output_vertex_coord = batch_output_vertex
        predicted_vertex_coords.extend(batch_output_vertex_coord)
        predicted_vertex_probs.extend(batch_prob_vertex)
        true_positive_ids.extend(np.array(batch_ids)[true_positive_patches])

      '''for line_net'''
      # NMS to filter vertices that close
      nms_threshhold = nms_th
      dropped_vertex_index = []

      if len(predicted_vertex_coords) == 0:
        continue
      predicted_vertex_coords = torch.stack(predicted_vertex_coords)
      for i in range(len(predicted_vertex_coords)):
          if i in dropped_vertex_index:
              continue
          dist_all = torch.norm(predicted_vertex_coords-predicted_vertex_coords[i], dim=1)
          same_region_indexes = (dist_all < nms_threshhold).nonzero()
          for same_region_i in same_region_indexes[0]:
              if same_region_i == i:
                  continue
              if predicted_vertex_probs[same_region_i] <= predicted_vertex_probs[i]:
                  dropped_vertex_index.append(same_region_i)
              else:
                  dropped_vertex_index.append(i)
      selected_vertex_index = [i for i in range(len(predicted_vertex_coords)) if i not in dropped_vertex_index]
      predicted_vertex_coords = predicted_vertex_coords[selected_vertex_index]
      true_positive_ids = true_positive_ids = np.array(true_positive_ids)[selected_vertex_index].tolist()

      # dynamic line samples
      dynamic_positive_num = 0
      dynamic_negative_num = 0
      predicted_vertex_features = []
      for coord in predicted_vertex_coords:
        pred_vertex_index = torch.argmin(torch.norm(pc_down - coord, dim=1))
        predicted_vertex_features.append(features[pred_vertex_index])

      # add dynamic samples, th_p for positive threshhold, th_n for negative threshhold
      point_num_in_line = 30
      th_p = line_positive_th
      th_n = line_negative_th
      for i, positive_patches in enumerate(static_positive_line_patches):
        if (positive_patches[0] in true_positive_ids) and (positive_patches[1] in true_positive_ids):
          dynamic_positive_line_feature = []
          predicted_e1, predicted_e2 = predicted_vertex_coords[true_positive_ids.index(positive_patches[0])], predicted_vertex_coords[true_positive_ids.index(positive_patches[1])]
          gt_e1, gt_e2 = static_positive_line_coords[i]
          d1, d2 = torch.norm(predicted_e1-gt_e1), torch.norm(predicted_e2-gt_e2)
          if d1 <= th_p and d2 <= th_p:
              # dynamic positive sample
              line_labels.append(torch.ones((1,)).long())
              dynamic_positive_num += 1
              e1_coord, e2_coord = predicted_e1, predicted_e2
              e1_feature, e2_feature = predicted_vertex_features[true_positive_ids.index(positive_patches[0])], predicted_vertex_features[true_positive_ids.index(positive_patches[1])]
          elif d1 >= th_n or d2 >= th_n:
              # dynamic negative sample
              line_labels.append(torch.zeros((1,)).long())
              dynamic_negative_num += 1
              e1_coord, e2_coord = predicted_e1, predicted_e2
              e1_feature, e2_feature = predicted_vertex_features[true_positive_ids.index(positive_patches[0])], predicted_vertex_features[true_positive_ids.index(positive_patches[1])]
          else:
              continue
          dynamic_positive_line_feature.append(e1_feature)
          for inter_point in range(1, point_num_in_line+1):
              inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1_coord + (1-float(inter_point)/(point_num_in_line+1))*e2_coord)
              inter_point_index = torch.argmin(torch.norm(pc_down - inter_point_coord, dim=1))
              dynamic_positive_line_feature.append(features[inter_point_index])
          dynamic_positive_line_feature.append(e2_feature)
          line_features.append(torch.stack(dynamic_positive_line_feature))
      # add dynamic negative samples
      point_num_in_line = 30
      for i, negative_patches in enumerate(static_negative_line_patches):
        if (negative_patches[0] in true_positive_ids) and (negative_patches[1] in true_positive_ids):
          dynamic_negative_line_feature = []
          e1_coord, e2_coord = predicted_vertex_features[true_positive_ids.index(negative_patches[0])][:3], predicted_vertex_features[true_positive_ids.index(negative_patches[1])][:3]
          dynamic_negative_line_feature.append(predicted_vertex_features[true_positive_ids.index(negative_patches[0])])
          for inter_point in range(1, point_num_in_line+1):
              inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1_coord + (1-float(inter_point)/(point_num_in_line+1))*e2_coord)
              inter_point_index = torch.argmin(torch.norm(pc_down - inter_point_coord, dim=1))
              dynamic_negative_line_feature.append(features[inter_point_index])
          dynamic_negative_line_feature.append(predicted_vertex_features[true_positive_ids.index(negative_patches[1])])
          line_features.append(torch.stack(dynamic_negative_line_feature))
          line_labels.append(torch.zeros((1,)).long())
          dynamic_negative_num += 1
      
      if len(line_features) == 0:
        continue

      # train lineNet
      line_input = torch.stack(line_features).transpose(1, 2)
      line_labels = torch.stack(line_labels).to(device)

      all_line_ids = list(range(0, len(line_labels)))
      random.shuffle(all_line_ids)
      for batch_id_start in range(0, len(all_line_ids), mini_batch):
        batch_ids_line = all_line_ids[batch_id_start : batch_id_start+mini_batch]
        batch_input_line = line_input[batch_ids_line]
        batch_output_line = line_net(batch_input_line)
        batch_labels_line = line_labels[batch_ids_line].squeeze().float()

        mini_loss_line += criterion_line(batch_output_line.squeeze(), batch_labels_line)

        # acc, TP, TN, FP, FN
        batch_labels_line = batch_labels_line.long()
        predicted = (torch.sigmoid(batch_output_line.squeeze())>=0.5).long()
        mini_acc_line += int((predicted == batch_labels_line).sum().item())

        predicted, batch_labels = predicted.cpu().numpy(), batch_labels_line.cpu().numpy()
        mini_TP_line += int((predicted & batch_labels).sum())
        mini_TN_line += int(((~predicted+2) & (~batch_labels+2)).sum())
        mini_FP_line += int(((predicted) & (~batch_labels+2)).sum())
        mini_FN_line += int(((~predicted+2) & (batch_labels)).sum())


      loss = loss_weight[0]*mini_loss_patch + loss_weight[1]*mini_loss_vertex + loss_weight[2]*mini_loss_line

      mini_acc_patch = mini_acc_patch / len(all_ids)
      mini_precision_patch = mini_TP_patch / (mini_TP_patch + mini_FP_patch + 1e-12)
      mini_recall_patch = mini_TP_patch / (mini_TP_patch + mini_FN_patch + 1e-12)

      mini_acc_line = mini_acc_line / len(line_labels)
      mini_precision_line = mini_TP_line / (mini_TP_line + mini_FP_line + 1e-12)
      mini_recall_line = mini_TP_line / (mini_TP_line + mini_FN_line + 1e-12)

      total_loss += float(loss)
      total_acc_patch += float(mini_acc_patch)
      total_precision_patch += float(mini_precision_patch)
      total_recall_patch += float(mini_recall_patch)
      total_loss_patch += float(mini_loss_patch)
      total_loss_vertex += float(mini_loss_vertex)
      total_loss_line += float(mini_loss_line)
      total_acc_line += float(mini_acc_line)
      total_precision_line += float(mini_precision_line)
      total_recall_line += float(mini_recall_line)
      total += 1

    
    total += 1
    return total_loss/total, total_loss_patch/total, total_loss_vertex/total, total_loss_line/total, total_acc_patch/total, total_precision_patch/total, total_recall_patch/total, total_acc_line/total, total_precision_line/total, total_recall_line/total

if __name__ == "__main__":
    train("./data", patch_size=1)
