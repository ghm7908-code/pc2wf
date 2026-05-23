import os
import argparse
from train_end2end import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training PC2WF.')
    default_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    parser.add_argument('-d', '--data_path', type=str, required=False, default=default_data_path, help='base dataset path')
    parser.add_argument('-p', '--patch_size', type=int, default=50, required=False, help='patch size, e.g., 50.')
    parser.add_argument('-b', '--mini_batch', type=int, default=512, required=False, help='batch size for training patchNet, vertexNet, and lineNet.')
    parser.add_argument('-nt', '--nms_th', type=float, default=0.1, required=False, help='NMS threshold for filtering redundant vertices, used after vertexNet and before lineNet.')
    parser.add_argument('-lpt', '--line_positive_th', type=float, default=0.2, required=False, help='Threshold for positive line endpoints.')
    parser.add_argument('-lnt', '--line_negative_th', type=float, default=0.4, required=False, help='Threshold for negative line endpoints.')
    parser.add_argument('-lwP', '--loss_weight_patch', type=float, default=1.0, required=False, help='loss weight of patchNet.')
    parser.add_argument('-lwV', '--loss_weight_vertex', type=float, default=50.0, required=False, help='loss weight of vertexNet.')
    parser.add_argument('-lwL', '--loss_weight_line', type=float, default=1.0, required=False, help='loss weight of lineNet.')
    parser.add_argument('-s', '--sigma', type=float, default=0.01, required=False, help='sigma of noise.')
    parser.add_argument('-c', '--clip', type=float, default=0.01, required=False, help='clip of noise.')
    parser.add_argument('-e', '--epochs', type=int, default=20, required=False, help='maximum number of training epochs.')
    parser.add_argument('--backbone_lr', type=float, default=5e-5, required=False, help='learning rate for the pretrained backbone.')
    parser.add_argument('--head_lr', type=float, default=3e-4, required=False, help='learning rate for patch/vertex/line heads.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, required=False, help='optimizer weight decay.')
    parser.add_argument('--patch_dropout', type=float, default=0.2, required=False, help='dropout rate in patchNet.')
    parser.add_argument('--vertex_dropout', type=float, default=0.1, required=False, help='dropout rate in vertexNet.')
    parser.add_argument('--line_dropout', type=float, default=0.3, required=False, help='dropout rate in lineNet.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, required=False, help='ReduceLROnPlateau factor.')
    parser.add_argument('--lr_decay_patience', type=int, default=2, required=False, help='ReduceLROnPlateau patience in epochs.')
    parser.add_argument('--min_lr', type=float, default=1e-5, required=False, help='minimum learning rate for ReduceLROnPlateau.')
    parser.add_argument('--early_stop_patience', type=int, default=6, required=False, help='stop after this many non-improving validation epochs.')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4, required=False, help='minimum validation-loss improvement to reset early stopping.')
    args = parser.parse_args()

    loss_weight = [args.loss_weight_patch, args.loss_weight_vertex, args.loss_weight_line]
    train(
        args.data_path,
        patch_size=args.patch_size,
        mini_batch=args.mini_batch,
        nms_th=args.nms_th,
        line_positive_th=args.line_positive_th,
        line_negative_th=args.line_negative_th,
        loss_weight=loss_weight,
        sigma=args.sigma,
        clip=args.clip,
        n_epoch=args.epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        patch_dropout=args.patch_dropout,
        vertex_dropout=args.vertex_dropout,
        line_dropout=args.line_dropout,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_patience=args.lr_decay_patience,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
