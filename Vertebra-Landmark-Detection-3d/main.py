import argparse

import my_eval
import train
import test
import spine_localisation_eval
from groundtruth_save import save_groundtruth


def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=2, help='down ratio')
    parser.add_argument('--down_size', type=int, default=4, help='显存不够，因而下采样')
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--input_s', type=int, default=400, help='input slice')
    parser.add_argument('--K', type=int, default=1, help='特征点的最大个数')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_150.pth', help='weights to be resumed')
    parser.add_argument('--data_dir', type=str, default='E:\\ZN-CT-nii', help='data directory')
    parser.add_argument('--phase', type=str, default='train', help='data directory')
    parser.add_argument('--mode', type=str, default='spine_localisation', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    if args.phase == 'spine_localisation_gt':
        save_groundtruth(nii_nums=49, landmarks_num=5, full=True,mode="spine_localisation",
                         input_slice=400,input_h=512,input_w=512,
                         down_ratio=args.down_ratio,downsize=args.downsize)
    elif args.phase == 'landmark_detection_gt':
        save_groundtruth(nii_nums=49, landmarks_num=30, full=True,mode="landmark_detection",
                         input_slice=240,input_h=400,input_w=400,
                         down_ratio=args.down_ratio,downsize=args.downsize)
    elif args.phase == 'train':
        is_object = train.Network(args)
        is_object.train_network(args)
    elif args.phase == 'test':
        is_object = test.Network(args)
        is_object.test(args, save=False)
    elif args.phase == 'spine_localisation_eval':
        is_object = spine_localisation_eval.spine_localisation_eval(args)
        is_object.eval(args, save=False)
        # is_object.eval_three_angles(args, save=False)
    elif args.phase == 'my_eval':
        is_object = my_eval.Network(args)
        is_object.eval(args,save = False,CT = "36.nii.gz")
