import argparse
from glob import glob
import os
import generateSegSections
import my_eval
import testSegmentation
import train
import test
import spine_localisation_eval
import trainSegmentation
from preprocess.groundtruth_save import save_groundtruth

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=2, help='down ratio')
    parser.add_argument('--down_size', type=int, default=4, help='显存不够，因而下采样')
    parser.add_argument('--input_h', type=int, default=400, help='input height')
    parser.add_argument('--input_w', type=int, default=400, help='input width')
    parser.add_argument('--input_s', type=int, default=240, help='input slice')
    parser.add_argument('--K', type=int, default=8, help='特征点的最大个数')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_200.pth', help='weights to be resumed')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--net', type=str, default='resnet34', help='net')
    parser.add_argument('--data_dir', type=str, default='/home/gpu/Spinal-Nailing/ZN-CT-nii', help='CT data directory')
    parser.add_argument('--model_dir', type=str, default='/home/gpu/Spinal-Nailing/weights_spinal', help='model data directory')
    parser.add_argument('--mode', type=str, default='lumbar_segmentation', help='step 1 or 2 or 3')
    parser.add_argument('--learnable_sigma', type=bool, default=False, help='learnable_sigma true or false')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    parser.add_argument('--k_fold', type=int, default=5, help='K_fold')
    parser.add_argument('--output_channel', type=int, default=20, help='output_channel')
    parser.add_argument('--KZdataset', type=bool, default=False, help='KZdataset')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    # if args.phase == 'spine_localisation_gt':
    #     save_groundtruth(nii_nums=49, landmarks_num=5, full=True,mode="spine_localisation",
    #                      input_slice=400,input_h=512,input_w=512,
    #                      down_ratio=args.down_ratio,downsize=args.downsize)
    # elif args.phase == 'landmark_detection_gt':
    #     save_groundtruth(nii_nums=49, landmarks_num=30, full=True,mode="landmark_detection",
    #                      input_slice=240,input_h=400,input_w=400,
    #                      down_ratio=args.down_ratio,downsize=args.downsize)
    if args.phase == 'train':
        if args.mode == 'landmark_detection':
            args.down_size = 2
            args.down_ratio= 2
            args.input_h = 400
            args.input_w = 400
            args.input_s = 240
            args.K = 1
            args.resume = 'landmark_detection/model_last_origin.pth'
            args.ki = 0
            args.learnable_sigma = True
            args.output_channel = 40
            args.KZdataset = True
            for i in range(4,args.k_fold):
                print("================================fold"+str(i)+"==begin==================================")
                args.ki = 0
                is_object = train.Network(args)
                is_object.train_network(args)
                print("================================fold"+str(i)+"==done===================================")
        elif args.mode == 'spine_localisation':
            args.down_size = 4
            args.down_ratio= 2
            args.input_h = 512
            args.input_w = 512
            args.input_s = 400
            args.K = 1
            args.output_channel = 5
            args.resume = 'spine_localisation/fold_2/model_last_origin.pth'
            args.learnable_sigma = False
            args.KZdataset = False
            for i in range(4,args.k_fold):
                print("================================fold"+str(i)+"==begin==================================")
                args.ki = 0
                is_object = train.Network(args)
                is_object.train_network(args)
                print("================================fold"+str(i)+"==done===================================")
        elif args.mode == 'lumbar_segmentation':
            args.down_size = 1
            args.down_ratio = 1
            args.input_h = 80 #120 or 160 or 128
            args.input_w = 160 #120 or 160 or 128
            args.input_s = 64
            args.K = 1
            args.batch_size = 1
            args.output_channel = 1
            args.learnable_sigma = False
            args.net = 'resnet34'
            # for i in range(1,args.k_fold):
            args.ki = 0
            #lumbars = reversed(['L5','L4','L3','L2','L1','ALL','ALL_top4'])
            lumbars = reversed(['L5','L4','L3','L2','L1','ALL'])
            for lumbar_number in lumbars:
                args.lumbar_number = lumbar_number
                # args.data_dir = os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg/big_view_all',args.lumbar_number)
                args.data_dir = os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole2',args.lumbar_number)
                args.resume = os.path.join('lumbar_segmentation',args.lumbar_number,'model_last_origin.pth')
                is_object = trainSegmentation.Network(args)
                is_object.train_network(args)
                break
    elif args.phase == 'test':
        if args.mode == 'landmark_detection':
            args.down_size = 2
            args.down_ratio= 2
            args.input_h = 400
            args.input_w = 400
            args.input_s = 240
            args.K = 1
            args.ki = 0
            args.resume = 'landmark_detection/model_last_origin.pth'
            args.output_channel = 40
            args.KZdataset = True
            #args.resume = 'landmark_detection/model_40.pth'
            args.resume = 'landmark_detection/all_points_k_fold_0/model_last_origin.pth'
            is_object = test.Network(args)
            is_object.test(args, save=False)
        elif args.mode == 'spine_localisation':
            for i in range(4,args.k_fold):
                args.ki = 2
                args.down_size = 4
                args.input_h = 512
                args.input_w = 512
                args.input_s = 400
                args.K = 1
                args.KZdataset = False
                tp_str = 'fold_'+str(args.ki)
                args.resume = os.path.join('spine_localisation',tp_str, 'model_last_origin.pth')
                args.resume = '/home/gpu/Spinal-Nailing/weights_spinal/spine_localisation/model_last_origin.pth'
                args.output_channel = 5
                is_object = test.Network(args)
                is_object.test(args, save=False)
        elif args.mode == 'lumbar_segmentation':
            args.down_size = 1
            args.down_ratio = 1
            args.input_h = 80 #120 or 160
            args.input_w = 160 #120 or 160
            args.input_s = 64
            args.K = 1
            args.net = 'resnet34'
            args.output_channel = 1
            args.learnable_sigma = False
            # for i in range(1,args.k_fold):
            args.ki = 0
            # lumbars = reversed(['L5','L4','L3','L2','L1','ALL','ALL_top4'])
            lumbars = reversed(['L5','L4','L3','L2','L1','ALL'])
            whole_len = 0
            Hausdorff_distance = 0
            mean_dice_score = 0
            ZNSeg = False
            for lumbar_number in lumbars:
                whole_len+=1
                args.lumbar_number = lumbar_number
                # args.data_dir = os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg/big_view_all',args.lumbar_number)
                args.data_dir = os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole2',args.lumbar_number)
                # args.data_dir = os.path.join('/home/gpu/Spinal-Nailing/ZN-CT-nii/single_lumbar',args.lumbar_number)
                args.resume = os.path.join('lumbar_segmentation',args.lumbar_number,'model_last_origin.pth')
                is_object = testSegmentation.Network(args)
                tp1,tp2 = is_object.test(args, save=False,ZNSeg = ZNSeg)
                Hausdorff_distance+=tp2
                mean_dice_score+=tp1
                break
            print("In 5 lumbars,mean_dice_score is {}, Hausdorff_distance is {}".format(mean_dice_score/whole_len,Hausdorff_distance/whole_len))
        
    elif args.phase == 'spine_localisation_eval':
        args.down_size = 4
        args.down_ratio= 2
        args.input_h = 512
        args.input_w = 512
        args.input_s = 400
        args.K = 1
        args.output_channel = 5
        args.resume = 'spine_localisation/model_last_origin.pth'
        args.learnable_sigma = False
        #for i in range(1,args.k_fold):
        args.ki = 0
        is_object = spine_localisation_eval.spine_localisation_eval(args)
        is_object.eval(args, save=False)
        # is_object.eval_three_angles(args, save=False)
    elif args.phase == 'my_eval':
        generateZNSeg = True
        if generateZNSeg:
            is_object = my_eval.Network(args,generateZNSeg)
            CT_path = '/home/gpu/Spinal-Nailing/ZN-CT-nii/data/spine_localisation_eval'
            CT_files = sorted(glob(os.path.join(CT_path, '*.nii.gz')))
            label_path = ''
            for file in CT_files:
                is_object.eval(args,save = False,CT_path = file,label_path = label_path)
        else:
            is_object = my_eval.Network(args,generateZNSeg)
            CT_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_sections/sub-verse825_dir-ax_ct.nii.gz'
            label_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_labels/sub-verse514_dir-iso_seg-subreg_ctd.json'
            is_object.eval(args,save = False,CT_path = CT_path,label_path = label_path)
    elif args.phase == 'generateSeg':
        generateZNSeg = False
        is_object = generateSegSections.Network(args)
        # CT_path = '/home/gpu/Spinal-Nailing/ZN-CT-nii/spine_eval_sections'
        # CT_path = '/home/gpu/Spinal-Nailing/ZN-CT-nii/segment_whole_sections'
        # CT_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_sections'
        CT_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_WholeView_Sections'
        #CT_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_bigView_Sections'
        label_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_labels/sub-verse514_dir-iso_seg-subreg_ctd.json'
        # output_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg'
        output_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole3'
        # output_path = '/home/gpu/Spinal-Nailing/ZN-CT-nii/single_lumbar'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        is_object.eval(args,save = False,CT_path = CT_path,label_path = label_path,output_path = output_path , generateZNSeg = generateZNSeg)