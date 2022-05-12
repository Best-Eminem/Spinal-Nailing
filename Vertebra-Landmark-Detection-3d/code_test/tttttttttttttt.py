import numpy as np
import os
output_paths = [os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole',seg_path) for seg_path in ['L5','L4','L3','L2','L1','ALL']]
output1_paths = [os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole2',seg_path) for seg_path in ['L5','L4','L3','L2','L1','ALL']]
output_paths = ['/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole2/ALL/data/']
output1_paths = ['/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_seg_boundingBox_whole2/ALL/data/']
for x, y in zip(output_paths,output1_paths):
    file_list = set(os.listdir(os.path.join(x, 'train')))
    file1_list = set(os.listdir(os.path.join(y, 'val')))
    print(x, file_list - file1_list)
    print(y, file1_list - file_list)
    

