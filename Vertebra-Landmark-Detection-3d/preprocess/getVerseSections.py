from glob import glob
import json
from itk.support.extras import output
import numpy as np
import sys
import os
from tqdm import tqdm
o_path = os.getcwd()
sys.path.append(o_path)
from draw.draw import draw_points_test
#import draw_distribute_points
import SimpleITK as sitk
from matplotlib import pyplot as plt

def clean_verse(CT_path,label_path,output_path):
    #print(CT_path,label_path)
    L1_coords = [] #(x,y,z)
    L2_coords = [] #(x,y,z)
    L3_coords = [] # (x,y,z)
    L4_coords = [] # (x,y,z)
    L5_coords = [] #(x,y,z)
    with open(label_path, 'r') as f:
        # load json file
        json_data = json.load(f)
        for dict in json_data:
            if 'direction' in dict:
                continue
            if 'label' in dict:
                if dict['label'] == 20:
                    L1_coords.append(dict['X'])
                    L1_coords.append(dict['Y'])
                    L1_coords.append(dict['Z'])
                elif dict['label'] == 21:
                    L2_coords.append(dict['X'])
                    L2_coords.append(dict['Y'])
                    L2_coords.append(dict['Z'])
                elif dict['label'] == 22:
                    L3_coords.append(dict['X'])
                    L3_coords.append(dict['Y'])
                    L3_coords.append(dict['Z'])
                elif dict['label'] == 23:
                    L4_coords.append(dict['X'])
                    L4_coords.append(dict['Y'])
                    L4_coords.append(dict['Z'])
                elif dict['label'] == 24:
                    L5_coords.append(dict['X'])
                    L5_coords.append(dict['Y'])
                    L5_coords.append(dict['Z'])
                    break
                else:continue
    if len(L5_coords) == 0 or len(L4_coords) == 0 or len(L3_coords) == 0 or len(L2_coords) == 0 or len(L1_coords) == 0:
        print(CT_path,"doesn't have a lumbar section!!" )
        return None
    L1_coords = np.asarray(L1_coords,dtype=np.int32)
    L2_coords = np.asarray(L2_coords,dtype=np.int32)
    L3_coords = np.asarray(L3_coords, dtype=np.int32)
    L4_coords = np.asarray(L4_coords, dtype=np.int32)
    L5_coords = np.asarray(L5_coords,dtype=np.int32)
    coords_average = (L1_coords+L2_coords+L3_coords+L4_coords+L5_coords)//5
    spine_distance = (L1_coords[2] - L5_coords[2]) // 4
    if spine_distance<10:#错误标注数据
        return None
    itk_img = sitk.ReadImage(CT_path)
    image_array = sitk.GetArrayFromImage(itk_img)
    Z,Y,X = image_array.shape
    if (Y!=512 or X!=512):
        return None
    #print(image_array.shape)
    # 截取腰椎部分并裁剪X，Y为200*200
    # image_array_section = image_array[L5_coords[2]-spine_distance if L5_coords[2]-spine_distance>0 else 0:L1_coords[2]+spine_distance if L1_coords[2]+spine_distance<Z else Z,
    #                       #Y-2*(Y-L1_coords[1]) if Y-2*(Y-L1_coords[1])>0 else 0:Y,
    #                       coords_average[1]-100:coords_average[1]+100,
    #                       # L1_coords[0]-200 if L1_coords[0]-200 >0 else 0 :L1_coords[0]+200 if L1_coords[0]+200 <0 else X
    #                       coords_average[0]-100:coords_average[0]+100
    #                       ]

    # 截取腰椎部分但是不裁剪X，Y，这是为了和医生数据保持同样的分辨率，方便加入verse数据到特征点识别数据集中
    image_array_section = image_array[L5_coords[2] - spine_distance if L5_coords[2] - spine_distance > 0 else 0:L1_coords[2] + spine_distance if L1_coords[2] + spine_distance < Z else Z,
                          # Y-2*(Y-L1_coords[1]) if Y-2*(Y-L1_coords[1])>0 else 0:Y,
                          :,
                          # L1_coords[0]-200 if L1_coords[0]-200 >0 else 0 :L1_coords[0]+200 if L1_coords[0]+200 <0 else X
                          :]
    # for i in range(image_array_section.shape[0]):
    #     image_array_section[i] = np.flip(image_array_section[i,:,:], axis=0)
    # image_array_section = image_array_section[:,:,190:350]
    # image_array_section = image_array_section[:,300:420,:]
    #print(image_array_section.shape)
    # plt.imshow(image_array_section[398,:,:], cmap = 'gray')
    # ax = plt.gca()
    # ax.scatter(269,349,s=10,c='red')
    # plt.show()

    # plt.imshow(image_array_section[:, :, image_array_section.shape[2]//2], cmap='gray')
    # plt.show()

    CTbasename = os.path.basename(CT_path)
    output_img = sitk.GetImageFromArray(image_array_section)
    output_img.SetSpacing(itk_img.GetSpacing())
    output_img.SetOrigin(itk_img.GetOrigin())
    # sitk.WriteImage(output_img,os.path.join(output_path, CTbasename)) #不设置元数据，默认origin为（1，1，1）,direction为[1, 0, 0], [0, 1, 0], [0, 0, 1],spacing为1，1，1
    newSimpleName = CTbasename[9:12]+("_msk" if CTbasename.find('msk')!=-1 else "") +".nii.gz"
    sitk.WriteImage(output_img, os.path.join(output_path,newSimpleName))  # 不设置元数据，默认origin为（1，1，1）,direction为[1, 0, 0], [0, 1, 0], [0, 0, 1],spacing为1，1，1
    # image_array_section = image_array_section/2048
    #draw_points_test(image_array_section.reshape(400,512,512), [[359,255,255]]) #draw 2D image to show the position of the landmarks
    # #pts_predict = pts_predict.tolist()

    # # 截取包含5段脊椎的部分
    # image_array_section = image_array[bottom_z:top_z,
    #                         (pts_center_y - 200) if pts_center_y - 200 >= 0 else 0:(
    #                                     pts_center_y + 200) if pts_center_y + 200 <= 512 else 512,
    #                         (pts_center_x - 200) if pts_center_x - 200 >= 0 else 0:(
    #                                     pts_center_x + 200) if pts_center_x + 200 <= 512 else 512]
    # print(image_array_section.shape)
    # output_image,_ = transform.Resize.resize(img = image_array_section.copy(),pts = np.asarray([[1,1,1]]),input_s=240, input_h=400, input_w=400)
def process_img_section(input_path,label_path,output_path):
    filenames = sorted(glob(os.path.join(input_path, '*.nii.gz')))
    labelnames = sorted(glob(os.path.join(label_path, '*.json')))
    if len(filenames)//2 != len(labelnames):
        print('the number of nii not match the number of labels!!')
        return False
    index = 0
    for i in range(len(filenames)):
        print(os.path.basename(filenames[i]),os.path.basename(labelnames[index]))
        clean_verse(filenames[i],labelnames[index],output_path)
        #clean_verse('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_processed/sub-verse561_dir-sag_ct.nii.gz', '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_labels/sub-verse561_dir-sag_seg-subreg_ctd.json', output_path)
        if i%2 == 1:
            index+=1

if __name__ == '__main__':
    input_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_processed_notSameHu'
    label_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_RAI_labels'
    # label_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/tp_label'
    # input_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/tp_img'
    # output_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_sections'
    # output_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_WholeView_Sections'
    output_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_OriginLumbar_Sections'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    process_img_section(input_path,label_path,output_path)

