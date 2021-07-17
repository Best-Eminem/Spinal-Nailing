from dataset import BaseDataset
import joblib
import numpy as np
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def save_groundtruth(nii_nums,landmarks_num,full,mode,input_slice,input_h,input_w,down_ratio,downsize):
    dataset = BaseDataset(data_dir='E:\\ZN-CT-nii',
                                       phase='gt',
                                       input_h=input_h,
                                       input_w=input_w,
                                       input_s=input_slice,
                                       down_ratio=down_ratio,downsize=downsize)
    for i in range(nii_nums):
        dict_series = dataset.preprocess(i,points_num = landmarks_num,full = full,mode=mode)
        #np.save('E:\\ZN-CT-nii\\groundtruth\\'+ str(i+1)+'.gt', dict)
        #for j in range(5):
        j = 0
        for dict in dict_series:
            #dict = dict_series[j]
            if full == True:
                if mode == "spine_localisation":
                    joblib.dump(dict, 'E:\\ZN-CT-nii\\groundtruth\\spine_localisation\\' + str(i + 1) + '.gt')
                elif mode == "landmark_detection":
                    joblib.dump(dict, 'E:\\ZN-CT-nii\\groundtruth\\landmark_detection\\' + str(i + 1) + '.gt')
            else:
                joblib.dump(dict, 'E:\\ZN-CT-nii\\groundtruth\\'+ str(i+1)+'_l'+str(5-j)+'.gt')
                j += 1

