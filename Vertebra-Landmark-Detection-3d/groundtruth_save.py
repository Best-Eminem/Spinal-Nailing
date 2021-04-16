from dataset import BaseDataset
import joblib
import numpy as np
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def save_groundtruth(nii_nums,landmarks_num,full,input_slice):
    dataset = BaseDataset(data_dir='E:\\ZN-CT-nii',
                                       phase='gt',
                                       input_h=512,
                                       input_w=512,
                                       input_s=input_slice,
                                       down_ratio=4)
    for i in range(nii_nums):
        dict_series = dataset.origin_getitem(i,points_num = landmarks_num,full = full)
        #np.save('E:\\ZN-CT-nii\\groundtruth\\'+ str(i+1)+'.gt', dict)
        #for j in range(5):
        j = 0
        for dict in dict_series:
            #dict = dict_series[j]
            if full == True:
                joblib.dump(dict, 'E:\\ZN-CT-nii\\groundtruth\\' + str(i + 1) + '.gt')
            else:
                joblib.dump(dict, 'E:\\ZN-CT-nii\\groundtruth\\'+ str(i+1)+'_l'+str(5-j)+'.gt')
                j += 1

save_groundtruth(8,15,True,240)