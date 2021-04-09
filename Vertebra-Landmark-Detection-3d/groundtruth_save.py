from dataset import BaseDataset
import joblib
import numpy as np
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dataset = BaseDataset(data_dir='E:\\ZN-CT-nii',
                                   phase='train',
                                   input_h=512,
                                   input_w=512,
                                   input_s=350,
                                   down_ratio=4)
for i in range(8):
    dict = dataset.origin_getitem(i)
    #np.save('E:\\ZN-CT-nii\\groundtruth\\'+ str(i+1)+'.gt', dict)
    joblib.dump(dict, 'E:\\ZN-CT-nii\\groundtruth\\'+ str(i+1)+'.gt')