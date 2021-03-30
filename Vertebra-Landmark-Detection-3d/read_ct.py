import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt


## save
'''
prob = np.array(prob).squeeze(1)[...,0]  # after: [64, 256, 256]
out = sitk.GetImageFromArray(prob)
sitk.WriteImage(out,out_dir+"/"+fname[i].replace(raw_type, proc_type))
'''

## input
itk_img = sitk.ReadImage('E:\\ZN-CT-nii\\data\\train\\35.nii.gz')
img = sitk.GetArrayFromImage(itk_img)
print("img shape:",img.shape)
# for i in range(img.shape[1]):
#     plt.imshow(img[:i],cmap='gray')
#     plt.show()

