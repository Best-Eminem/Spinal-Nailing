import SimpleITK as sitk
import numpy as np
import cv2
from matplotlib import pyplot as plt


## save
'''
prob = np.array(prob).squeeze(1)[...,0]  # after: [64, 256, 256]
out = sitk.GetImageFromArray(prob)
sitk.WriteImage(out,out_dir+"/"+fname[i].replace(raw_type, proc_type))
'''

## input
itk_img = sitk.ReadImage('E:\\ZN-CT-nii\\data\\gt\\35.nii.gz')
img = sitk.GetArrayFromImage(itk_img)
min = np.min(img)
max = np.max(img)
# out_image = np.clip(out_image, a_min=0., a_max=255.)
# 不需要调换顺序
out_image = img / np.max(abs(img))
out_image = np.asarray(out_image, np.float32)
out_image = out_image[:,55:467,55:467]
out_image = np.reshape(out_image, (-1, 412, 412))
new_image = out_image.copy()
new_image[new_image>=0.1] = 1
new_image[new_image<0.1] = -1
slice = range(0,350,50)
for i in slice:
    cv2.imshow('out_image', out_image[i])
    cv2.imshow('new_image', new_image[i])
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        exit()

#print("img shape:",img.shape)
# for i in range(img.shape[1]):
#     plt.imshow(img[:i],cmap='gray')
#     plt.show()

