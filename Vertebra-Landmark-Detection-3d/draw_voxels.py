import numpy as np
import cv2
import joblib
from matplotlib import pyplot as plt
import SimpleITK as sitk
import xyz2irc_irc2xyz

data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\landmark_detection\\36.gt')
lab = data_dict['input'][0]
print(1)

labb=np.transpose(lab,(2,1,0))
xlim=[50,150]
ylim=[50,150]
zlim=[50,60]
xyzvalues=labb[xlim[0]:xlim[1],ylim[0]:ylim[1],zlim[0]:zlim[1]]
#
mycolormap = plt.get_cmap('Greys')
# xyzminvalue=xyzvalues.min()
# xyzmaxvalue=xyzvalues.max()
#
relativevalue=np.round(xyzvalues,1)
colorsvaluesr = np.empty(xyzvalues.shape, dtype=object)
alpha=0.5
d,w,h=xyzvalues.shape
sizeall=xyzvalues.size
zt=np.reshape(relativevalue,(sizeall,))
#
#
#
colorsvalue=np.array([(mycolormap(i)[0],mycolormap(i)[1],mycolormap(i)[2],alpha) for i in zt])
colorsvalues=np.reshape(colorsvalue,(d,w,h,4))
#
#
fig = plt.figure(figsize=(7, 4.5))
ax = fig.gca(projection='3d')
pos=ax.voxels(xyzvalues, edgecolor='k',linewidth=0.3,shade=False,)



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Voxel Map')

plt.show()

