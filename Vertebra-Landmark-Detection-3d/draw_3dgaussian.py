import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d
import matplotlib as mpl
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)
def gaussian3D(shape, sigma=1):
    sigma = shape[0]/6
    s, m, n = [(ss - 1.) / 2. for ss in shape]
    #生成三个array
    z, y, x = np.ogrid[-s:s+1,-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    #让h中极小的值取0
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1 #直径
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6) #

    x, y, z = int(center[0]), int(center[1]),int(center[2])

    slice, height, width = heatmap.shape[0:3]

    behind, front = min(z, radius), min(slice - z, radius + 1)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[z - behind:z + front, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - behind:radius + front, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

# hm = np.zeros((1,1, 350//4, 512//4, 512//4), dtype=np.float32)
# hm2 = np.zeros((1,1, 350//4, 512//4, 512//4), dtype=np.float32)
# radius = gaussian_radius((math.ceil(6), math.ceil(6)))
# radius = max(0, int(radius))
# #生成脊锥中心点的热图
# ct_landmark = np.asarray([30,50,50], dtype=np.float32)
# ct_landmark_int = ct_landmark.astype(np.int32)
# #print(hm)
# hm[0,0,:,:,:] = draw_umich_gaussian(hm[0,0,:,:,:], ct_landmark_int, radius=radius)

#以下代码为可视化gaussian

# xyzvalues = gaussian3D([10,10,10])
# #xyzvalues = np.random.choice(range(0,10), size=(10,10,10))#数组大小:10*10*10, 数值范围0~9.
# mycolormap = plt.get_cmap('plasma')
# xyzminvalue=xyzvalues.min()
# xyzmaxvalue=xyzvalues.max()#根据三维数组中的最大和最小值来定义每个数值的相对强度,范围0~1.0
# relativevalue=np.zeros((10,10,10))#色温强度矩阵大小与xyz测试数组大小一致
# for i in range(0,relativevalue.shape[0]):
#     for j in range(0,relativevalue.shape[1]):
#         for k in range(0,relativevalue.shape[2]):
#             relativevalue[i][j][k]=round(xyzvalues[i][j][k]/xyzmaxvalue,4)#round函数取小数点后1位
# colorsvalues = np.empty(xyzvalues.shape, dtype=object)
# alpha=0.5#透明度,视显示效果决定
# for i in range(0,relativevalue.shape[0]):
#     for j in range(0,relativevalue.shape[1]):
#         for k in range(0,relativevalue.shape[2]):
#             tempc = mycolormap(relativevalue[i][j][k])#tempc为tuple变量,存储当前数值的颜色值(R,G,B,Alpha)
#             colorreal = (tempc[0],tempc[1],tempc[2],alpha)#tuple为不可变数据类型,所以替换自定义alpha值时需要重新定义
#             colorsvalues[i][j][k] = colorreal#最终每个数值所对应的颜色
# fig = plt.figure(figsize=(14, 9))# Make a figure and axes with dimensions as desired.#需要注意的是,3Dplot不支持设置xyz的比例尺相同,这就带来了一些麻烦:#保存图片时长宽比例受限,这个问题以后再做说明解决
# ax = fig.gca(projection='3d')
# #ax.voxels(xyzvalues, facecolors=colorsvalues, edgecolor='k',shade=False,)
# ax.voxels(xyzvalues[3:7,3:7,3:7], facecolors=colorsvalues[3:7,3:7,3:7], edgecolor=None,shade=False,)#关键函数voxels:用以无缝绘制每个像素格
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Gaussian')
# #新建区域ax1,用以额外绘制colorbar#ref:https://matplotlib.org/examples/api/colorbar_only.html#位置为figure的百分比,从figure 0%的位置开始绘制, 高是figure的80%
# left, bottom, width, height = 0.1, 0.1, 0.05, 0.8#获得绘制的句柄
# ax1 = fig.add_axes([left, bottom, width, height])# Set the colormap and norm to correspond to the data for which# the colorbar will be used.
# cmap=mpl.cm.plasma #colormap与绘制voxel图保持一致
# norm = mpl.colors.Normalize(vmin=xyzminvalue, vmax=xyzmaxvalue) #色温colorbar的数值范围可选择实际xyz数组中的数值范围(其实不应该从0开始)
# cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=norm,orientation='vertical')
# cb1.set_label('Units')
# plt.show()