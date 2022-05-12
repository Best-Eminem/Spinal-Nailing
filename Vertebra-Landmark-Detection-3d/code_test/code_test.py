import numpy as np
from numpy.core.numeric import NaN
import torch
import torch.nn as nn
from torch.utils import data
import math
def define_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D


def point2area_distance(point1, point2, point3, point4):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param point4:
    :return:点到面的距离
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d
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

if __name__ == '__main__':
    # pts = [[82, 132, 249], [98, 188, 249], [110, 176, 193], [110, 176, 301], [110, 184, 217],[114, 120, 245], [114, 188, 273], [122, 180, 245],
    #        [130, 116, 249], [134, 180, 245],[154, 180, 201], [154, 184, 221], [154, 184, 285], [154, 188, 265], [162, 116, 245],[166, 180, 245],
    #        [178, 180, 245], [182, 120, 245], [194, 192, 209], [194, 192, 281],[194, 196, 221], [194, 196, 265], [210, 188, 245], [214, 128, 245],
    #        [218, 192, 245],[226, 136, 245], [238, 212, 209], [238, 212, 277], [238, 216, 221], [238, 216, 265],[250, 204, 241], [254, 148, 241],
    #        [258, 208, 241], [266, 152, 241], [278, 228, 277],[278, 232, 209], [278, 232, 221], [278, 232, 265], [290, 220, 245], [294, 164, 245]]
    # # pts1 = [[4,2,2],[3,8,3],[3,6,2],[3,6,1]]
    # pts.sort(key=lambda x:(x[0]))
    # landmarks = []
    # for i in range(5):
    #     temp = pts[8*i:8*(i+1)]
    #     temp.sort(key=lambda x:x[2])
    #     pedical_point = temp[0:2]
    #     pedical_point.extend(temp[-2:])
    #     plain_point = temp[2:6]
    #     plain_point.sort(key=lambda x:x[0])
    #     bottom_point = plain_point[0:2]
    #     top_point = plain_point[2:4]
    #     bottom_point.sort(key=lambda x:x[1])
    #     top_point.sort(key=lambda x:x[1])
    #     landmarks.extend(top_point)
    #     landmarks.extend(bottom_point)
    #     landmarks.extend(pedical_point)
    #
    
    # str1 = ['1','2','3']
    # file = open('data.txt', 'w')
    # file.write(str(str1))
    # file.close()
    # f = open(r'/home/gpu/Spinal-Nailing/weights_spinal/spine_localisation/fold_'+str(0)+'/val_set.txt','r')
    # data = list(f)[0]
    # f.close
    # data = data.replace('[','');data = data.replace(']','')
    # data = data.replace('\'','')
    # data_list = data.split(',')

    # a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    # b = torch.tensor([[3, 5], [8, 6]], dtype=torch.float)
    
    # loss_fn2 = torch.nn.MSELoss(reduction='sum')
    # loss2 = loss_fn2(a.float(), b.float())
    # print(loss2)   # 输出结果：tensor(42.)


    # heatmap_sigmas = torch.nn.Parameter(torch.FloatTensor(20 * [3]), requires_grad=True)
    # print(heatmap_sigmas.detach().numpy().tolist())

    # points = [[12, 148, 184], [24, 232, 188], [40, 136, 184], [52, 216, 192], [64, 128, 196], [64, 216, 196], [92, 128, 200], [96, 216, 200], [112, 136, 204], [108, 220, 204], [140, 148, 204], [136, 232, 204], [156, 160, 204], [148, 240, 208], [188, 176, 204], [180, 260, 208], [200, 188, 208], [192, 268, 208], [228, 208, 208], [224, 284, 204]]
    # points = np.asarray(points)
    # sigma = []
    # for i in range(5):
    #     pts = points[4*i:4*i+4]
    #     bbox_h = np.mean([np.sqrt(np.sum((pts[0,:]-pts[2,:])**2)),
    #                       np.sqrt(np.sum((pts[1,:]-pts[3,:])**2))])
    #     bbox_w = np.mean([np.sqrt(np.sum((pts[0,:]-pts[1,:])**2)),
    #                       np.sqrt(np.sum((pts[2,:]-pts[3,:])**2))])
    #     radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)),min_overlap=0.9)
    #     sigma.append(radius/3)
    # heatmap_sigmas = torch.nn.Parameter(torch.FloatTensor(20 * [12.0]))
    # a = torch.mean(torch.pow(heatmap_sigmas,2)) * 0.00000001
    # print(1)
    # aa = torch.tensor([1,2,3,-1,NaN,NaN])
    # print(torch.tensor(5)+torch.tensor(NaN))
    # aa[torch.isnan(aa)] = 10
    # print(aa)
    
    # print('fuck nan occurs --------------------------------------')
    # aa_min = torch.min(aa)
    # cc = torch.isnan(aa).int().sum()
    # print(aa,aa_min,cc)
    # print(torch.sigmoid(torch.tensor(-100)))
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))
    plt.show()
