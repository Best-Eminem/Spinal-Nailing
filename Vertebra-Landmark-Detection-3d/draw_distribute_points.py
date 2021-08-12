import numpy as np
import math
import joblib
from matplotlib import pyplot as plt
import xyz2irc_irc2xyz
import SimpleITK as sitk
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction,PiecewiseFunction

# path = 'E:\\ZN-CT-nii\\data\\gt\\36.nii.gz' #path为文件的路径
# image = sitk.ReadImage(path)           #利用sampleItk读入mha数据，读入办法不唯一
# image = sitk.GetArrayFromImage(image)  #获得图像的数组形式的数据，需要注意顺序
def point2area_distance(Ax, By, Cz, D,point):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param point4:
    :return:点到面的距离
    """
    #Ax, By, Cz, D = define_area(point1, point2, point3,point4)
    mod_d = Ax * point[:,0] + By * point[:,1] + Cz * point[:,2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d
def define_area(points_plane):
    """
    法向量    ：n={A,B,C}
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1,point2,point3,point4 = points_plane
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    point4 = np.asarray(point4)
    AB = np.asmatrix(point2 - point1)
    CD = np.asmatrix(point4 - point3)
    N = np.cross(AB, CD)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D
def draw_model(image,points):


    # x, y, z, value = np.random.randint(50,100,(4, 40))
    # print(x,y,z,value)

    center_points = []
    for i in range(5):
        a, b, c, d = points[i * 6 + 0], points[i * 6 + 1], points[i * 6 + 4], points[i * 6 + 5]
        coner_points = np.asarray([a, b, c, d])
        center = np.mean(coner_points, axis=0)
        center_points.append(center)
    center_points = np.asarray(center_points, dtype='int32')


    for i in range(5):
        x = points[6 * i+0:6 * i+6, 0]
        y = points[6 * i+0:6 * i+6, 1]
        z = points[6 * i+0:6 * i+6, 2]
        point = mlab.points3d(x, y, z, scale_factor=5)
        #mlab.outline()
    #画矢状面中心点
    # x = center_points[:, 0]
    # y = center_points[:, 1]
    # z = center_points[:, 2]
    # point = mlab.points3d(x, y, z, scale_factor=5)
    #spine = mlab.contour3d(image, color=(0.5, 0.5, 0.5), opacity=0.3, transparent=True)
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(image), name='3-d ultrasound ')
    ctf = ColorTransferFunction()  # 该函数决定体绘制的颜色、灰度等
    #for gray_v in range(256):
    ctf.add_rgb_point(-1, 1, 1, 1)
    vol._volume_property.set_color(ctf)  # 进行更改，体绘制的colormap及color
    vol._ctf = ctf
    vol.update_ctf = True

    otf = PiecewiseFunction()
    otf.add_point(0.4,0.005)
    vol._otf = otf
    vol._volume_property.set_scalar_opacity(otf)
    # Also, it might be useful to change the range of the ctf::
    ctf.range = [-1, 1]
    #mlab.show()
def draw_plane(image,points):
    plane_points = []
    position_smaller = []
    normal_vectors = []
    for i in range(5):
        img = image.copy()
        plane_points.append(points[6*i+2:6*i+6])
        Ax, By, Cz, D = define_area(points[6*i+2:6*i+6])
        normal_vectors.append([Ax, By, Cz])

        position = np.argwhere(img != None)
        d = point2area_distance(Ax, By, Cz, D, position)
        p_position_larger = np.argwhere(d>0.5)
        #p_position_smaller = np.argwhere(d<=0.5)
        position_larger = position[p_position_larger.reshape(p_position_larger.shape[0])]
        #position_smaller.append(position[p_position_smaller.reshape(p_position_smaller.shape[0])])
        img[position_larger[:,0],position_larger[:,1],position_larger[:,2]] = -1
        small = np.argwhere(img == 1)
        position_smaller.append(small)
        spine = mlab.contour3d(img, color=(0, 0.5, 0.5))
    position_smaller = np.asarray(position_smaller)
    #img[position_smaller[:, 0], position_smaller[:, 1], position_smaller[:, 2]] = 1
    draw_model(image, points)
    mlab.outline()
    mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    draw_nail(image, points,normal_vectors)
    draw_bounding_box(points,normal_vectors)
    mlab.show()
def draw_nail(image,points,normal_vectors):
    #计算平面夹角时，由于用法向量进行计算，需要添加符号位，来表示倾斜的方向
    sign_bit = -1
    for i in range(5):
        if i >2:
            sign_bit = 1
        nail_points = points[6*i+2:6*i+4]
        nail_point_left = nail_points[0] if nail_points[0,0]<nail_points[1,0] else nail_points[1]
        nail_point_right = nail_points[1] if nail_points[0,0]<nail_points[1,0] else nail_points[0]
        left_points = np.zeros((100,3))
        left_points[:,0] = [nail_point_left[0]]*100
        left_points[:, 1] = range(nail_point_left[1]-50, nail_point_left[1]+50)
        left_points[:, 2] = [nail_point_left[2]] * 100
        #让钉子向下偏转
        left_points = structureRotateLineByAngle(left_points.tolist(),
                                                            [nail_point_left[0]-25, nail_point_left[1],
                                                             nail_point_left[2]],
                                                            [nail_point_left[0]+25, nail_point_left[1],
                                                             nail_point_left[2]],
                                                            sign_bit * getAngleFromVectors(np.asarray([0,0,1]),np.asarray(normal_vectors[i])))
        # 让钉子向横截面中心偏转
        left_points = np.asarray(structureRotateLineByAngle(left_points,
                                                            [nail_point_left[0], nail_point_left[1],
                                                             nail_point_left[2] - 25],
                                                            [nail_point_left[0], nail_point_left[1],
                                                             nail_point_left[2] + 25], -15), dtype='int32')
        #left_points = structureRotateLineByAngle(left_points, [nail_point_left[0] - 10],[0, 0, nail_point_left[0] + 10], 0, 0, 15)

        line_left = mlab.plot3d( left_points[:,0],left_points[:,1],left_points[:,2],tube_radius=5, tube_sides=6,colormap='Spectral')

        right_points = np.zeros((100, 3))
        right_points[:, 0] = [nail_point_right[0]] * 100
        right_points[:, 1] = range(nail_point_right[1] - 50, nail_point_right[1] + 50)
        right_points[:, 2] = [nail_point_right[2]] * 100
        right_points = structureRotateLineByAngle(right_points.tolist(),
                                                             [nail_point_right[0]- 25, nail_point_right[1],
                                                              nail_point_right[2]],
                                                             [nail_point_right[0] + 25, nail_point_right[1],
                                                              nail_point_right[2]], sign_bit * getAngleFromVectors(np.asarray([0,0,1]),np.asarray(normal_vectors[i])))
        right_points = np.asarray(structureRotateLineByAngle(right_points,
                                                             [nail_point_right[0], nail_point_right[1],
                                                              nail_point_right[2] - 25],
                                                             [nail_point_right[0], nail_point_right[1],
                                                              nail_point_right[2] + 25], 15), dtype='int32')
        # left_points = structureRotateLineByAngle(left_points, [nail_point_left[0] - 10],[0, 0, nail_point_left[0] + 10], 0, 0, 15)

        line_right = mlab.plot3d(right_points[:, 0], right_points[:, 1], right_points[:, 2], tube_radius=5, tube_sides=6,
                                colormap='Spectral')

    #mlab.show()
def getDistanceBetweenTwoPoints(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

# 空间中，一个结构式围绕一条直线line旋转angle角之后。求旋转之后的结构式上所有点的坐标
# 已知直线上两个点为p1，p2。两个点确定一条直线。无所谓结构式上的点是否在此直线上。
# 参考：https://www.cnblogs.com/leejxyz/p/5250935.html
# 参考网页中错误较多，需要改正
# 我不知道具体的原理是什么，反正结果是正确的。

def structureRotateLineByAngle(listPoint, p1, p2, angle):
    distance = getDistanceBetweenTwoPoints(p1, p2)
    # 方向向量(u,v,w)需为单位向量！！！请仔细查看什么是单位向量，怎么计算单位向量中每个方向的值
    # 网页中的单位向量用的是sqrt(3) /3。这是不对的
    # 注意，如果p1和p2这两个点重叠，则无法计算单位向量
    u = (p1[0] - p2[0]) / distance
    v = (p1[1] - p2[1]) / distance
    w = (p1[2] - p2[2]) / distance

    SinA = math.sin(angle * math.pi / 180)
    CosA = math.cos(angle * math.pi / 180)

    uu = u * u
    vv = v * v
    ww = w * w
    uv = u * v
    uw = u * w
    vw = v * w

    t00 = uu + (vv + ww) * CosA
    t10 = uv * (1 - CosA) + w * SinA
    t20 = uw * (1 - CosA) - v * SinA

    t01 = uv * (1 - CosA) - w * SinA
    t11 = vv + (uu + ww) * CosA
    t21 = vw * (1 - CosA) + u * SinA

    t02 = uw * (1 - CosA) + v * SinA
    t12 = vw * (1 - CosA) - u * SinA  # 网页中的t12被写成t21
    t22 = ww + (uu + vv) * CosA

    a0 = p2[0]
    b0 = p2[1]
    c0 = p2[2]

    t03 = (a0 * (vv + ww) - u * (b0 * v + c0 * w)) * (1 - CosA) + (b0 * w - c0 * v) * SinA
    t13 = (b0 * (uu + ww) - v * (a0 * u + c0 * w)) * (1 - CosA) + (c0 * u - a0 * w) * SinA
    t23 = (c0 * (uu + vv) - w * (a0 * u + b0 * v)) * (1 - CosA) + (a0 * v - b0 * u) * SinA

    # 新的坐标的list
    newlistPoint = []

    for p in listPoint:
        newPoint = [t00 * p[0] + t01 * p[1] + t02 * p[2] + t03,
                         t10 * p[0] + t11 * p[1] + t12 * p[2] + t13,
                         t20 * p[0] + t21 * p[1] + t22 * p[2] + t23]

        newlistPoint.append(newPoint)

    # 返回旋转之后的坐标list
    return newlistPoint
def getAngleFromVectors(vector_a,vector_b):
    # 引入numpy模块并创建两个向量
    # 分别计算两个向量的模：
    l_x = np.sqrt(vector_a.dot(vector_a))
    l_y = np.sqrt(vector_b.dot(vector_b))
    #print('向量的模=', l_x, l_y)

    # 计算两个向量的点积
    dot = vector_a.dot(vector_b)
    #print('向量的点积=', dian)

    # 计算夹角的cos值：
    cos_ = dot / (l_x * l_y)
    #print('夹角的cos值=', cos_)

    # 求得夹角（弧度制）：
    radian = np.arccos(cos_)
    #print('夹角（弧度制）=', radian)

    # 转换为角度值：
    angle = radian * 180 / np.pi
    return angle
def plot3Dboxes(corners):
    for i in range(corners.shape[0]):
        corner = corners[i]
        plot3Dbox(corner)

def plot3Dbox(corner):
    idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
    x = corner[0, idx]
    y = corner[1, idx]
    z = corner[2, idx]
    mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)
    mlab.show()
def get_points_from_point(point,distence):
    a = point.copy()
    a[0] = a[0] - distence
    b = point.copy()
    b[0] = b[0] + distence
    return a,b
def draw_bounding_box(points,normal_vectors):
    sign_bit = -1
    bounding_points_pair = []
    for i in range(5):
        if i >2:
            sign_bit = 1
        #point_a, point_b = points[6 * i + 0:6 * i + 2]
        point_a, point_b = points[6 * i + 4:6 * i + 6]
        if point_a[1] > point_b[1]:
            temp = point_b.copy()
            point_b = point_a
            point_a = temp
        point_a[1] -= 5;
        point_a[2] += (5 if point_a[2] >= point_b[2] else 10)
        point_b = point_a.copy();point_b[1] += 120;
        point_c = point_a.copy();point_c[2] -= 40
        point_d = point_b.copy();point_d[2] -= 40

        # if point_c[1] < point_d[1]:
        #     point_d = get_point_from_line(point_c, point_d)
        # else:
        #     point_c = get_point_from_line(point_c, point_d)

        point_a_left, point_a_right = get_points_from_point(point_a,50)
        point_b_left, point_b_right = get_points_from_point(point_b, 50)
        point_c_left, point_c_right = get_points_from_point(point_c, 50)
        point_d_left, point_d_right = get_points_from_point(point_d, 50)

        box_center_point = np.mean((point_a,point_b,point_c,point_d),axis=0)
        box_center_point_left,box_center_point_right = get_points_from_point(box_center_point, 50)
        #让八个顶点旋转
        vertex_points = [point_a_left, point_a_right,
                        point_b_left, point_b_right,
                        point_c_left, point_c_right,
                        point_d_left, point_d_right]

        point_a_left, point_a_right,point_b_left, point_b_right,point_c_left, point_c_right,point_d_left, point_d_right = structureRotateLineByAngle(vertex_points,
                                                 box_center_point_left,
                                                 box_center_point_right,
                                                 sign_bit * getAngleFromVectors(np.asarray([0, 0, 1]),
                                                                                np.asarray(normal_vectors[i])))

        #12条棱
        bounding_points_pair.append([point_a_left,point_a_right])
        bounding_points_pair.append([point_b_left, point_b_right])
        bounding_points_pair.append([point_c_left, point_c_right])
        bounding_points_pair.append([point_d_left, point_d_right])

        bounding_points_pair.append([point_a_left, point_b_left])
        bounding_points_pair.append([point_a_right, point_b_right])
        bounding_points_pair.append([point_c_left, point_d_left])
        bounding_points_pair.append([point_c_right, point_d_right])

        bounding_points_pair.append([point_a_left, point_c_left])
        bounding_points_pair.append([point_a_right, point_c_right])
        bounding_points_pair.append([point_b_left, point_d_left])
        bounding_points_pair.append([point_b_right, point_d_right])

    bounding_points_pair = np.asarray(bounding_points_pair,dtype='int32')
    for i in range(bounding_points_pair.shape[0]):
        points_pair  = bounding_points_pair[i]
        mlab.plot3d(points_pair[:, 0], points_pair[:, 1], points_pair[:, 2], color=(1, 1, 0), tube_radius=0.5,colormap='Spectral')
    #mlab.show()

def get_point_from_line(point_a,point_b):
    #给定一个y坐标，求空间直线上一个点的坐标
    vector = point_a - point_b
    a,b,c = vector
    if point_a[1] < point_b[1]:
        k = 100 / b
        x = k * a + point_a[0]
        z = k * c + point_a[2]
        y = point_a[1] + 100
    else:
        k = 100 / b
        x = k * a + point_b[0]
        z = k * c + point_b[2]
        y = point_b[1] + 100
    return [x,y,z]

if __name__ == '__main__':

    image = joblib.load('E:\\ZN-CT-nii\\my_eval\\36.nii.gz')
    image = np.transpose(image, (2, 1, 0))
    image[image < 0.05] = -1
    image[image >= 0.1] = 1
    points = [[8, 156, 204], [24, 212, 204], [36, 204, 160], [36, 204, 244], [40, 144, 204], [48, 204, 200],
              [60, 140, 204], [64, 204, 200], [80, 208, 168], [80, 212, 232], [92, 140, 204], [96, 204, 200], [108, 144, 200],
              [108, 208, 200], [124, 220, 172], [124, 220, 228], [140, 212, 200], [144, 152, 200], [148, 216, 200],
              [156, 160, 200], [168, 236, 228], [168, 240, 172], [184, 232, 200], [188, 172, 200], [192, 236, 200], [200, 180, 200],
              [212, 252, 228], [212, 256, 172], [224, 244, 200], [228, 188, 200]]
    points = np.asarray(points)
    temp = points[:, 0].copy()
    points[:, 0] = points[:, 2]
    points[:, 2] = temp
    draw_plane(image,points)
    #draw_model(image,points)
    #draw_nail(image,points)

    #plot3Dboxes(corners)
    #draw_bounding_box(points)



    print(1)


