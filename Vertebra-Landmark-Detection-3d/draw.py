import numpy as np
import cv2
import joblib
from matplotlib import pyplot as plt
from PIL import Image
import SimpleITK as sitk
import xyz2irc_irc2xyz
# img_ori = sitk.ReadImage('/home/gpu/Spinal-Nailing/ZN-CT-nii/data/train/3.nii.gz')
# point = [-25.6957,-1.1312,1819.4020]
# point_zyx = xyz2irc_irc2xyz.xyz2irc(img_ori,point)
colors = [[0.76590096, 0.0266074, 0.9806378],
           [0.54197179, 0.81682527, 0.95081629],
           [0.0799733, 0.79737015, 0.15173816],
           [0.93240442, 0.8993321, 0.09901344],
           [0.73130136, 0.05366301, 0.98405681],
           [0.01664966, 0.16387004, 0.94158259],
           [0.54197179, 0.81682527, 0.45081629],
           # [0.92074915, 0.09919099 ,0.97590748],
           [0.83445145, 0.97921679, 0.12250426],
           [0.7300924, 0.23253621, 0.29764521],
           [0.3856775, 0.94859286, 0.9910683],  # 10
           [0.45762137, 0.03766411, 0.98755338],
           [0.99496697, 0.09113071, 0.83322314],
           [0.96478873, 0.0233309, 0.13149931],
           [0.33240442, 0.9993321 , 0.59901344],
            # [0.77690519,0.81783954,0.56220024],
           # [0.93240442, 0.8993321, 0.09901344],
           [0.95815068, 0.88436046, 0.55782268],
           [0.03728425, 0.0618827, 0.88641827],
           [0.05281129, 0.89572238, 0.08913828],

           ]

def draw_points(img_series,pts2,pts_gt,mode):
    img_series_x = img_series[0][0]
    img_series_a = img_series_x.copy()
    img_series_a_gt = img_series_x.copy()
    img_series_b = img_series_x.copy()
    img_series_b_gt = img_series_x.copy()
    i = 0
    for pt, pt_gt in zip(pts2,pts_gt):
        print(pt,pt_gt)
        # color = np.random.rand(3)
        color = colors[0]
        i+=1
        # print(i+1, color)
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        z_axis = int(pt[0])
        y_axis = int(pt[1])
        x_axis = int(pt[2])

        z_axis_gt = int(pt_gt[0])
        y_axis_gt = int(pt_gt[1])
        x_axis_gt = int(pt_gt[2])
        if mode == 'spine_localisation':
            tp = np.full((56, 512), -1, dtype=np.float32)
        else:
            tp = np.full((80, 400), -1, dtype=np.float32)
        ori_image_regress_z = img_series_a[z_axis]
        ori_image_regress_x = np.transpose(img_series_b[:,:,x_axis],(0,1))
        ori_image_regress_x = np.r_[tp,ori_image_regress_x]
        ori_image_regress_x = np.r_[ori_image_regress_x,tp]

        ori_image_regress_z_gt = img_series_a_gt[z_axis_gt]
        ori_image_regress_x_gt = np.transpose(img_series_b_gt[:, :, x_axis_gt], (0, 1))
        ori_image_regress_x_gt = np.r_[tp, ori_image_regress_x_gt]
        ori_image_regress_x_gt = np.r_[ori_image_regress_x_gt, tp]




        if mode == 'spine_localisation':
            cv2.circle(ori_image_regress_z, (x_axis, y_axis), 2, color_255, -1, 1)
            cv2.circle(ori_image_regress_x, (y_axis, z_axis + 56), 2, color_255, -1, 1)
            cv2.circle(ori_image_regress_z_gt, (x_axis_gt, y_axis_gt), 2, color_255, -1, 1)  # 画gt点
            cv2.circle(ori_image_regress_x_gt, (y_axis_gt, z_axis_gt + 56), 2, color_255, -1, 1)  # 画gt点
        else:
            cv2.circle(ori_image_regress_z, (x_axis, y_axis), 2, color_255, -1, 1)
            cv2.circle(ori_image_regress_x, (y_axis, z_axis + 80), 2, color_255, -1, 1)
            cv2.circle(ori_image_regress_z_gt, (x_axis_gt, y_axis_gt), 2, color_255, -1, 1)  # 画gt点
            cv2.circle(ori_image_regress_x_gt, (y_axis_gt, z_axis_gt + 80), 2, color_255, -1, 1)  # 画gt点




        cv2.imshow('ori_image_regress_z', ori_image_regress_z) #确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x', ori_image_regress_x)  #确定x轴，画侧视图

        cv2.imshow('ori_image_regress_z_gt', ori_image_regress_z_gt)  # 确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x_gt', ori_image_regress_x_gt)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()
def draw_points_test(img_series,pts2):
    s,h,w = img_series.shape
    img_series_x = img_series
    img_series_a = img_series_x.copy()
    img_series_a_gt = img_series_x.copy()
    img_series_b = img_series_x.copy()
    img_series_b_gt = img_series_x.copy()
    for pt in pts2:
        # color = np.random.rand(3)
        color = [0.1, 0.1 , 0.1]
        # print(i+1, color)
        color_255 = (255 * 1, 255 * 0, 255 * 0)
        z_axis = int(pt[0])
        y_axis = int(pt[1])
        x_axis = int(pt[2])

        #tp = np.full((68,256),-1,dtype=np.float32)
        tp = np.full((int((h-s)//2), h), -1, dtype=np.float32)
        #tp = np.full((56, 512), -1, dtype=np.float32)
        ori_image_regress_z = img_series_a[z_axis]
        ori_image_regress_x = np.transpose(img_series_b[:, :, x_axis], (0, 1))
        ori_image_regress_x = np.r_[tp, ori_image_regress_x]
        ori_image_regress_x = np.r_[ori_image_regress_x, tp]


        cv2.circle(ori_image_regress_z, (x_axis, y_axis), 2, color_255, -1, 1)
        cv2.circle(ori_image_regress_x, (y_axis, z_axis + int((h-s)/2)), 2, color_255, -1, 1)


        cv2.imshow('ori_image_regress_z', ori_image_regress_z) #确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x', ori_image_regress_x)  #确定x轴，画侧视图
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()

def draw_by_matplotlib():
    im = np.array(Image.open('/home/gpu/图片/1.jpeg'))
    # 绘制图像
    plt.imshow(im)
    # 一些点
    x = [20,40,60]
    y = [40,80,120]
    # 使用红色星状标记绘制点
    plt.plot(x, y, 'r*')
    # 绘制连接前两个点的线
    # plot(x[:2],y[:2])
    # 添加标题，显示绘制的图像
    plt.title('Plotting: "empire.jpg"')
    plt.show()
#
# draw_by_matplotlib()