import numpy as np
import cv2
import joblib
from matplotlib import pyplot as plt
import SimpleITK as sitk
import xyz2irc_irc2xyz
# img_ori = sitk.ReadImage('E:\\ZN-CT-nii\\data\\gt\\3.nii.gz')
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

def draw_points(img_series,pts2,pts_gt):
    img_series_a = img_series[0][0]
    img_series_b = img_series_a.copy()
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
        #tp = np.full((80,400),-1,dtype=np.float32)
        tp = np.full((136, 512), -1, dtype=np.float32)
        ori_image_regress_z = img_series_a[z_axis]
        ori_image_regress_x = np.transpose(img_series_b[:,:,x_axis],(0,1))
        ori_image_regress_x = np.r_[tp,ori_image_regress_x]
        ori_image_regress_x = np.r_[ori_image_regress_x,tp]

        ori_image_regress_z_gt = img_series_a[z_axis_gt]
        ori_image_regress_x_gt = np.transpose(img_series_b[:, :, x_axis_gt], (0, 1))
        ori_image_regress_x_gt = np.r_[tp, ori_image_regress_x_gt]
        ori_image_regress_x_gt = np.r_[ori_image_regress_x_gt, tp]

        cv2.circle(ori_image_regress_z, (x_axis, y_axis), 2, color_255, -1, 1)

        # cv2.circle(ori_image_regress_x, (y_axis, z_axis + 80), 2, color_255, -1, 1)
        cv2.circle(ori_image_regress_x, (y_axis, z_axis + 136), 2, color_255, -1, 1)

        cv2.circle(ori_image_regress_z_gt, (x_axis_gt, y_axis_gt), 2, color_255, -1, 1) #画gt点
        #cv2.circle(ori_image_regress_x_gt, (y_axis_gt, z_axis_gt + 80), 2, color_255, -1, 1)#画gt点
        cv2.circle(ori_image_regress_x_gt, (y_axis_gt, z_axis_gt + 136), 2, color_255, -1, 1)#画gt点

        cv2.imshow('ori_image_regress_z', ori_image_regress_z) #确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x', ori_image_regress_x)  #确定x轴，画侧视图

        cv2.imshow('ori_image_regress_z_gt', ori_image_regress_z_gt)  # 确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x_gt', ori_image_regress_x_gt)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()
def draw_points_test(img_series,pts2):
    img_series_a = img_series[0][0]
    img_series_b = img_series_a.copy()
    for pt in pts2:
        # color = np.random.rand(3)
        color = [0.1, 0.1 , 0.1]
        # print(i+1, color)
        color_255 = (255 * 1, 255 * 0, 255 * 0)
        z_axis = int(pt[0])
        y_axis = int(pt[1])
        x_axis = int(pt[2])

        tp = np.full((68,256),-1,dtype=np.float32)
        #tp = np.full((80, 400), -1, dtype=np.float32)
        ori_image_regress_z = img_series_a[z_axis]
        ori_image_regress_x = np.transpose(img_series_b[:,:,x_axis],(0,1))
        ori_image_regress_x = np.r_[tp,ori_image_regress_x]
        ori_image_regress_x = np.r_[ori_image_regress_x,tp]


        cv2.circle(ori_image_regress_z, (x_axis, y_axis), 2, color_255, -1, 1)

        cv2.circle(ori_image_regress_x, (y_axis, z_axis + 68), 2, color_255, -1, 1)


        cv2.imshow('ori_image_regress_z', ori_image_regress_z) #确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x', ori_image_regress_x)  #确定x轴，画侧视图
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()

#用来测试作为label的landmark位置是否准确
# for i in range(1,18):
#     print(i)
#     img_id = str(i)+'.gt'
#     data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\spine_localisation\\' + img_id)
#     pts2 = data_dict['landmarks'] * 2
#     a,b,c,d = data_dict['input'].shape
#     img_series = data_dict['input'].reshape((1,a,b,c,d))
#     draw_points_test(img_series,pts2)
# pts7_upsample    = [[29, 124, 184], [38, 172, 156], [42, 172, 224], [86, 176, 160], [86, 176, 212], [89, 120, 184], [130, 192, 160], [130, 188, 208], [137, 136, 180], [174, 208, 156], [174, 204, 204], [218, 220, 152], [218, 216, 200], [222, 168, 172], [73, 116, 196]]
# pts7_upsample_gt = [[24, 120, 188], [32, 172, 156], [36, 172, 224], [84, 176, 160], [84, 176, 212], [88, 120, 184], [128, 192, 160], [128, 192, 208], [136, 136, 180], [172, 208, 156], [172, 208, 204], [212, 216, 152], [216, 216, 200], [220, 164, 172],[180, 152, 180]]
# pts_gt_7 = [[27, 122, 188], [35, 173, 157], [37, 174, 225], [84, 178, 162], [85, 178, 214], [89, 120, 187], [128, 194, 160], [129, 193, 209], [137, 136, 182], [174, 209, 158], [175, 208, 204], [182, 153, 181], [215, 219, 154], [218, 219, 201], [222, 167, 175]]
# pts7 = [[29, 123, 186], [38, 170, 163], [38, 170, 227], [87, 178, 162], [87, 179, 209], [85, 123, 186], [135, 196, 163], [136, 187, 201], [133, 131, 186], [175, 203, 164], [175, 203, 202], [182, 156, 178], [225, 218, 155], [224, 218, 203], [222, 163, 178]]
# pts_gt_8 = [[28, 133, 193], [37, 202, 248], [39, 198, 143], [80, 202, 151], [80, 205, 237], [80, 127, 194], [125, 218, 156], [126, 219, 233], [129, 138, 196], [170, 235, 157], [171, 237, 228], [177, 156, 189], [216, 248, 157], [217, 253, 223], [220, 170, 189]]
# pts8 = [[29, 131, 178], [40, 202, 242], [40, 195, 130], [79, 194, 147], [88, 211, 233], [85, 122, 194], [128, 211, 153], [128, 219, 233], [133, 139, 194], [176, 235, 156], [176, 236, 226], [181, 155, 194], [225, 250, 156], [224, 251, 218], [222, 173, 186]]
#draw_points(img_series,pts7_upsample,pts7_upsample_gt)

#pts_gt_7 = [[27, 122, 188], [35, 173, 157], [37, 174, 225], [84, 178, 162], [85, 178, 214], [89, 120, 187], [128, 194, 160], [129, 193, 209], [137, 136, 182], [174, 209, 158], [175, 208, 204], [182, 153, 181], [215, 219, 154], [218, 219, 201], [222, 167, 175]]
#pts7 =     [[31, 125, 194], [40, 171, 163], [40, 170, 226], [88, 178, 164], [88, 176, 211], [87, 125, 187], [138, 194, 163], [137, 196, 211], [136, 137, 179], [178, 210, 154], [177, 203, 203], [184, 153, 178], [225, 217, 154], [225, 218, 203], [224, 171, 177]]
# img_id = '25.gt'
# data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\spine_localisation\\' + img_id)
# pts2 = data_dict['landmarks'] * 2
# a,b,c,d = data_dict['origin_image'].shape
# for i in range(b):
#     plt.imshow(data_dict['origin_image'][0][i],cmap='gray')
#     plt.show()