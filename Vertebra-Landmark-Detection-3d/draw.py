import numpy as np
import cv2
import joblib
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
# img_id = '8.gt'
# data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\' + img_id)
#
# pts2 = data_dict['landmarks'] * 4
# img_series = data_dict['input'][0]
def draw_points(img_series,pts2):
    img_series_a = img_series[0][0]
    img_series_b = img_series_a.copy()
    for i, pt in enumerate(pts2):
        # color = np.random.rand(3)
        color = colors[i]
        # print(i+1, color)
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        z_axis = int(pt[0])
        y_axis = int(pt[1])
        x_axis = int(pt[2])
        tp = np.full((136,512),-1,dtype=np.float32)
        ori_image_regress_z = img_series_a[z_axis]
        ori_image_regress_x = np.transpose(img_series_b[:,:,x_axis],(0,1))
        ori_image_regress_x = np.r_[tp,ori_image_regress_x]
        ori_image_regress_x = np.r_[ori_image_regress_x,tp]
        #cv2.circle(ori_image_regress_z, (x_axis, y_axis), 4, color_255, -1, 1)
        cv2.circle(ori_image_regress_x, (y_axis, z_axis + 136), 4, color_255)#, -1, 1)
        #cv2.imshow('ori_image_regress_z', ori_image_regress_z) #确定z轴，画剖面图
        cv2.imshow('ori_image_regress_x', ori_image_regress_x)  #确定x轴，画侧视图
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()
# draw_points(img_series,pts2)