import SimpleITK as sitk
import cv2
import torch
# from draw_gaussian import *
from draw_3dgaussian import *
import transform
import math
from transform import resize_image_itk
from xyz2irc_irc2xyz import xyz2irc
import torch.nn as nn

def processing_train(image, pts, points_num,image_h, image_w, image_s,
                     down_ratio, aug_label, img_id,full,spine_localisation_eval_dict):
    # filter pts ----------------------------------------------------
    # h,w,c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    pts = np.array(pts)
    pts = pts.astype('float32')
    pts_irc = []  # ['index', 'row', 'col']
    for i in range(len(pts)):
        pts_irc.append(xyz2irc(image, pts[i]))
    pts_irc = np.asarray(pts_irc) #小数形式
    image_array = sitk.GetArrayFromImage(image)
    data_series = []
    data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),  # 转为float32格式
                                            # transform.PhotometricDistort(), #对比度，噪声，亮度
                                            # transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                            # transform.RandomMirror_w(),
                                            transform.Resize(s=image_s, h=image_h, w=image_w) #Resize了点坐标和img
                                            ]),  # resize
                'val': transform.Compose([transform.ConvertImgFloat(),
                                          transform.Resize(s=image_s, h=image_h, w=image_w)
                                          ])}
    if full == False:
        for i in range(5):
            #获取每一段脊椎的上下两点的z坐标，生成截取坐标
            top_z = pts_irc[7*i,0] + 5
            bottom_z = pts_irc[7*i+2,0] - 5
            # bottom_z = pts_irc[:,0].min()
            # top_z = pts_irc[:, 0].max()
            # 所有点的z轴坐标要改变
            pts_irc[7*i:7*(i+1),0] = pts_irc[7*i:7*(i+1),0] - bottom_z
            # 截取包含5段脊椎的部分
            image_array_section = image_array[bottom_z:top_z,:,:]
            if aug_label:
                out_image, pts_2 = data_aug['train'](image_array_section.copy(), pts_irc[7*i:7*(i+1)])
            else:
                out_image, pts_2 = data_aug['val'](image_array_section.copy(), pts_irc[7*i:7*(i+1)])

            min = np.min(out_image)
            max = np.max(out_image)
            #out_image = np.clip(out_image, a_min=0., a_max=255.)
            #不需要调换顺序
            out_image = out_image / np.max(abs(out_image))
            out_image = np.asarray(out_image, np.float32)
            out_image = np.reshape(out_image, (1, image_s, image_h, image_w))
            #out_image = np.transpose(out_image / 255. - 0.5, (0,1,2))
            # pts = rearrange_pts(pts)

            pts2 = transform.rescale_pts(pts_2, down_ratio=down_ratio) #将坐标大小缩小
            data_series.append((out_image,pts2))
    else:
        spine_localisation_eval_pts = spine_localisation_eval_dict['pts']
        spine_localisation_eval_center = spine_localisation_eval_dict['pts_center']

        # spine_localisation_bottom_z = spine_localisation_eval_dict['spine_localisation_bottom_z'][0]
        # spine_localisation_eval_center[0] += spine_localisation_bottom_z
        bo = pts_irc[:,0].min()
        # spine_localisation_eval_pts[:,0] += spine_localisation_bottom_z
        bottom_z = (spine_localisation_eval_pts[0][0]  - 25) if spine_localisation_eval_pts[0][0]  - 25 >=0 else 0
        bottom_z = np.floor(bottom_z).astype(np.int32)
        top_z = spine_localisation_eval_pts[4][0]  + 25
        top_z = np.ceil(top_z).astype(np.int32)
        # 所有点的z轴坐标要改变

        pts_irc[:, 0] = pts_irc[:, 0] - bottom_z
        # 根据预测的整个ct的中心点的x,y坐标取出原ct的一部分
        pts_center_y = spine_localisation_eval_center[1]
        pts_center_x = spine_localisation_eval_center[2]
        # 所有点的x，y轴坐标要减小
        pts_irc[:, 1] = pts_irc[:, 1] - ((pts_center_y-200) if pts_center_y-200>=0 else 0)
        pts_irc[:, 2] = pts_irc[:, 2] - ((pts_center_x-200) if pts_center_x-200>=0 else 0)
        # 截取包含5段脊椎的部分
        image_array_section = image_array[bottom_z:top_z,
                                          (pts_center_y-200) if pts_center_y-200>=0 else 0:(pts_center_y+200) if pts_center_y+200<=512 else 512,
                                          (pts_center_x-200) if pts_center_x-200>=0 else 0:(pts_center_x+200) if pts_center_x+200<=512 else 512]
        if aug_label:
            out_image, pts_2 = data_aug['train'](image_array_section.copy(), pts_irc)
        else:
            out_image, pts_2 = data_aug['val'](image_array_section.copy(), pts_irc)

        min = np.min(out_image)
        max = np.max(out_image)
        temp = 2048
        # out_image = np.clip(out_image, a_min=0., a_max=255.)
        # 不需要调换顺序
        #out_image = out_image / np.max(abs(out_image))
        out_image = out_image / temp
        out_image = np.asarray(out_image, np.float32)
        intense_image = out_image.copy()
        intense_image[intense_image >= 0.1] += 0.2
        intense_image[intense_image > 1] = 1
        intense_image = np.reshape(intense_image, (1, image_s, image_h, image_w))
        out_image = np.reshape(out_image, (1, image_s, image_h, image_w))
        # out_image = np.transpose(out_image / 255. - 0.5, (0,1,2))
        # pts = rearrange_pts(pts)

        pts2 = transform.rescale_pts(pts_2, down_ratio=down_ratio)

        data_series.append((out_image,intense_image, pts2,img_id,bottom_z))

    #return np.asarray(out_image, np.float32), pts2
    return data_series



def generate_ground_truth(image,
                          intense_image,
                          pts_2,
                          points_num,
                          image_s,
                          image_h,
                          image_w,
                          img_id,
                          full,
                          downsize,down_ratio):
    #因为要downsize为1/2，所以要将参数大小除2取整数

    hm = np.zeros((image_s//downsize, image_h//downsize, image_w//downsize), dtype=np.float32)
    # print(hm.shape)
    # plt.imshow(hm[0, :, :])
    # plt.show()
    ####################################### 待修改
    # wh = np.zeros((35, 2*4), dtype=np.float32)
    reg = np.zeros((points_num, 3), dtype=np.float32)  #reg表示ct_int和ct的差值
    ind = np.zeros((points_num), dtype=np.int64) # 每个landmark坐标值与ct长宽高形成约束？
    reg_mask = np.zeros((points_num), dtype=np.uint8)

    if pts_2[:,0].max()>image_s:
        print('s is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())
    if pts_2[:,2].max()>image_w:
        print('w is big', pts_2[:,2].max())

    if pts_2.shape[0]!=40:
        print('ATTENTION!! image {} pts does not equal to 40!!! '.format(img_id))

    #for k in range(5):
    #pts_2为irc坐标
    # pts = pts_2[7*k:7*k+7,:]
    pts = pts_2
    # 计算同一侧椎弓更上两点在x和y轴上的间距
    ct_landmark = []
    min_diameter = 99
    if full == False:
        zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[3,:]-pts[4,:])**2))
            # min([np.sqrt((pts[3,1]-pts[4,1])**2),
            #               np.sqrt((pts[5,1]-pts[6,1])**2)])
        zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[5,:]-pts[6,:])**2))
            # min([np.sqrt((pts[3,2]-pts[4,2])**2),
            #               np.sqrt((pts[5,2]-pts[6,2])**2)])
        cen_z1, cen_y1, cen_x1 = np.mean(pts[3:5], axis=0)
        cen_z2, cen_y2, cen_x2 = np.mean(pts[5:7], axis=0)
        ct_landmark.append(pts[1])
        ct_landmark.append([cen_z1, cen_y1, cen_x1])
        ct_landmark.append([cen_z2, cen_y2, cen_x2])
        min_diameter = min(zhuigonggeng_left_landmarks_distance, zhuigonggeng_right_landmarks_distance)
    else:
        for i in range(5):
            zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[8*i+4, :] - pts[8*i+5, :]) ** 2))
            # min([np.sqrt((pts[3,1]-pts[4,1])**2),
            #               np.sqrt((pts[5,1]-pts[6,1])**2)])
            zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[8*i+6, :] - pts[8*i+7, :]) ** 2))
            cen_z1, cen_y1, cen_x1 = np.mean(pts[8*i+4:8*i+6], axis=0)
            cen_z2, cen_y2, cen_x2 = np.mean(pts[8*i+6:8*i+8], axis=0)
            # ct_landmark.append(pts[7*i+1])
            for j in range(4):
                ct_landmark.append(pts[8 * i + j])

            ct_landmark.append([cen_z1, cen_y1, cen_x1])
            ct_landmark.append([cen_z2, cen_y2, cen_x2])
            diameter = min(zhuigonggeng_left_landmarks_distance, zhuigonggeng_right_landmarks_distance)
            min_diameter = min(min_diameter,diameter)
    # 椎弓更的landmark
    #要downsize，所以要除2
    ct_landmark = np.asarray(ct_landmark, dtype=np.float32)/downsize
    ct_landmark_int = np.floor(ct_landmark).astype(np.int32)
    #radius = gaussian_radius((math.ceil(distance_y), math.ceil(distance_x)))
    radius = max(0, int(min_diameter))//2
    #生成脊锥中心点的热图
    #draw_umich_gaussian(hm[0,:,:], ct_landmark_int, radius=radius)
    image_w = image_w//downsize
    image_h = image_h//downsize
    image_s = image_s//downsize
    for i in range(points_num):
        hm = draw_umich_gaussian(hm, ct_landmark_int[i], radius=radius)
        # 每个landmark坐标值与ct长宽高形成约束？
        ind[i] = ct_landmark_int[i][2] + ct_landmark_int[i][1] * image_w + ct_landmark_int[i][0] * (image_w * image_h)
        reg[i] = (ct_landmark[i] - ct_landmark_int[i]) * downsize * down_ratio  #reg 为坐标点的误差，因为坐标相对于原始坐标缩小为原大小的1/8，所以要乘8
        reg_mask[i] = 1
    normal_vector = [] #求三点构成平面的法向量
    # for i in range(5):
    #     point_a = ct_landmark_int[i]
    #     point_b = ct_landmark_int[i+1]
    #     point_c = ct_landmark_int[i+2]
    #     # point1 = np.asarray(point1)
    #     # point2 = np.asarray(point2)
    #     # point3 = np.asarray(point3)
    #     AB = np.asmatrix(point_b - point_a)
    #     AC = np.asmatrix(point_c - point_a)
    #     N = np.cross(AB, AC) # 向量叉乘，求法向量
    #     # 法向量：n={-C,-B,-A} 因为我们的顺序是{z,y,x},但是直接认为法向量是n={A,B,C}，不影响训练
    #     A = N[0, 0]
    #     B = N[0, 1]
    #     C = N[0, 2]
    #     #单位化法向量
    #     sum = np.sqrt(A**2 + B**2 + C**2)
    #     A = A/sum
    #     B = B/sum
    #     C = C/sum
    #     for j in range(3):
    #         normal_vector.append([A,B,C])
    normal_vector = np.asarray(normal_vector,dtype=np.float32)
    # for i in range(4):
    #     wh[k,2*i:2*i+2] = ct_landmark-pts[i,:]
    #hm = hm.reshape([1,1,image_s, image_h, image_w])
    #hm = torch.from_numpy(hm)
    #为了节约存储，将输入以及hm downsize为原来的1/2，input大小为120*200*200，hm大小为30*50*50
    max_pool_downsize = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
    #hm = max_pool_downsize(hm).numpy()
    #hm = hm.reshape(hm.shape[2:])
    #增强hm中的landmark的像素值
    # hm[hm>=1] +=1

    # image = torch.from_numpy(image)
    # image = max_pool_downsize(image).numpy()
    intense_image = torch.from_numpy(intense_image)
    intense_image = max_pool_downsize(intense_image).numpy()
    #ct_landmark_int = ct_landmark_int//2
    ret = {'input': intense_image,
           'origin_image':image,
           'hm': hm,
           'ind': ind,
           'reg': reg,
           # 'wh': wh,
           'reg_mask': reg_mask,
           'landmarks':ct_landmark_int,
           'normal_vector': normal_vector,
           }
    return ret

def spine_localisation_processing_train(image, pts, points_num,image_h, image_w, image_s, down_ratio, aug_label, img_id,itk_information, full):
    pts = np.array(pts)
    pts = pts.astype('float32')
    pts_irc = []  # ['index', 'row', 'col']
    for i in range(len(pts)):
        pts_irc.append(xyz2irc(image, pts[i]))
    pts_irc = np.asarray(pts_irc) #小数形式
    image_array = sitk.GetArrayFromImage(image)
    data_series = []
    data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),  # 转为float32格式
                                            # transform.PhotometricDistort(), #对比度，噪声，亮度
                                            # transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                            # transform.RandomMirror_w(),
                                            transform.Resize(s=image_s, h=image_h, w=image_w) #Resize了点坐标和img
                                            ]),  # resize
                'val': transform.Compose([transform.ConvertImgFloat(),
                                          transform.Resize(s=image_s, h=image_h, w=image_w)
                                          ])}
    #bottom_z = pts_irc[:,0].min() - 5
    #bottom_z = np.floor(bottom_z).astype(np.int32)
    bottom_z = 0
    #top_z = pts_irc[:,0].max() + 5
    #top_z = np.ceil(top_z).astype(np.int32)
    # 所有点的z轴坐标要改变
    #pts_irc[:, 0] = pts_irc[:, 0] - bottom_z
    # 截取包含5段脊椎的部分
    #image_array_section = image_array[bottom_z:top_z,:,:]
    image_array_section = image_array
    bottom_z = np.asarray([bottom_z])
    if aug_label:
        out_image, pts_2 = data_aug['train'](image_array_section.copy(), pts_irc)
    else:
        out_image, pts_2 = data_aug['val'](image_array_section.copy(), pts_irc)

    min = np.min(out_image)
    max = np.max(out_image)
    # out_image = np.clip(out_image, a_min=0., a_max=255.)
    # 不需要调换顺序
    temp = 2048

    #out_image = out_image / np.max(abs(out_image))  某些值过大，导致椎体区域归一化值很小
    out_image = out_image / temp
    out_image = np.asarray(out_image, np.float32)
    intense_image = out_image.copy()
    # intense_image[intense_image >= 0.1] += 0.2
    #intense_image[intense_image > 1] = 1
    intense_image = np.reshape(intense_image, (1, image_s, image_h, image_w))
    out_image = np.reshape(out_image, (1, image_s, image_h, image_w))
    # out_image = np.transpose(out_image / 255. - 0.5, (0,1,2))
    # pts = rearrange_pts(pts)

    pts2 = transform.rescale_pts(pts_2, down_ratio=down_ratio)
    data_series.append((out_image,intense_image, pts2,itk_information,img_id))

    #return np.asarray(out_image, np.float32), pts2
    return data_series

def spine_localisation_generate_ground_truth(image,
                          intense_image,
                          pts_2,
                          points_num,
                          image_s,
                          image_h,
                          image_w,
                          img_id,
                          itk_information,
                          full,
                          downsize,down_ratio):
    #因为要downsize为1/2，所以要将参数大小除2取整数

    hm = np.zeros((5,image_s//downsize, image_h//downsize, image_w//downsize), dtype=np.float32)
    # print(hm.shape)
    # plt.imshow(hm[0, :, :])
    # plt.show()
    ####################################### 待修改
    # wh = np.zeros((35, 2*4), dtype=np.float32)
    reg = np.zeros((points_num, 3), dtype=np.float32)  #reg表示ct_int和ct的差值
    ind = np.zeros((points_num), dtype=np.int64) # 每个landmark坐标值与ct长宽高形成约束？
    reg_mask = np.zeros((points_num), dtype=np.uint8)

    if pts_2[:,0].max()>image_s:
        print('s is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())
    if pts_2[:,2].max()>image_w:
        print('w is big', pts_2[:,2].max())

    if pts_2.shape[0]!=40:
        print('ATTENTION!! image {} pts does not equal to 40!!! '.format(img_id))

    #for k in range(5):
    #pts_2为irc坐标
    # pts = pts_2[7*k:7*k+7,:]
    pts = pts_2
    # 计算同一侧椎弓更上两点在x和y轴上的间距
    ct_landmark = []
    min_diameter = 99
    if full == False:
        zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[4,:]-pts[5,:])**2))
            # min([np.sqrt((pts[3,1]-pts[4,1])**2),
            #               np.sqrt((pts[5,1]-pts[6,1])**2)])
        zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[6,:]-pts[7,:])**2))
            # min([np.sqrt((pts[3,2]-pts[4,2])**2),
            #               np.sqrt((pts[5,2]-pts[6,2])**2)])
        temp1 = np.mean(pts[0:2], axis=0)
        temp2 = np.mean(pts[2:4], axis=0)
        cen_z, cen_y, cen_x = np.mean(temp1+temp2, axis=0)
        ct_landmark.append(pts[0]);ct_landmark.append(pts[1]);ct_landmark.append(pts[2]);ct_landmark.append(pts[3])
        ct_landmark.append([cen_z, cen_y, cen_x])
        min_diameter = min(zhuigonggeng_left_landmarks_distance, zhuigonggeng_right_landmarks_distance)
    else:
        for i in range(5):
            zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[8*i+4, :] - pts[8*i+5, :]) ** 2))
            # min([np.sqrt((pts[3,1]-pts[4,1])**2),
            #               np.sqrt((pts[5,1]-pts[6,1])**2)])
            zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[8*i+6, :] - pts[8*i+7, :]) ** 2))
            temp1 = np.mean(pts[8*i+0:8*i+2], axis=0)
            temp2 = np.mean(pts[8*i+2:8*i+4], axis=0)
            cen_z, cen_y, cen_x = np.mean([temp1,temp2], axis=0)
            #ct_landmark.append(pts[8*i+0]);ct_landmark.append(pts[8*i+1]);ct_landmark.append(pts[8*i+2]);ct_landmark.append(pts[8*i+3])
            ct_landmark.append([cen_z, cen_y, cen_x])
            diameter = min(zhuigonggeng_left_landmarks_distance, zhuigonggeng_right_landmarks_distance)
            min_diameter = min(min_diameter,diameter)
    # 椎弓更的landmark
    #要downsize，所以要除
    ct_landmark = np.asarray(ct_landmark, dtype=np.float32)/downsize
    ct_landmark_int = np.floor(ct_landmark).astype(np.int32)
    #radius = gaussian_radius((math.ceil(distance_y), math.ceil(distance_x)))
    radius = max(0, int(min_diameter))//2
    #生成脊锥中心点的热图
    #draw_umich_gaussian(hm[0,:,:], ct_landmark_int, radius=radius)
    image_w = image_w//downsize
    image_h = image_h//downsize
    image_s = image_s//downsize
    for i in range(points_num):
        hm[i] = draw_umich_gaussian(hm[i], ct_landmark_int[i], radius=radius)
        # 每个landmark坐标值与ct长宽高形成约束？
        ind[i] = ct_landmark_int[i][2] + ct_landmark_int[i][1] * image_w + ct_landmark_int[i][0] * (image_w * image_h)
        reg[i] = (ct_landmark[i] - ct_landmark_int[i]) * downsize * down_ratio  #reg 为坐标点的误差，因为坐标相对于原始坐标缩小为原大小的1/8，所以要乘8
        reg_mask[i] = 1
    normal_vector = [] #求三点构成平面的法向量
    # for i in range(5):
    #     point_a = ct_landmark_int[i]
    #     point_b = ct_landmark_int[i+1]
    #     point_c = ct_landmark_int[i+2]
    #     # point1 = np.asarray(point1)
    #     # point2 = np.asarray(point2)
    #     # point3 = np.asarray(point3)
    #     AB = np.asmatrix(point_b - point_a)
    #     AC = np.asmatrix(point_c - point_a)
    #     N = np.cross(AB, AC) # 向量叉乘，求法向量
    #     # 法向量：n={-C,-B,-A} 因为我们的顺序是{z,y,x},但是直接认为法向量是n={A,B,C}，不影响训练
    #     A = N[0, 0]
    #     B = N[0, 1]
    #     C = N[0, 2]
    #     #单位化法向量
    #     sum = np.sqrt(A**2 + B**2 + C**2)
    #     A = A/sum
    #     B = B/sum
    #     C = C/sum
    #     for j in range(3):
    #         normal_vector.append([A,B,C])
    normal_vector = np.asarray(normal_vector,dtype=np.float32)
    # for i in range(4):
    #     wh[k,2*i:2*i+2] = ct_landmark-pts[i,:]
    #hm = hm.reshape([1,1,image_s, image_h, image_w])
    #hm = torch.from_numpy(hm)
    #为了节约存储，将输入以及hm downsize为原来的1/2，input大小为120*200*200，hm大小为30*50*50
    max_pool_downsize = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
    #hm = max_pool_downsize(hm).numpy()
    #hm = hm.reshape(hm.shape[2:])
    #增强hm中的landmark的像素值
    # hm[hm>=1] +=1

    # image = torch.from_numpy(image)
    # image = max_pool_downsize(image).numpy()
    intense_image = torch.from_numpy(intense_image)
    intense_image = max_pool_downsize(intense_image)
    intense_image = max_pool_downsize(intense_image).numpy()
    #ct_landmark_int = ct_landmark_int//2
    ret = {'input': intense_image,
           'origin_image':image,
           'hm': hm,
           'ind': ind,
           'reg': reg,
           # 'wh': wh,
           'reg_mask': reg_mask,
           'landmarks':ct_landmark_int,
           'normal_vector': normal_vector,
           'itk_information':itk_information
           }
    return ret
