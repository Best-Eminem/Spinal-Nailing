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


def processing_test(image, input_h, input_w, input_s):
    # print(image.shape)
    image = resize_image_itk(image, (input_w, input_h, input_s),resamplemethod=sitk.sitkLinear)
    image = sitk.GetArrayFromImage(image)
    out_image = image / np.max(abs(image))
    out_image = np.asarray(out_image, np.float32)

    out_image = out_image.reshape(1, 1, input_s,input_h, input_w)
    out_image = torch.from_numpy(out_image)
    return out_image


def draw_spinal(pts, out_image):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for i in range(4):
        cv2.circle(out_image, (int(pts[i, 0]), int(pts[i, 1])), 3, colors[i], 1, 1)
        cv2.putText(out_image, '{}'.format(i+1), (int(pts[i, 0]), int(pts[i, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0),1,1)
    for i,j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(out_image,
                 (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[j, 0]), int(pts[j, 1])),
                 color=colors[i], thickness=1, lineType=1)
    return out_image


def rearrange_pts(pts):
    # rearrange left right sequence
    boxes = []
    centers = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        # argsort返回排序后的原始序号
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
        centers.append(np.mean(pts_4, axis=0))
    bboxes = np.asarray(boxes, np.float32)
    # rearrange top to bottom sequence
    centers = np.asarray(centers, np.float32)
    sort_tb = np.argsort(centers[:,1])
    new_bboxes = []
    for sort_i in sort_tb:
        new_bboxes.append(bboxes[4*sort_i, :])
        new_bboxes.append(bboxes[4*sort_i+1, :])
        new_bboxes.append(bboxes[4*sort_i+2, :])
        new_bboxes.append(bboxes[4*sort_i+3, :])
    new_bboxes = np.asarray(new_bboxes, np.float32)
    return new_bboxes


def generate_ground_truth(image,
                          intense_image,
                          pts_2,
                          points_num,
                          image_s,
                          image_h,
                          image_w,
                          img_id,
                          full):
    #因为要downsize为1/2，所以要将参数大小除2取整数
    downsize = 2
    down_ratio = 4
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

    if pts_2.shape[0]!=35:
        print('ATTENTION!! image {} pts does not equal to 35!!! '.format(img_id))

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
        cen_z1, cen_y1, cen_x1 = np.round(np.mean(pts[3:5], axis=0))
        cen_z2, cen_y2, cen_x2 = np.round(np.mean(pts[5:7], axis=0))
        ct_landmark.append(pts[1])
        ct_landmark.append([cen_z1, cen_y1, cen_x1])
        ct_landmark.append([cen_z2, cen_y2, cen_x2])
        min_diameter = min(zhuigonggeng_left_landmarks_distance, zhuigonggeng_right_landmarks_distance)
    else:
        for i in range(5):
            zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[7*i+3, :] - pts[7*i+4, :]) ** 2))
            # min([np.sqrt((pts[3,1]-pts[4,1])**2),
            #               np.sqrt((pts[5,1]-pts[6,1])**2)])
            zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[7*i+5, :] - pts[7*i+6, :]) ** 2))
            cen_z1, cen_y1, cen_x1 = np.mean(pts[7*i+3:7*i+5], axis=0)
            cen_z2, cen_y2, cen_x2 = np.mean(pts[7*i+5:7*i+7], axis=0)
            ct_landmark.append(pts[7*i+1])
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
    image_w = image_w//2
    image_h = image_h//2
    image_s = image_s//2
    for i in range(points_num):
        hm = draw_umich_gaussian(hm, ct_landmark_int[i], radius=radius)
        # 每个landmark坐标值与ct长宽高形成约束？
        ind[i] = ct_landmark_int[i][2] + ct_landmark_int[i][1] * image_w + ct_landmark_int[i][0] * (image_w * image_h)
        reg[i] = (ct_landmark[i] - ct_landmark_int[i]) * down_ratio * downsize #reg 为坐标点的误差，因为坐标相对于原始坐标缩小为原大小的1/8，所以要乘8
        reg_mask[i] = 1
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
           'landmarks':ct_landmark_int
           }
    return ret

# def filter_pts(pts, w, h):
#     pts_new = []
#     for pt in pts:
#         if any(pt) < 0 or pt[0] > w - 1 or pt[1] > h - 1:
#             continue
#         else:
#             pts_new.append(pt)
#     return np.asarray(pts_new, np.float32)


def processing_train(image, pts, points_num,image_h, image_w, image_s, down_ratio, aug_label, img_id,full):
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

            pts2 = transform.rescale_pts(pts_2, down_ratio=down_ratio)
            data_series.append((out_image,pts2))
    else:
        bottom_z = pts_irc[:,0].min() - 5
        bottom_z = np.floor(bottom_z).astype(np.int32)
        top_z = pts_irc[:,0].max() + 5
        top_z = np.ceil(top_z).astype(np.int32)
        # 所有点的z轴坐标要改变
        pts_irc[:, 0] = pts_irc[:, 0] - bottom_z
        # 所有点的x，y轴坐标要减小
        pts_irc[:, 1] = pts_irc[:, 1] - 55
        pts_irc[:, 2] = pts_irc[:, 2] - 55
        # 截取包含5段脊椎的部分
        image_array_section = image_array[bottom_z:top_z,55:455,55:455]
        if aug_label:
            out_image, pts_2 = data_aug['train'](image_array_section.copy(), pts_irc)
        else:
            out_image, pts_2 = data_aug['val'](image_array_section.copy(), pts_irc)

        min = np.min(out_image)
        max = np.max(out_image)
        # out_image = np.clip(out_image, a_min=0., a_max=255.)
        # 不需要调换顺序
        out_image = out_image / np.max(abs(out_image))
        out_image = np.asarray(out_image, np.float32)
        intense_image = out_image.copy()
        intense_image[intense_image >= 0.1] += 0.2
        intense_image[intense_image > 1] = 1
        intense_image = np.reshape(intense_image, (1, image_s, image_h, image_w))
        out_image = np.reshape(out_image, (1, image_s, image_h, image_w))
        # out_image = np.transpose(out_image / 255. - 0.5, (0,1,2))
        # pts = rearrange_pts(pts)

        pts2 = transform.rescale_pts(pts_2, down_ratio=down_ratio)
        data_series.append((out_image,intense_image, pts2))

    #return np.asarray(out_image, np.float32), pts2
    return data_series