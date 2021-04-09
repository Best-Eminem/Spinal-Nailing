import SimpleITK as sitk
import cv2
import torch
# from draw_gaussian import *
from draw_3dgaussian import *
import transform
import math
from transform import resize_image_itk
from xyz2irc_irc2xyz import xyz2irc


def processing_test(image, input_h, input_w, input_s):
    # print(image.shape)
    image = resize_image_itk(image, (512, 512, 350),resamplemethod=sitk.sitkLinear)
    image = sitk.GetArrayFromImage(image)
    out_image = image.astype(np.float32) / 255.
    out_image = out_image - 0.5

    # 转置，将通道数放在前 （1024，512,3）->（3，1024，512）
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
                          pts_2,
                          image_s,
                          image_h,
                          image_w,
                          img_id):

    hm = np.zeros((image_s, image_h, image_w), dtype=np.float32)
    # print(hm.shape)
    # plt.imshow(hm[0, :, :])
    # plt.show()
    ####################################### 待修改
    # wh = np.zeros((35, 2*4), dtype=np.float32)
    reg = np.zeros((15, 3), dtype=np.float32)  #reg表示ct_int和ct的差值
    ind = np.zeros((15), dtype=np.int64) # 每个landmark坐标值与ct长宽高形成约束？
    reg_mask = np.zeros((15), dtype=np.uint8)

    if pts_2[:,0].max()>image_s:
        print('s is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())
    if pts_2[:,2].max()>image_w:
        print('w is big', pts_2[:,2].max())

    if pts_2.shape[0]!=35:
        print('ATTENTION!! image {} pts does not equal to 15!!! '.format(img_id))

    for k in range(5):
        #pts_2为irc坐标
        pts = pts_2[7*k:7*k+7,:]
        # 计算同一侧椎弓更上两点在x和y轴上的间距
        zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[3,1:]-pts[4,1:])**2))
            # min([np.sqrt((pts[3,1]-pts[4,1])**2),
            #               np.sqrt((pts[5,1]-pts[6,1])**2)])
        zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[5,1:]-pts[6,1:])**2))
            # min([np.sqrt((pts[3,2]-pts[4,2])**2),
            #               np.sqrt((pts[5,2]-pts[6,2])**2)])
        # 椎弓更的landmark
        ct_landmark = []
        cen_z1, cen_y1, cen_x1 = np.round(np.mean(pts[3:5], axis=0))
        cen_z2, cen_y2, cen_x2 = np.round(np.mean(pts[5:7], axis=0))
        ct_landmark.append(pts[1])
        ct_landmark.append([cen_z1, cen_y1, cen_x1])
        ct_landmark.append([cen_z2, cen_y2, cen_x2])
        ct_landmark = np.round(np.asarray(ct_landmark, dtype=np.float32))
        ct_landmark_int = ct_landmark.astype(np.int32)
        #radius = gaussian_radius((math.ceil(distance_y), math.ceil(distance_x)))
        diameter = min(zhuigonggeng_left_landmarks_distance,zhuigonggeng_right_landmarks_distance)
        radius = max(0, int(diameter))//2
        #生成脊锥中心点的热图
        #draw_umich_gaussian(hm[0,:,:], ct_landmark_int, radius=radius)
        for i in range(3):
            hm = draw_umich_gaussian(hm, ct_landmark_int[i], radius=radius)
            # 每个landmark坐标值与ct长宽高形成约束？
            ind[3*k+i] = ct_landmark_int[i][2] * image_w + ct_landmark_int[i][1] * image_h + ct_landmark_int[i][0]
            reg[3*k+i] = ct_landmark[i] - ct_landmark_int[i]
            reg_mask[3*k+i] = 1
        # for i in range(4):
        #     wh[k,2*i:2*i+2] = ct_landmark-pts[i,:]
    hm.reshape([1,1,image_s, image_h, image_w])

    ret = {'input': image,
           'hm': hm,
           'ind': ind,
           'reg': reg,
           # 'wh': wh,
           'reg_mask': reg_mask,
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


def processing_train(image, pts, image_h, image_w, image_s, down_ratio, aug_label, img_id):
    # filter pts ----------------------------------------------------
    # h,w,c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    pts = np.array(pts)
    pts = pts.astype('float32')
    pts_irc = []  # ['index', 'row', 'col']
    for i in range(len(pts)):
        pts_irc.append(xyz2irc(image, pts[i]))
    pts_irc = np.asarray(pts_irc)
    image_array = sitk.GetArrayFromImage(image)
    data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),  #转为float32格式
                                            # transform.PhotometricDistort(), #对比度，噪声，亮度
                                            #transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                            #transform.RandomMirror_w(),
                                            transform.Resize(s=image_s,h=image_h, w=image_w)
                                            ]),# resize
                'val': transform.Compose([transform.ConvertImgFloat(),
                                          transform.Resize(s=image_s,h=image_h, w=image_w)
                                          ])}
    if aug_label:
        out_image, pts_2 = data_aug['train'](image_array.copy(), pts_irc)
    else:
        out_image, pts_2 = data_aug['val'](image_array.copy(), pts_irc)
    min = np.min(out_image)
    max = np.max(out_image)
    #out_image = np.clip(out_image, a_min=0., a_max=255.)
    #不需要调换顺序
    out_image = out_image / np.max(abs(out_image))
    #out_image = np.transpose(out_image / 255. - 0.5, (0,1,2))
    # pts = rearrange_pts(pts)

    pts2 = transform.rescale_pts(pts_2, down_ratio=down_ratio)

    return np.asarray(out_image, np.float32), pts2