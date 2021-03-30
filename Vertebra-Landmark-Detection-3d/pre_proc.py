import SimpleITK as sitk
import cv2
import torch
from draw_gaussian import *
import transform
import math
# simpleItk resize 三维ct图像
def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    origin = itkimage.GetOrigin()
    #print('origin: ', origin)
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    #print('originSize: ', originSize)
    originSpacing = itkimage.GetSpacing()
    #print('originSpacing: ',originSpacing)
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)   # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    #itkimgResampled = itkimgResampled.SetSpacing([1, 1, 1])
    return itkimgResampled

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
                          image_h,
                          image_w,
                          img_id):
    hm = np.zeros((1, image_h, image_w), dtype=np.float32)
    # print(hm.shape)
    # plt.imshow(hm[0, :, :])
    # plt.show()
    wh = np.zeros((17, 2*4), dtype=np.float32)
    reg = np.zeros((17, 2), dtype=np.float32)
    ind = np.zeros((17), dtype=np.int64)
    reg_mask = np.zeros((17), dtype=np.uint8)

    if pts_2[:,0].max()>image_w:
        print('w is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())

    if pts_2.shape[0]!=68:
        print('ATTENTION!! image {} pts does not equal to 68!!! '.format(img_id))

    for k in range(17):
        pts = pts_2[4*k:4*k+4,:]
        # 上下两点间距和左右两点间距
        bbox_h = np.mean([np.sqrt(np.sum((pts[0,:]-pts[2,:])**2)),
                          np.sqrt(np.sum((pts[1,:]-pts[3,:])**2))])
        bbox_w = np.mean([np.sqrt(np.sum((pts[0,:]-pts[1,:])**2)),
                          np.sqrt(np.sum((pts[2,:]-pts[3,:])**2))])
        # 骨锥中心点
        cen_x, cen_y = np.mean(pts, axis=0)
        ct = np.asarray([cen_x, cen_y], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
        radius = max(0, int(radius))
        #生成脊锥中心点的热图
        #draw_umich_gaussian(hm[0,:,:], ct_int, radius=radius)
        hm[0,:,:] = draw_umich_gaussian(hm[0,:,:], ct_int, radius=radius)
        # plt.imshow(hm[0,:,:])
        # plt.show()
        ind[k] = ct_int[1] * image_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        for i in range(4):
            wh[k,2*i:2*i+2] = ct-pts[i,:]

    ret = {'input': image,
           'hm': hm,
           'ind': ind,
           'reg': reg,
           'wh': wh,
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


def processing_train(image, pts, image_h, image_w, input_s, down_ratio, aug_label, img_id):
    # filter pts ----------------------------------------------------
    h,w,c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),  #转为float32格式
                                            transform.PhotometricDistort(), #对比度，噪声，亮度
                                            transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                            transform.RandomMirror_w(),
                                            # transform.Resize(h=image_h, w=image_w)
                                            ]),# resize
                'val': transform.Compose([transform.ConvertImgFloat(),
                                          #transform.Resize(h=image_h, w=image_w)
                                          ])}
    if aug_label:
        out_image, pts = data_aug['train'](image.copy(), pts)
    else:
        out_image, pts = data_aug['val'](image.copy(), pts)

    out_image = np.clip(out_image, a_min=0., a_max=255.)
    #不需要调换顺序
    out_image = np.transpose(out_image / 255. - 0.5, (0,1,2))
    # pts = rearrange_pts(pts)
    pts2 = transform.rescale_pts(pts, down_ratio=down_ratio)

    return np.asarray(out_image, np.float32), pts2

