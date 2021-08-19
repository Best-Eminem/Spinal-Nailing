
import numpy as np
from numpy import random
import cv2
import SimpleITK as sitk

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
def rescale_pts(pts, down_ratio):
    return np.asarray(pts, np.float32)/float(down_ratio)

def float_uniform(low, high, size=None):
    """
    Create random floats in the lower and upper bounds - uniform distribution.
    :param low: Minimum value.
    :param high: Maximum value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.float32)
    return float(values)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, hm):
        for t in self.transforms:
            img, hm = t(img, hm)
        return img, hm

class ConvertImgFloat(object):
    def __call__(self, img, pts):
        return img.astype(np.float32), pts.astype(np.float32)

class RandomContrast(object):
    def __init__(self, lower=0.75, upper=1.25):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img,hm):
        #if random.randint(2):
        alpha = random.uniform(self.lower, self.upper)
        img *= alpha
        return img,hm


class RandomBrightness(object):
    def __init__(self, lower=-0.25, upper=0.25):
        self.lower = lower
        self.upper = upper

    def __call__(self, img,hm):
        #if random.randint(2):
        delta = random.uniform(self.lower, self.upper)
        img += delta
        return img,hm

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, pts):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, pts


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast() #将图像中所有数随机乘上一个数
        self.rb = RandomBrightness()#将图像中所有数随机加上一个数
        # self.rln = RandomLightingNoise()# 改变图像中的通道顺序,ct只有1个通道，因此不考虑

    def __call__(self, img, hm):
        img_array = sitk.GetArrayFromImage(img)
        if random.randint(2):
            img_array,hm = self.rb(img_array, hm)
        if random.randint(2):
            img_array,hm = self.pd(img_array, hm)
        # img, pts = self.rln(img, pts)
        #img = sitk.GetImageFromArray(img_array)
        for i in range(len(hm)):
            hm[i] = sitk.GetArrayFromImage(hm[i])
        return img_array, hm


class Expand(object):
    def __init__(self, max_scale = 1.5, mean = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, pts):
        if random.randint(2):
            return img, pts
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        z1 = random.uniform(0, s*ratio-s)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        if np.max(pts[:,0])+int(x1)>w-1 or np.max(pts[:,1])+int(y1)>h-1:  # keep all the pts
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
            expand_img[:,:,:] = self.mean
            expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win = 0.9):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            # (0.1, None),
            # (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, pts):
        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, pts
            for _ in range(50):
                current_img = img
                current_pts = pts
                w = random.uniform(self.min_win*width, width)
                h = random.uniform(self.min_win*height, height)
                if h/w<self.ratio[0] or h/w>self.ratio[1]:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                current_pts[:, 0] -= rect[1]
                current_pts[:, 1] -= rect[0]
                pts_new = []
                for pt in current_pts:
                    if any(pt)<0 or pt[0]>current_img.shape[1]-1 or pt[1]>current_img.shape[0]-1:
                        continue
                    else:
                        pts_new.append(pt)

                return current_img, np.asarray(pts_new, np.float32)

class RandomMirror_w(object):
    def __call__(self, img, pts):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1,:]
            pts[:,0] = w-pts[:,0]
        return img, pts

class RandomMirror_h(object):
    def __call__(self, img, pts):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:,:]
            pts[:,1] = h-pts[:,1]
        return img, pts


class Resize(object):
    def __init__(self, s, h, w):
        self.pts_dsize = (s,h,w)
        self.img_dsize = (w,h,s)

    def __call__(self, img, pts):
        s,h,w = img.shape
        pts[:, 0] = pts[:, 0]/s*self.pts_dsize[0]
        pts[:, 1] = pts[:, 1]/h*self.pts_dsize[1]
        pts[:, 2] = pts[:, 2]/w*self.pts_dsize[2]
        img = resize_image_itk(sitk.GetImageFromArray(img), newSize=self.img_dsize, resamplemethod=sitk.sitkLinear)
        return sitk.GetArrayFromImage(img), pts

class RandomTranslation(object):
    def __init__(self, dim, offset,spacing):

        self.random_offset = offset
        self.current_offset = np.asarray([float_uniform(-self.random_offset[i], self.random_offset[i])
                          for i in range(len(self.random_offset))])
        self.spacing = np.asarray(spacing)
        self.current_offset = self.current_offset * spacing
        t = sitk.AffineTransform(dim)
        t_hm = sitk.AffineTransform(dim)
        self.offset_with_used_dimensions_only = [o if used else 0 for used, o in zip([True,True,False], self.current_offset)]
        self.offset_with_used_dimensions_only_hm = [o/2 if used else 0 for used, o in zip([True,True,False], self.current_offset)]
        #print(self.offset_with_used_dimensions_only)
        t.Translate(self.offset_with_used_dimensions_only)
        t_hm.Translate(self.offset_with_used_dimensions_only_hm)

        self.get_translate_transform = t
        self.get_translate_transform_hm = t_hm
    def __call__(self, img, hm):
        if random.randint(2):
            img = sitk.Resample(img,self.get_translate_transform)
            for i in range(len(hm)):
                hm[i] = sitk.Resample(hm[i],self.get_translate_transform_hm)
        return img, hm


class RandomRotation(object):
    def __init__(self, dim, angles, center):
        self.center = center
        radian = np.asarray(angles) * np.pi / 180
        self.random_angles = radian
        angle = float_uniform(-self.random_angles[0], self.random_angles[0])
        self.current_angles = [angle] * dim
        r = sitk.AffineTransform(dim)
        r.SetCenter(center)
        #print(self.current_angles)
        #t.Rotate(1, 2, angle=self.current_angles[0])
        # rotate about y axis
        #t.Rotate(0, 2, angle=self.current_angles[1])
        # rotate about z axis
        r.Rotate(0, 1, angle=self.current_angles[2])
        self.get_rotation_transform = r

    def __call__(self, img, hm):
        if random.randint(2):
            img = sitk.Resample(img, self.get_rotation_transform)
            self.get_rotation_transform.SetCenter(self.center / 2)
            for i in range(len(hm)):
                hm[i] = sitk.Resample(hm[i],self.get_rotation_transform)

        return img, hm

class RandomScale(object):
    def __init__(self, dim, scale_factor, center):
        self.center = center
        self.random_scale = scale_factor
        scale = 1.0 + float_uniform(-self.random_scale, self.random_scale)
        self.current_scale = [scale] * 3
        #print(self.current_scale)
        s = sitk.AffineTransform(dim)
        s.SetCenter(center)
        s.Scale(self.current_scale)
        self.get_scale_transform = s
    def __call__(self, img,hm):
        if random.randint(2):
            img = sitk.Resample(img, self.get_scale_transform)
            self.get_scale_transform.SetCenter(self.center / 2)
            for i in range(len(hm)):
                hm[i] = sitk.Resample(hm[i],self.get_scale_transform)
                #hm = sitk.Resample(hm, self.get_scale_transform)
        return img,hm