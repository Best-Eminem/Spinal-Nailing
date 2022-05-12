import nibabel as nib
import shutil
import argparse
from glob import glob
import os
import numpy as np
import itk
import SimpleITK as sitk
import multiprocessing
import nibabel as nib
import nibabel.orientations as nio

def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented


def smooth(image, sigma):
    """
    Smooth image with Gaussian smoothing.
    :param image: ITK image.
    :param sigma: Sigma for smoothing.
    :return: Smoothed image.
    """
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetSigma(sigma)
    filter.Update()
    smoothed = filter.GetOutput()
    return smoothed


def clamp(image):
    """
    Clamp image between -1024 to 8192.
    :param image: ITK image.
    :return: Clamped image.
    """
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.ClampImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetBounds(-1024, 8192)
    filter.Update()
    clamped = filter.GetOutput()
    return clamped

def setDicomWinWidthWinCenter(img_data, winwidth = 2178, wincenter = 253):
    img_temp = img_data
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in range(img_data.shape[0]):
        img_temp[i] = (img_temp[i]-min)
    #img_temp = img_temp.astype(np.int)
    min_index = img_temp < min
    img_temp[min_index] = -1024
    max_index = img_temp > max
    img_temp[max_index] = 8192

    return img_temp

def process_itk_error(filename):
    # image_folder = r'D:\CT\Verse CT\2020\error_itk\seg'
    # source_folder = r'D:\CT\Verse CT\2020\error_itk\source'
    # filenames = glob(os.path.join(image_folder, '*.nii.gz'))
    # filenames_source = glob(os.path.join(source_folder, '*.nii.gz'))
    # for i in range(len(filenames)):
    # img_seg = nib.load(filenames[i])
    img_source = nib.load(filename)
    qform = img_source.get_qform()
    img_source.set_qform(qform)
    nib.save(img_source, filename)

def process_image(filename, output_folder, sigma,setWidthCenter = False):
    """
    Reorient image at filename, smooth with sigma, clamp and save to output_folder.
    :param filename: The image filename.
    :param output_folder: The output folder.
    :param sigma: Sigma for smoothing.
    """
    process_itk_error(filename) #处理转换方向时的报错
    basename = os.path.basename(filename)
    basename_wo_ext = basename[:basename.find('.nii.gz')]
    print(basename_wo_ext)
    ImageType = itk.Image[itk.SS, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(filename)
    image = reader.GetOutput()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    reoriented = reorient_to_rai(image)
    if not basename_wo_ext.endswith('_msk'):
        if setWidthCenter:
            reoriented = itk.GetArrayFromImage(reoriented)
            reoriented = setDicomWinWidthWinCenter(reoriented,2178,253)
            reoriented = itk.GetImageFromArray(reoriented)
            reoriented = smooth(reoriented, sigma)
        else:
            reoriented = smooth(reoriented, sigma)
            reoriented = clamp(reoriented)
    reoriented.SetOrigin([0,0,0])
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    reoriented.SetDirection(m)
    reoriented.SetSpacing(spacing)
    reoriented.Update()
    itk.imwrite(reoriented, os.path.join(output_folder, basename_wo_ext + '.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    input = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/rawdata'
    #input = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/derivatives'

    #input = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/val/dataset-verse20validation/dataset-02validation/rawdata'
    # input = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/val/dataset-verse20validation/dataset-02validation/derivatives'


    #input = r'/home/gpu/Spinal-Nailing/ZN-CT-nii/data/k_fold_origin'
    parser.add_argument('--image_folder', type=str, default=input)
    parser.add_argument('--output_folder', type=str, default=r'/home/gpu/Spinal-Nailing/ZN-CT-nii/data/ourdata_processed')
    parser.add_argument('--sigma', type=float, default=0.75)
    parser.add_argument('--setWidthCenter', type=bool, default=True) #表示是否修改hu值
    parser_args = parser.parse_args()
    parser_args.output_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_processed'
    if not os.path.exists(parser_args.output_folder):
        os.makedirs(parser_args.output_folder)
    filenames = glob(os.path.join(parser_args.image_folder, '*/*.nii.gz'))
    # print(filenames)
    pool = multiprocessing.Pool(8)
    pool.starmap(process_image, [(filename, parser_args.output_folder,parser_args.sigma,parser_args.setWidthCenter) for filename in sorted(filenames)])
    # for filename in sorted(filenames):
    #     process_image(filename,parser_args.output_folder,parser_args.sigma)

    # path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/rawdata/sub-verse586/sub-verse586_dir-iso_ct.nii.gz'
    # img = sitk.ReadImage(path)
    # img = sitk.GetArrayFromImage(img)
    # img = setDicomWinWidthWinCenter(img,1000,130)
    # img = sitk.GetImageFromArray(img)
    # sitk.WriteImage(img,r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/verse586_temp.nii.gz')
