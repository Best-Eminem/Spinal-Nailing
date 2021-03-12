# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 17:08
# @Author  : Yike Cheng
# @FileName: landmark_load.py
# @Software: PyCharm
import csv
import numpy as np
import SimpleITK as sitk

def load_landmarks(file_name, num_landmarks, dim):
    #从csv文件加载landmarks，根据图像id返回landmarks的list
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            # assert num_entries == len(
            #     row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries,
            #                                                                                          len(row))
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if np.isnan(float(row[i])):
                    landmark = 'nan'
                elif dim == 3:
                    coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    landmark = coords
                landmarks.append(landmark)
            landmarks_dict[id] = landmarks
    print(landmarks_dict)
    return landmarks_dict
verse004 =  load_landmarks('landmarks.csv', 26, 3)['verse004']
print(verse004.shape)
# ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', array([ 58.569016,  91.04301 , 298.7844  ], dtype=float32), array([ 58.83996,  89.57451, 275.3636 ], dtype=float32), array([ 59.071316,  85.364815, 248.5304  ], dtype=float32), array([ 59.88155,  81.28029, 218.6988 ], dtype=float32), array([ 60.812527,  77.72729 , 192.53966 ], dtype=float32), array([ 61.897564,  65.51127 , 166.84135 ], dtype=float32), array([ 63.70792,  59.27282, 134.9608 ], dtype=float32), array([ 64.93076,  58.55445, 104.3237 ], dtype=float32), array([65.117226, 66.66109 , 75.51525 ], dtype=float32), 'nan', 'nan']
def iterator(id_list_filename, random = False, now_index = -1):
    #返回文件中的所有图像文件名或下一个图像文件名
    image_id_list = []
    with open(id_list_filename, 'r') as file:
        for line in file:
            image_id_list.append(line[:-1])
    #print(image_id_list)
    if now_index != -1:
        return image_id_list[now_index + 1]

    return image_id_list

def intensity_preprocessing_ct(image):
    """
    输入为sitk图像。输出为预处理后的sitk图像
    Intensity preprocessing function, 在重新采样之前，对加载的sitk图像进行处理.
    """
    clamp_filter = sitk.ClampImageFilter()

    # 计算sitk图像的最大 intensity
    min_max_filter = sitk.MinimumMaximumImageFilter()
    min_max_filter.Execute(image)
    clamp_max = min_max_filter.GetMaximum()

    #设置图像的 intensity的最大和最小值，超过max的设置到max，小于 min的设置到 min
    clamp_filter.SetLowerBound(float(clamp_min=-1024))
    clamp_filter.SetUpperBound(float(clamp_max))
    output_image = clamp_filter.Execute(image)

    image = output_image
    # 高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。该函数对图像进行高斯滤波平滑去噪。
    return sitk.SmoothingRecursiveGaussian(image, input_gaussian_sigma = 1.0)

def intensity_postprocessing_ct_random(self, image):
    """
    Intensity postprocessing for CT input. Random augmentation version.
    :param image: The np input image.
    :return: The processed image.
    """
    return ShiftScaleClamp(shift=0,
                           scale=1 / 2048,
                           random_shift=self.random_intensity_shift,
                           random_scale=self.random_intensity_scale,
                           clamp_min=-1.0,
                           clamp_max=1.0)(image)

def ImageDataSource(image_base_folder, file_ext, sitk_pixel_type=sitk.sitkInt16, preprocessing=intensity_preprocessing_ct):
    return 1

def datasources(image_base_folder, landmark_file_name, num_landmarks, dim):
    # 返回指定的一批数据的图像和landmarks
    """
    Returns the data sources that load data.
    {
    'image:' CachedImageDataSource that loads the image files.
    'labels:' CachedImageDataSource that loads the groundtruth labels.
    'landmarks:' LandmarkDataSource that loads the landmark coordinates.
    }
    :param iterator: The dataset iterator.
    :param cached: If true, use CachedImageDataSource, else ImageDataSource.
    :return: A dict of data sources.
    """
    datasources_dict = {}
    # image_data_source = CachedImageDataSource if cached else ImageDataSource
    image_data_source = ImageDataSource
    # ImageDataSource待补充
    datasources_dict['image'] = image_data_source(image_base_folder,
                                                  '.nii.gz',
                                                  sitk_pixel_type=sitk.sitkInt16,
                                                  preprocessing=intensity_preprocessing_ct)

    datasources_dict['landmarks'] = load_landmarks(landmark_file_name, num_landmarks, dim)
    return datasources_dict

def data_generators(self, iterator, datasources, transformation, image_post_processing, random_translation_single_landmark, image_size):
    """
    Returns the data generators that process one input. See datasources() for dict values.
    :param datasources: datasources dict.
    :param transformation: transformation.
    :param image_post_processing: The np postprocessing function for the image data generator.
    :return: A dict of data generators.
    """
    generators_dict = {}
    generators_dict['image'] = ImageGenerator(self.dim,
                                              image_size,
                                              self.image_spacing,
                                              interpolator='linear',
                                              post_processing_np=image_post_processing,
                                              data_format=self.data_format,
                                              resample_default_pixel_value=self.image_default_pixel_value,
                                              name='image',
                                              parents=[datasources['image'], transformation])
    if self.generate_landmark_mask:
        generators_dict['landmark_mask'] = ImageGenerator(self.dim,
                                                          image_size,
                                                          self.image_spacing,
                                                          interpolator='nearest',
                                                          data_format=self.data_format,
                                                          resample_default_pixel_value=0,
                                                          name='landmark_mask',
                                                          parents=[datasources['landmark_mask'], transformation])
    if self.generate_labels or self.generate_single_vertebrae:
        generators_dict['labels'] = ImageGenerator(self.dim,
                                                   image_size,
                                                   self.image_spacing,
                                                   interpolator='nearest',
                                                   post_processing_np=self.split_labels,
                                                   data_format=self.data_format,
                                                   name='labels',
                                                   parents=[datasources['labels'], transformation])
    if self.generate_heatmaps or self.generate_spine_heatmap:
        generators_dict['heatmaps'] = LandmarkGeneratorHeatmap(self.dim,
                                                               image_size,
                                                               self.image_spacing,
                                                               sigma=self.heatmap_sigma,
                                                               scale_factor=1.0,
                                                               normalize_center=True,
                                                               data_format=self.data_format,
                                                               name='heatmaps',
                                                               parents=[datasources['landmarks'], transformation])
    if self.generate_landmarks:
        generators_dict['landmarks'] = LandmarkGenerator(self.dim,
                                                         image_size,
                                                         self.image_spacing,
                                                         data_format=self.data_format,
                                                         name='landmarks',
                                                         parents=[datasources['landmarks'], transformation])
    if self.generate_single_vertebrae_heatmap:
        single_landmark = LambdaNode(lambda id_dict, landmarks: landmarks[int(id_dict['landmark_id']):int(id_dict['landmark_id']) + 1],
                                     name='single_landmark',
                                     parents=[iterator, datasources['landmarks']])
        if random_translation_single_landmark:
            single_landmark = LambdaNode(lambda l: [Landmark(l[0].coords + float_uniform(-self.random_translation_single_landmark, self.random_translation_single_landmark, [self.dim]), True)],
                                         name='single_landmark_translation',
                                         parents=[single_landmark])
        generators_dict['single_heatmap'] = LandmarkGeneratorHeatmap(self.dim,
                                                                     image_size,
                                                                     self.image_spacing,
                                                                     sigma=self.heatmap_sigma,
                                                                     scale_factor=1.0,
                                                                     normalize_center=True,
                                                                     data_format=self.data_format,
                                                                     name='single_heatmap',
                                                                     parents=[single_landmark, transformation])
    if self.generate_single_vertebrae:
        if self.data_format == 'channels_first':
            generators_dict['single_label'] = LambdaNode(lambda id_dict, images: images[int(id_dict['landmark_id']) + 1:int(id_dict['landmark_id']) + 2, ...],
                                                         name='single_label',
                                                         parents=[iterator, generators_dict['labels']])
        else:
            generators_dict['single_label'] = LambdaNode(lambda id_dict, images: images[..., int(id_dict['landmark_id']) + 1:int(id_dict['landmark_id']) + 2],
                                                         name='single_label',
                                                         parents=[iterator, generators_dict['labels']])
    if self.generate_spine_heatmap:
        generators_dict['spine_heatmap'] = LambdaNode(lambda images: gaussian(np.sum(images, axis=0 if self.data_format == 'channels_first' else -1, keepdims=True), sigma=self.spine_heatmap_sigma),
                                                      name='spine_heatmap',
                                                      parents=[generators_dict['heatmaps']])

    return generators_dict

def spatial_transformation_augmented(self, iterator, datasources, image_size):
    """
    随机增强的空间图像变换。
    :param datasources: datasources dict.
    :return: The transformation.
    """
    translate_to_center_landmarks = True
    transformation_list = []
    kwparents = {'image': datasources['image']}
    if translate_to_center_landmarks:
        kwparents['landmarks'] = datasources['landmarks']
        transformation_list.append(translation.InputCenterToOrigin(self.dim, used_dimensions=[False, False, True]))
        transformation_list.append(landmark.Center(self.dim, True, used_dimensions=[True, True, False]))

    if self.translate_by_random_factor:
        transformation_list.append(translation.RandomFactorInput(self.dim, [0, 0, 0.5], [0, 0, self.image_spacing[2] * image_size[2]]))
    transformation_list.extend([translation.Random(self.dim, [self.random_translation] * self.dim),
                                rotation.Random(self.dim, [self.random_rotate] * self.dim),
                                scale.RandomUniform(self.dim, self.random_scale),
                                scale.Random(self.dim, [self.random_scale] * self.dim),
                                translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing),
                                deformation.Output(self.dim, [6, 6, 6], [self.random_deformation] * self.dim, image_size, self.image_spacing)])
    comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
    return LambdaNode(lambda comp: sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, size=self.image_size, outputSpacing=self.image_spacing)),
                      name='image',
                      kwparents={'comp': comp})

def dataset_train(id_list_filename, image_base_folder, landmark_file_name, num_landmarks, dim = 3):
    """
    Returns the training dataset. Random augmentation is performed.
    :return: The training dataset.
    """
    images_id = iterator(id_list_filename, now_index = -1)
    datasources_train_dict = datasources(image_base_folder, landmark_file_name, num_landmarks, dim)
    image_size = [None, None, None]

    reference_transformation = spatial_transformation_augmented(iterator, datasources_train_dict, image_size)
    generators = data_generators(iterator, datasources_train_dict, reference_transformation, postprocessing_random = intensity_postprocessing_ct_random, True, image_size)

    return GraphDataset(data_generators=list(generators.values()),
                        data_sources=list(sources.values()),
                        transformations=[reference_transformation],
                        iterator=iterator,
                        debug_image_folder='debug_train' if self.save_debug_images else None)