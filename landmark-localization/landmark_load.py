# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 17:08
# @Author  : Yike Cheng
# @FileName: landmark_load.py
# @Software: PyCharm
import csv
import numpy as np

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

def iterator(self, id_list_filename, random):
    #返回访问指定一些数据的迭代器
    return 1

def datasources(file_name, num_landmarks, dim):
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
    datasources_dict['image'] = image_data_source(self.image_base_folder,
                                                  '',
                                                  '',
                                                  '.nii.gz',
                                                  set_zero_origin=False,
                                                  set_identity_direction=False,
                                                  set_identity_spacing=False,
                                                  sitk_pixel_type=sitk.sitkInt16,
                                                  preprocessing=self.preprocessing,
                                                  name='image',
                                                  parents=[iterator])

    datasources_dict['landmarks'] = load_landmarks(file_name, num_landmarks, dim)
    return datasources_dict


load_landmarks('landmarks.csv',26,dim)