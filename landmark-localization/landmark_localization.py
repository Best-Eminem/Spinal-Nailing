# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 20:33
# @Author  : Yike Cheng
# @FileName: landmark_localization.py
# @Software: PyCharm
from landmark_load import datasources,iterator
def init_datasets(id_list_filename, image_base_folder, landmark_file_name, num_landmarks, dim = 3):
    images_id = iterator(id_list_filename)
    dataset_train = datasources(image_base_folder, landmark_file_name, num_landmarks, dim)
    image_size = [None, None, None]





def model():
    return 'model'

def train():
    return ''

def test():
    return ''

def landmark_visualization(heatmap):
    # landmark热图可视化
    return 'landmark_visualization'