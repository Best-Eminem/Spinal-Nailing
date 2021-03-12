# -*- coding: utf-8 -*-
# @Time    : 2021/3/12 20:10
# @Author  : Yike Cheng
# @FileName: test.py
# @Software: PyCharm
import os
import csv
import re
import numpy as np
class Landmark(object):
    """
    Landmark object that has coordinates, is_valid, a scale and value.
    """
    def __init__(self,
                 coords=None,
                 is_valid=None,
                 scale=1.0,
                 value=None):
        """
        Initializer.
        :param coords: The landmark coordinates.
        :param is_valid: Defines, if the landmark is valid, i.e., has coordinates.
                         If coords is not None and is_valid is None, self.is_valid will be set to True.
        :param scale: The scale of the landmark.
        :param value: The value of the landmark.
        """
        self.coords = coords
        self.is_valid = is_valid
        if self.is_valid is None:
            self.is_valid = self.coords is not None
        self.scale = scale
        self.value = value


def get_mean_coords(landmarks):
    """
    Returns mean coordinates of a landmark list.
    :param landmarks: Landmark list.
    :return: np.array of mean coordinates.
    """
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return np.nanmean(np.stack(valid_coords, axis=0), axis=0)

def get_mean_landmark(landmarks):
    """
    Returns a Landmark object, where the coordinates are the mean coordinates of the
    given landmark list. scale and value are ignored.
    :param landmarks: Landmark list.
    :return: Landmark object with the mean coordinates.
    """
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return Landmark(np.nanmean(np.stack(valid_coords, axis=0), axis=0))

def get_mean_landmark_list(*landmarks):
    """
    Returns a list of mean Landmarks for two or more given lists of landmarks. The given lists
    must have the same length. The mean of corresponding list entries is calculated with get_mean_landmark.
    :param landmarks: Two or more given lists of landmarks.
    :return: List of mean landmarks.
    """
    return [get_mean_landmark(l) for l in zip(*landmarks)]



def load_csv(file_name, num_landmarks, dim):
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries, len(row))
            # print(len(points_dict), name)
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if np.isnan(float(row[i])):
                    landmark = Landmark(None, False)
                else:
                    if dim == 2:
                        coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                    elif dim == 3:
                        coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    landmark = Landmark(coords)
                landmarks.append(landmark)
            landmarks_dict[id] = landmarks
    return landmarks_dict
landmarks_dict = load_csv('landmarks.csv', 26, 3)
print(landmarks_dict['verse004'][0].)