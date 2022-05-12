import SimpleITK as sitk
import os
from tqdm import tqdm
# coding: utf8
from pydicom import dcmread


# 利用os.listdir()、os.walk()获取文件夹和文件名
def GetFileName(fileDir):
    list_name = []
    for dir in os.listdir(fileDir):  # 获取当前目录下所有文件夹和文件(不带后缀)的名称
        filePath = os.path.join(fileDir, dir)  # 得到文件夹和文件的完整路径
        # print(filePath)
        if os.path.isdir(filePath):
            # 获取根目录路径、子目录路径，根目录和子目录下所有文件名
            for root, subDir, files in os.walk(filePath):
                for subfilepath in subDir:
                    subfilepath = os.path.join(root, subfilepath)  # 得到文件上层目录

                    list_name.append(subfilepath)

    # 判断文件夹中是否只含图像文件
    list = []
    for i in range(len(list_name)):
        temp = os.listdir(list_name[i])
        for doc in temp:
            path = os.path.join(list_name[i], doc)
            if os.path.isdir(path) == 1:
                list.append(i)
                break
    # 反向循环删除不需要的文件夹的路径，只保留只含图像文件的文件夹
    for i in reversed(list):
        del list_name[i]

    return list_name
if __name__ == '__main__':
    metas = [
            "PatientID",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "InstitutionName",
        ]
    anonymizations = {
            "PatientName": "HUA QIANG",
            "PatientBirthDate": "20210102",
            "PatientSex": "M",
            "InstitutionName":"a ba a ba",
        }
    list_name = []
    fileDir = 'F:/CT/JJL-CT'
    files = GetFileName(fileDir)
    for i in tqdm(range(len(files))):
        path = files[i]
        for dir in os.listdir(path):  # 获取当前目录下所有文件夹和文件(不带后缀)的名称
            filePath = os.path.join(path, dir)  # 得到文件夹和文件的完整路径
            ds = dcmread(filePath,force=True)
            # for meta in metas:
            #     if anonymizationKey in ds.dir():
            #         print(ds.data_element(meta))
            for anonymizationKey, val in anonymizations.items():
                if anonymizationKey in ds.dir(): 
                    ds.data_element(anonymizationKey).value = val

            # 写回
            ds.save_as(filePath)