# 第一步：当前目录下所有文件夹下的文件名(不带后缀)
import os
import SimpleITK as sitk

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


# 如果文件夹不存在创建文件夹
def Makedir(path):
    folder = os.path.exists(path)
    if (not folder):
        os.makedirs(path)

def del_package(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_package(c_path)
        else:
            os.remove(c_path)
    os.removedirs(path)


if __name__ == '__main__':

    fileDir = "E:/ZN-CT-nii/VertebraeCT/JJL-CT"  # 输入文件夹路径
    files = GetFileName(fileDir)
    # for path in files:
    #     if path[-7:-1] =='output':
    #         del_package(path)
    # print(files)

    # 第二步：转化为.nii

    # 设定输出.nii文件的路径，为了方便这里将路径中的xuezhong文件夹旁新建xuezhong_output文件夹，xuezhong_output文件夹内的文件名保持不变
    # outputfiles = []
    # for i in range(len(files)):
    #     outputfiles.append(files[i][:51] + "_output" + files[i][51:])
    #
    # for path in outputfiles:
    #     Makedir(path)

    for i in range(len(files)):
        filepath = files[i]  # 读取路径
        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
        series_reader = sitk.ImageSeriesReader()  # 读取数据端口
        series_reader.SetFileNames(series_file_names)
        images = series_reader.Execute()  # 读取数据
        sitk.WriteImage(images, "E:/ZN-CT-nii/VertebraeCT/JJL-CT/nii/"+str(i+1+50)+".nii.gz")  # 保存为nii




