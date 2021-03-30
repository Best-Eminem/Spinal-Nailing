import os
from shutil import move
train_path = 'E:\\ZN-CT'

file_list=[]
for i,j,k in os.walk(train_path):
    for file in k:
        if file.endswith('.gz'):
            file_list.append(os.path.join(i,file))
print(len(file_list))
for i in range(len(file_list)):
    move(file_list[i],'E:\\ZN-CT-nii\\'+str(i+1)+'.nii.gz')