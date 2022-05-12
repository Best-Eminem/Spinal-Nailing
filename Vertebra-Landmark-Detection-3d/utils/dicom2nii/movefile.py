import os
from shutil import move
train_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_OriginLumbarMSK_Sections'

file_list=[]
for i,j,k in os.walk(train_path):
    for file in k:
        if file.find('msk')!=-1:
            file_list.append(os.path.join(i,file))
print(len(file_list))
for i in range(len(file_list)):
    move(file_list[i],'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_CT_OriginLumbarMSK_Sections/'+os.path.basename(file_list[i])[-14:])