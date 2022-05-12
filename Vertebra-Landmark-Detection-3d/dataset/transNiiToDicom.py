import os
import SimpleITK as sitk
input_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/VerseOriginLumbarSections/Verse2020_CT_OriginLumbar_Sections'
output_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/VerseOriginLumbarSections/VerseDicom'
metadicomDir = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/VerseOriginLumbarSections/I1080000'
metadicom = sitk.ReadImage(metadicomDir)
file_list=[]
output_list = []
for i,j,k in os.walk(input_path):
    for file in k:
        file_list.append(os.path.join(i,file))
        out_file = os.path.join(output_path,os.path.basename(file)[:3])
        output_list.append(out_file)
        if not os.path.isdir(out_file):
            os.mkdir(out_file)

# filedir = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/VerseOriginLumbarSections/Verse2020_CT_OriginLumbar_Sections/500.nii.gz'
# outdir = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/VerseOriginLumbarSections/VerseDicom'
# if not os.path.isdir(outdir):
#     os.mkdir(outdir)
for infile,outfile in zip(file_list,output_list):
    img = sitk.ReadImage(infile)
    keys = metadicom.GetMetaDataKeys()
    space = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    # img_array = sitk.GetArrayFromImage(img)
    # for i in range(img_array.shape[0]):
    #     select_img = sitk.GetImageFromArray(img_array[i])
    # 'ITK_original_direction'
    # 'ITK_original_spacing'
    for key in keys:
        if key=='ITK_original_spacing' or key=='ITK_original_direction':
            print(metadicom.GetMetaData(key))
            img.SetMetaData(key,metadicom.GetMetaData(key))
        # select_img.SetSpacing(space)
        # select_img.SetOrigin(origin)
        # select_img.SetDirection(direction)
    sitk.WriteImage(img,os.path.join(outfile,str(100001)+'.dcm'))
