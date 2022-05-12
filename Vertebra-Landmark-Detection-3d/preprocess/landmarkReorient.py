import nibabel as nib
import nibabel.orientations as nio
from glob import glob
import os
import numpy as np
import shutil
import json

from numpy import source

def centroids_to_dict(ctd_list):
    """Converts the centroid list to a dictionary of centroids
    
    Parameters:
    ----------
    ctd_list: the centroid list
    
    Returns:
    ----------
    dict_list: a dictionart of centroids having the format dict[vertebra] = ['X':x, 'Y':y, 'Z': z]
    
    """
    dict_list = []
    for v in ctd_list:
        if any('nan' in str(v_item) for v_item in v): continue   #skipping invalid NaN values
        v_dict = {}
        if isinstance(v, tuple):
            v_dict['direction'] = v
        else:
            v_dict['label'] = int(v[0])
            v_dict['X'] = v[1]
            v_dict['Y'] = v[2]
            v_dict['Z'] = v[3]
        dict_list.append(v_dict)
    return dict_list
def load_centroids(ctd_path):
    """loads the json centroid file
    
    Parameters:
    ----------
    ctd_path: the full path to the json file
    
    Returns:
    ----------
    ctd_list: a list containing the orientation and coordinates of the centroids
    
    """
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):            #skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']]) 
    return ctd_list

def save_centroids(ctd_list, out_path):
    """Saves the centroid list to json file
    
    Parameters:
    ----------
    ctd_list: the centroid list
    out_path: the full desired save path
    
    """
    if len(ctd_list) < 2:
        print("[#] Centroids empty, not saved:", out_path)
        return
    json_object = centroids_to_dict(ctd_list)
    # Problem with python 3 and int64 serialisation.
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError
    with open(out_path, 'w') as f:
        json.dump(json_object, f, default=convert)
    print("[*] Centroids saved:", out_path)

def reorient_centroids_to(ctd_list,img,decimals=1, verb=False):
    """reorient centroids to image orientation
    
    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image 
    decimals: rounding decimal digits
    
    Returns:
    ----------
    out_list: reoriented list of centroids 
    
    """
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present") 
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    #axcodes_to = ('R','A','I')
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list

def process_itk():
    image_folder = r'D:\CT\Verse CT\2020\error_itk\seg'
    source_folder = r'D:\CT\Verse CT\2020\error_itk\source'
    filenames = glob(os.path.join(image_folder, '*.nii.gz'))
    filenames_source = glob(os.path.join(source_folder, '*.nii.gz'))
    for i in range(len(filenames)):
        img_seg = nib.load(filenames[i])
        img_source = nib.load(filenames_source[i])
        print(filenames[i],filenames_source[i])
        qform = img_source.get_qform()
        img_seg.set_qform(qform)
        sfrom = img_source.get_sform()
        img_seg.set_sform(sfrom)
        nib.save(img_seg, filenames[i])

def process_and_move_json():
    # json_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/derivatives'
    # json_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/val/dataset-verse20validation/dataset-02validation/derivatives' 
    # jsonnames = glob(os.path.join(json_folder, '*/*.json'))

    json_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_labels'
    jsonnames = sorted(glob(os.path.join(json_folder, '*.json')))

    image_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_processed'
    
    image_names = sorted(glob(os.path.join(image_folder, '*.nii.gz')))
    for i in range(len(jsonnames)):
        # shutil.copy(jsonnames[i],r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_labels')
        # print(jsonnames[i])
        flag = 0
        with open(jsonnames[i], 'r') as f:
            # load json file
            json_data = json.load(f)
            for dict in json_data:
                if 'direction' in dict:
                    if dict['direction'] != ['L', 'A', 'S']:
                        flag = 1
                    break
        if flag == 1:continue
        img = nib.load(image_names[i*2])
        ctd_list = load_centroids(jsonnames[i])
        ctd_list = reorient_centroids_to(ctd_list,img)
        savepath = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_RAI_labels'
        if not os.path.exists(savepath):
                os.makedirs(savepath)
        savepath = os.path.join(r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_RAI_labels',os.path.basename(jsonnames[i]))
        save_centroids(ctd_list,savepath)

def select_RPI():
    # json_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/derivatives'
    json_folder = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/val/dataset-verse20validation/dataset-02validation/derivatives'
    jsonnames = glob(os.path.join(json_folder, '*/*.json'))
    print(len(jsonnames))
    for jsonname in sorted(jsonnames):
        with open(jsonname, 'r') as f:
            # load json file
            json_data = json.load(f)
            for dict in json_data:
                if 'direction' in dict:
                    if dict['direction'] != ['L', 'A', 'S']:
                        print(os.path.basename(jsonname),dict['direction'])

# select_RPI()
process_and_move_json()