'''
 # @ Author: Abhinanda Ranjit Punnakkal
 # @ Create Time: 2022-06-07 10:44:22
 # @ Modified by: Abhinanda Ranjit Punnakkal
 # @ Modified time: 2022-08-04 17:06:16
 # @ Description:
 '''

from ast import Store, parse
import cc3d
from nbformat import write
import numpy as np
import ipdb
import pandas as pd
import glob
from scipy import ndimage as ndi
# from dask_image.ndmeasure import label as dlabel
import argparse
import re
# import tifffile as tif

from tifffile import imread, imwrite
from skimage.morphology import ball, cube
from utils import write_pickle, mkdir_if_not_existing, does_not_exist
from utils import str2slices, resize_voxel 


def pre_process_dataset(labels_in_path, dno=0, is_labelled=False, 
            start_at = 0, sel=None):
    if not is_labelled:
        labels_in = imread(labels_in_path).astype('uint8')
    else:
        labels_in == imread(labels_in_path)
    if sel != None:
        stype = re.split('\d+', sel)[0]
        rad = float(re.findall('\d+', sel)[0])
        sel = globals()[stype](rad)
    ipdb.set_trace()
    connectivity = 6
    labels_filtered = cc3d.dust(labels_in, threshold=2000, connectivity=connectivity, in_place =False)

    label_erode = ndi.binary_erosion(labels_filtered, structure=sel).astype('uint')
    labels_filtered = cc3d.dust(labels_in, threshold=2000, connectivity=connectivity, in_place =False)

    labels_out, num_mito = cc3d.connected_components(labels_filtered, 
        connectivity=connectivity, out_dtype=np.uint32, return_N=True)
    stats = cc3d.statistics(labels_out)
    
    centroids = stats['centroids']
    bounding_boxes = stats['bounding_boxes']
    voxel_counts = stats['voxel_counts']
    print('Numer of ccs in data', len( centroids))

    df = pd.DataFrame([tuple(l) for l in centroids], columns=['cx', 'cy', 'cz'])
    df['voxel_counts'] = pd.Series(voxel_counts)
    df = df.join(pd.DataFrame(bounding_boxes, columns=['bbx', 'bby', 'bbz']))
    df.to_csv( labels_in_path.replace('.tif', str(rad)+'b_stats.csv'))

    imwrite(labels_in_path.replace('.tif',  str(rad)+'b_labelled.tif'), labels_out)
    return num_mito


def main():
    # root_dir="../../FIB-SEMdata/EMPIAR-10982/*/*/*mito*.tif"
    ipdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type =str, default="../../../FIB-SEMdata/EMPIAR-10791/1857_Obese_Climp63_ER/1857_Obese_Climp63_ER_24*.tif")
    parser.add_argument( "--dno", type=str, default='0')
    parser.add_argument("--is_labelled", action="store_true", default=False)
    parser.add_argument("--structuring_element", default = None)
    args = parser.parse_args()
    print(args)
    root_dir=str(args.root_dir).replace("'", '"')
    dno = args.dno
    num_processed  = 0
    # print(prob)
    datasets =  glob.glob(root_dir)
    print("Datasets", datasets)
    
    for dataset in datasets[::]:
        base_name =  dataset.split('\\')[-1]
        print("*"*50)
        print('Processing dataset ', dataset)
        num_processed = pre_process_dataset(dataset, dno, is_labelled=args.is_labelled, 
            start_at = num_processed, sel=args.structuring_element)

        
    print('Completed ')

if __name__ == "__main__":
    main()

