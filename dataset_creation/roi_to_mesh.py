'''
 # @ Author: Abhinanda Ranjit Punnakkal
 # @ Create Time: 2022-06-07 10:44:22
 # @ Modified by: Abhinanda Ranjit Punnakkal
 # @ Modified time: 2022-08-04 17:06:16
 # @ Description:
 '''

from enum import unique
import cc3d
import numpy as np
import ipdb
import pandas as pd
import glob
import os
from scipy import ndimage as ndi
import argparse
import re

from tifffile import imread, imwrite
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.morphology import ball, cube
import mcubes
from utils import write_pickle, mkdir_if_not_existing, does_not_exist
from utils import str2slices, resize_voxel

def write_files(extracted_image, i ,
            #  bb,
            dataset_root, dno,
            write_tiff = False, 
            write_mid_slice = False,
            write_objs = False, 
            write_smooth_obj = True, 
            write_voxels = False, 
            write_resize_voxel = False, 
            write_resize_mesh = False, 
            iso_value = 0.5,             
            smoothing_value = 0., 
            ):
    
    # crop = extracted_image[bb]
    if write_tiff and does_not_exist(dataset_root+ '/voxels/mito_'+str(i)+'.tif') :
        imwrite( dataset_root+ '/voxels/'+str(i)+'.tif', extracted_image)   
    
    if write_mid_slice and does_not_exist(dataset_root+ '/mid_slice/mito_'+str(i)+'.tif') :
        imwrite( dataset_root+ '/mid_slice/'+str(i)+'.tif', extracted_image[32,:,:])

    if write_objs and does_not_exist( dataset_root+ '/objects/'+str(i)+'_render.obj') :
        vertices, triangles = mcubes.marching_cubes(extracted_image, iso_value)
        mcubes.export_obj(vertices, triangles,  dataset_root+ '/objects/'+str(i)+'_render.obj')

    if write_smooth_obj  and does_not_exist (dataset_root+ '/objects/'+str(i)+'_smooth.off'):
        # ipdb.set_trace()
        smoothed_crop = mcubes.smooth(extracted_image)
        vertices, triangles = mcubes.marching_cubes(smoothed_crop, smoothing_value)
        mcubes.export_off(vertices, triangles,  dataset_root+ '/'+dno+ '_'+str(i)+'.off')
        # mcubes.export_off(vertices, triangles,  dataset_root+ '/objects_test_3_5/'+dno+ '_'+str(i)+'.off')

    if write_voxels and does_not_exist ( dataset_root+ '/rois/'+str(i)+'.pkl'):
        write_pickle(extracted_image,  dataset_root+ '/rois/'+str(i)+'.pkl')

    if write_resize_voxel and does_not_exist(dataset_root+ '/roi_resized/'+str(i)+'.pkl') :
        resized = resize_voxel(extracted_image, 64)
        write_pickle(resized,  dataset_root+ '/voxels_64/'+str(i)+'.pkl')

    if write_resize_mesh and does_not_exist(dataset_root+ '/roi_resized/'+str(i)+'_resized.obj') :
        vertices, triangles = mcubes.marching_cubes(resized, iso_value)
        mcubes.export_obj(vertices, triangles, dataset_root+ '/roi_resized/'+str(i)+'_resized.obj')

def process_roi(labels_in_path = '../../../FIB-SEMdata/EMPIAR-10982/1_c_elegans/c_elegans/c_elegans_mito.tif', 
                dno=None, 
                is_labelled = False, 
                start_at = 0,
                sel = None,
                ):
    dataset_root = os.path.dirname(labels_in_path)  
    mkdir_if_not_existing(dataset_root) 

    if not is_labelled:
        print('Dataset not labelled, labelling now...')
        labels_in = imread(labels_in_path).astype('uint8')
        connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out = cc3d.dust(labels_in, threshold=2000, connectivity=connectivity, in_place=False)

       
        labels_out, num_mito = cc3d.connected_components(labels_out, 
            connectivity=connectivity, out_dtype=np.uint32, return_N=True)
        stats = cc3d.statistics(labels_out)
        
        centroids = stats['centroids']
        bounding_boxes = stats['bounding_boxes']
        voxel_counts = stats['voxel_counts']
        print('Numer of ccs in data', len( centroids))
        del labels_in
    else:
        print('Dataset is labelled, leading ...')
        labels_in = imread(labels_in_path)
        unique_labels = list(np.unique( labels_in) )

    if does_not_exist(labels_in_path.replace('.*','_stats.csv')):
        print('Writing stats')
        if not is_labelled: ipdb.set_trace()
        df = pd.DataFrame([tuple(l) for l in centroids], columns=['cx', 'cy', 'cz'])
        df['voxel_counts'] = pd.Series(voxel_counts)
        df = df.join(pd.DataFrame(bounding_boxes, columns=['bbx', 'bby', 'bbz']))
        df.to_csv( labels_in_path.replace('.tif','_stats.csv'))
        del df
    else: 
        df = pd.read_csv( labels_in_path.replace('_labelled.tif','_stats.csv'))
        assert len(unique_labels) == len(df)
        print('Reading stats', len(df))
        if sel != None:
            stype = re.split('\d+', sel)[0]
            rad = 3.5
            sel = globals()[stype](rad)


    write_tiff = False
    write_mid_slice = False
    write_objs = False
    write_smooth_obj = True
    write_voxels = False
    write_resize_voxel = False
    write_resize_mesh = False

    mkdir_if_not_existing(os.path.join(dataset_root, 'objects_test_3_5'))
    if (write_voxels) : mkdir_if_not_existing(os.path.join(dataset_root, 'rois'))
    if ( write_resize_voxel or  write_resize_mesh) : mkdir_if_not_existing(os.path.join(dataset_root, 'roi_resized'))

    discard = 0
    ipdb.set_trace()
    if is_labelled:
        for i, row in df.iterrows():
            if i == 0: continue    
            bb = str2slices(row['bbx']), str2slices(row['bby']), str2slices(row['bbz'] )
            # extracted_image = ((labels_in * (labels_in == i))/i).astype('uint16')

            instance = labels_in[bb] == (i)
            target_shape = 64
            instance_sub = instance[ ::2, :, : ] 
            x_dim, y_dim, z_dim = instance_sub.shape
            if (np.max(instance.shape)>target_shape):
                discard += 1
                continue
            block = (( 25,25), (25, 25), (25,25))           # for MiShape V1

            x_pad = target_shape - x_dim
            y_pad = target_shape - y_dim
            z_pad = target_shape - z_dim 
            block = (( 2,2), (2, 2), (2,2))

             # Subsample z axis by 2
            extracted_image = np.pad(instance_sub, 
                                    ( (x_pad//2, x_pad//2 + x_pad%2), 
                                    (y_pad//2, y_pad//2 + y_pad%2), (z_pad//2, z_pad//2 + z_pad%2) )
                                    , 'constant', constant_values=0 )
           
            extracted_image = ndi.binary_dilation(extracted_image, sel).astype('uint')
            bodys = ndi.find_objects(extracted_image)

            bodys = [o for o in bodys if i is not None]
            assert len(bodys) == 1, print('i', i)
            write_files(extracted_image, i, dataset_root, dno, write_tiff, write_mid_slice, write_objs, 
            write_smooth_obj, write_voxels, write_resize_voxel, write_resize_mesh , iso_value=0., smoothing_value = 0.5 )
                     
        return len(unique_labels)
    else:
        for i in range(1, num_mito+1): 

            extracted_image = (labels_out * (labels_out == i))
            write_files(extracted_image, i, dataset_root, dno, write_tiff, write_objs, 
            # write_files(extracted_image, i, bounding_boxes[i], dataset_root, dno, write_tiff, write_objs, 
            write_smooth_obj, write_voxels, write_resize_voxel, write_resize_mesh , iso_value=0., smoothing_value = 0.5 )

        return num_mito

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type =str, default="../../../FIB-SEMdata/EMPIAR-10791/1857_Obese_Climp63_Mitochondria/1857_24nm3_5b_labelled*.tif")
    parser.add_argument( "--dno", type=str, default='0')
    parser.add_argument("--is_labelled", action="store_true", default=False, help='Set true if the dataset is already labelled')
    parser.add_argument("--structuring_element", default = None, help="E.g. ball3.5")
    args = parser.parse_args()
    print(args)
    root_dir=str(args.root_dir).replace("'", '"')
    dno = args.dno

    datasets =  glob.glob(root_dir)
    print("Datasets", datasets)

    for dataset in datasets[::]:
        print("*"*50)
        print('Processing dataset ', dataset)
        num_processed = process_roi(dataset, dno, is_labelled=args.is_labelled, 
            start_at = num_processed, sel=args.structuring_element)
        
    print('Completed  ' )

if __name__ == "__main__":
    main()

