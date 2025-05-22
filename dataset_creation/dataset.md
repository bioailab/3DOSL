# 3DOSL Dataset Repository

This repository accompanies the 3DOSL dataset. The steps to ge the data from source and create 3D OSL is explainde here. 

## Get the data
3DOSL is created from the EM segmentation available from [EMPIAR-10791-High resolution 3D imaging of liver subcellular architecture and its link to metabolic function](https://www.ebi.ac.uk/empiar/EMPIAR-10791/). Download the mitochondria segmentations of 4.1857 Obese Climp63 Liver Dataset - FIB-SEM and Segmentation. 


## Data Preprocessing 
0. Set up the environment: A seperate environement for the pre-processing (Steps 1-3 below) and for creation of different forms is suggested (in step xxx ). 
```
python3 -m venv .shape_venv
source .shape_venv/bin/activate
python3 -m pip install -r requirements.txt
```

1. Downsample: To extract the individual instances of mitochondria, the connected components(CC) algorithm ((cc3d)[https://github.com/seung-lab/connected-components-3d]) is applied. To reduce computational complexity for CC, we downsample the data from 8nm to 24 nm using (Imagej)[https://imagej.net/ij/]. You can use other tools like python+dask to do this. Save the sownsampled stack.  


2. Label and pre-process:  
Run code to save the labelled stack. The data is saved with a prefix "b_labelled.tif". It also saves some stats of the individual mitochondria instance. 
```
python pre_process_label.py --root_dir path_to_24nm_stack.tif
```
3.  Extract the individual components
Run the file 'roi_to_mesh.py' to save meshes of the individual instance. 
```
python roi_to_mesh.py --root_dir path_to_24nm_stack_labelled.tif --structuring_element ball3.5 
```

##  Write pointclouds and occupancy formats 
The code for creating the pointcloud and occupancy formats are borrowed from Occupancy NetWork
0. Set up the environment:




## Simulate Images



## Related Citations

@article{Parlakgul2022,
author = {Parlakg{\"{u}}l, G{\"{u}}neş and Arruda, Ana Paula and Pang, Song and Cagampan, Erika and Min, Nina and G{\"{u}}ney, Ekin and Lee, Grace Yankun and Inouye, Karen and Hess, Harald F. and Xu, C. Shan and Hotamışlıgil, G{\"{o}}khan S.},
doi = {10.1038/s41586-022-04488-5},
issn = {1476-4687},
journal = {Nature},
keywords = {3,D reconstruction,Endocrine system and metabolic diseases,Homeostasis,Organelles},
month = {mar},
number = {7902},
pages = {736--742},
pmid = {35264794},
publisher = {Nature Publishing Group},
title = {{Regulation of liver subcellular architecture controls metabolic homeostasis}},
url = {https://www.nature.com/articles/s41586-022-04488-5},
volume = {603},
year = {2022}
}

@inproceedings{Occupancy Networks,
    title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
    author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}


@article{li2017fast,
  title={Fast and accurate three-dimensional point spread function computation for fluorescence microscopy},
  author={Li, Jizhou and Xue, Feng and Blu, Thierry},
  journal={JOSA A},
  volume={34},
  number={6},
  pages={1029--1034},
  year={2017},
  publisher={Optica Publishing Group}
}




