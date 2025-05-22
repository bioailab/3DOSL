# 3DOSL Dataset Repository

This repository accompanies the 3DOSL dataset.

## Unpacking the dataset
3DOSL is published in DataverseNo(link[https://doi.org/10.18710/JX6JXF]). 
Download the dataset arracnged into 10 zip files into the 'data_zip' folder.  

To unpack the dataset, run the following command:

```
python src/data_unpack.py
```
This will unpack individual instances in the 'data' folder. This step will take a while to complete. If you would liek to get a sneak-peak of the data, we suggest you keep only 1 zip file of the 10 in the full dataset, in the 'data_zip' folder. 

### Data creation from EM segmentation
Follow the instructions in 'dataset_creation/dataset.md' to generate the different forms of instances available in 3DOSL. 

## Reading and Visualization
Use the jupyter notbook 'data_read_vis.ipynb' to read and vizualize the different data formats. 

## Benchmarks
Follow the instructions in 'benchmarks/reconstruction.md' and 'benchmarks/segmentation.md' for setting up and evaluating different benchmarks used for 2d to 3D reconstruction and segmentation of microscope images of mitochondria using 3DOSL. 

## Citation
You can cite the dataset as : 
```
@data{JX6JXF_2024,
author = {Punnakkal, Abhinanda Ranjit and Jadhav, Suyog S and Celeste, Aaron Vaughn and Horsch, Alexander and Agarwal, Krishna and Prasad, Dilip K},
publisher = {DataverseNO},
title = {{3DOSL - 3D Organelle Shape Library for Optical Microscopy}},
year = {2024},
version = {V1},
doi = {10.18710/JX6JXF},
url = {https://doi.org/10.18710/JX6JXF}
}
```