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

## Reading and Visualization
Use the jupyter notbook 'data_read_vis.ipynb' to read and vizualize the different data formats. 


## Citing the paper
```
You can cite the paper as : 
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