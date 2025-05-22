# 3DOSL Dataset Repository

This repository accompanies the 3DOSL dataset. The steps to ge the data from source and create 3D OSL is explainde here. 

## Get the data
3DOSL is created from the EM segmentation available from [EMPIAR-10791-High resolution 3D imaging of liver subcellular architecture and its link to metabolic function](https://www.ebi.ac.uk/empiar/EMPIAR-10791/). Download the mitochondria segmentations of 4.1857 Obese Climp63 Liver Dataset - FIB-SEM and Segmentation. 


## Data Preprocessing 
0. Set up the environment: A seperate environement for the pre-processing (Steps 1-3 below) and for creation of different forms is suggested (in Write pointclouds and occupancy formats and Image simulation ). 
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
Run the file `roi_to_mesh.py` to save meshes of the individual instance. 
```
python roi_to_mesh.py --root_dir path_to_24nm_stack_labelled.tif --structuring_element ball3.5 
```

##  Write pointclouds and occupancy formats 
The code for creating the pointcloud and occupancy formats are borrowed from Occupancy NetWork. 
0. Set up the environment:

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `mesh_funcspace` using
```
cd occupancy_networks
conda env create -f environment.yaml
conda activate mesh_funcspace
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

To compile the dmc extension, you have to have a cuda enabled device set up.
If you experience any errors, you can simply comment out the `dmc_*` dependencies in `setup.py`.
You should then also comment out the `dmc` imports in `im2mesh/config.py`.
* build our modified version of [mesh-fusion](https://github.com/davidstutz/mesh-fusion) by following the instructions in the `external/mesh-fusion` folder


1. You are now ready to build the dataset:
```
cd occupancy_networks/scripts
bash dataset_mito/build.sh
``` 

This command will build the dataset in `data/mito.build`. Also generates the emitter locations for the simulator now! 

3. To install the dataset, run
```
bash occupancy_networks/dataset_mito/install.sh
```
If everything worked out, this will copy the dataset into `data/mito.install`.


## Simulate Images
0. Set up the environement. 
The simulation code is adapted from (Sekh.et.al)[https://doi.org/10.5281/zenodo.5017066]. An environment with matlab runtime engine is required for this. The code uses python's matlab interface for generating noise. 
```
python3 -m venv .sim_venv
source .sim_venv/bin/activate
python3 -m pip install -r simulation/sim_en_requirements.txt
```
1. Simulate images from saved emitters
```
cd occupancy_networks/scripts
bash dataset_mito/simulate.sh
```
2. Add noise and/or create montages of images (for segmentation)
Add noise to the simulated images. Replace or set values to `SIMULATE_IMAGES_PATH` and `OUT_PATH` with the path of the simulated images in the previous step and the destination oath to save the noisy images respectively. 
```
python3 add_noise_and_stack.py $SIMULATE_IMAGES_PATH configs/epi2.py --install_folder $OUT_PATH
```

## Related Citations
Source dataset

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

Occupancy Networks

    @inproceedings{Occupancy Networks,
        title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
        author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
        booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2019}
    }

Point Spread Function and Image Simulation

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

    @article{Sekh2021,
        author = {Sekh, Arif Ahmed and Opstad, Ida S. and Godtliebsen, Gustav and Birgisdottir, {\AA}sa Birna and Ahluwalia, Balpreet Singh and Agarwal, Krishna and Prasad, Dilip K.},
        doi = {10.1038/s42256-021-00420-0},
        issn = {2522-5839},
        journal = {Nature Machine Intelligence},
        keywords = {Cellular imaging,Computer science,Microscopy,Mitochondria,Multivesicular bodies},
        month = {dec},
        number = {12},
        pages = {1071--1080},
        publisher = {Nature Publishing Group},
        title = {{Physics-based machine learning for subcellular segmentation in living cells}},
        url = {https://www.nature.com/articles/s42256-021-00420-0},
        volume = {3},
        year = {2021}
    }




