The details to run the different benchmarks of 2D to 3D reconstruction is provided here. 

# 1. Occupancy Network
Occupancy network takes 3D occupancy representation and 2D images as input while training.  
1. Train the model
To train a new network from scratch, run
```
cd ../occupancy_networks
python train.py configs\img\onet_EMPIAR10791.yaml
```
After training the model, we can generate 3D shapes for a given input image. 
To generate meshes using a trained model, 
```
python generate.py configs\img\onet_EMPIAR10791.yaml
```

# 2. DISPR

1. Prepare the data 
Save the 3D mitochondria shapes of 3D OSL as 64x64x64 sized voxels as tiff files. For this set the `write_resize_voxel` to True and `write_smooth_obj` to False in `roi_to_mesh.py` and run: 
```
python roi_to_mesh.py --root_dir path_to_24nm_stack_labelled.tif 
```
The 3D voxel data is saved into `roi_resized` folder.

Also save the mid slice of these voxel data to the folder `mid_slice`. 

2. Train the model
Please follow the instructions in DISPR/README.md for training and generation.

# 3. Fiji Pipeline

A stack of 3 consecutive images along the z-axis as input for the FiJi pipeline baseline. This pipeline is a combination of simple thresholding-based 3D segmentation followed by marching cubes for reconstruction. All processing is done in (Fiji-ImajeJ)[https://imagej.net/ij/]. So, download and install the ImajeJ software. 

1.  Generate the 3-image stack as input data

```
cd occupancy_networks/scripts
source .sim_venv/bin/activate
bash dataset_mito/simulate_stack.sh

```
Add noise to the simulated images. Replace or set values to `SIMULATE_IMAGES_PATH` and `OUT_PATH` with the path of the simulated images in the previous step and the destination oath to save the noisy images respectively. 
```
python3 add_noise_and_stack.py $SIMULATE_IMAGES_PATH configs/epi2_stack.py --install_folder $OUT_PATH

```
2. Segment and convert to meshes
Use the Ostu thresholding and the File> SaveAs > Wavefront(.obj) options to save the 3D shape. 
Convert these to `.off` files using trimesh. Modify the `occupancy_network/scripts/dataset_mito/build.sh script and run to creat the unit scaled meshes for evaluation. 


# Evaluation
For evaluation of the models, we provide two scripts: `eval.py` and `eval_meshes.py`.

The main evaluation script is `eval_meshes.py`.
You can run it using
```
python eval_meshes.py configs\img\onet_EMPIAR10791.yaml
```
The script takes the meshes generated in the previous step and evaluates them using a standardized protocol.
The output will be written to `.pkl`/`.csv` files in the corresponding generation folder which can be processed using [pandas](https://pandas.pydata.org/).



