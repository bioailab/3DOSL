The details to run the different segmentation benchmarks of is provided here. 

# Get the test data 

We use the real image test set released as part of the work `Physics based machine learning for sub-cellular segmentation in living cells` in the (URL)[https://zenodo.org/records/5017066]. The real image dataset consists of a training and test subset containing 368 and 144 images respectively of size 256x256. These images were obtained using an epifluresence microscope. The ground truth annotations are manual annotations released as part of the work. 
Images with all background pixels in the ground truth segmentations were ignored for evaluation. 


## 1. MitoGraph
Please refer (repository)[https://github.com/vianamp/MitoGraph] for running instructions on MacOS. You may use (Darling)[https://www.darlinghq.org/] if you would like to run on linux machine. We downloaded the MacOS binaries from the (release) [https://github.com/vianamp/MitoGraph/releases/tag/v3.0] and emulated usinf Darling. 

## 2. MitoMeter

For mitometer, we cloned the latest code from the authors at the (commit)[https://github.com/aelefebv/Mitometer/tree/9d5788ed7ca9908e9c136e469e2f1b8903d1cda0] and followed the instructions within their README to install the Matlab app and evaluate on our data.

## 3. Weka

We used the trainable Weka segmentation from Fiji with default paramerer setting. Instructions for usage are available in the (link)[https://imagej.net/plugins/tws/]. Prompts were provided manually to select regions for the background and foreground on one image for training the model.  

## 4. Kanezaki et. al.

We cloned the latest code from the authors at the (commit) [https://github.com/kanezaki/pytorch-unsupervised-segmentation/tree/478e67e46948216de82ea0bc6ba242ad6a79d442] and followed the instructions provided in their README file to run their unsupervised segmentation algorithm on each of the test files. 

## 5. MitoSegNet

We used the binaries os Mitosegnet to fine-tune a segmentation model using the training set of the real dataset. Please refer the instruction in the segmentation tool provided in the (MitoSegnet)[https://github.com/MitoSegNet/MitoS-segmentation-tool] official repository.

## 6. PhySeg 
We use the model released in the (URL)[https://zenodo.org/records/5017066] as part of the work Physics based machine learning for sub-cellular segmentation in living cells. We follow the instructions provided in the PPT for fine-tuning the model on the real images.

## 7. 3DOSL 

We use the model released in the (URL)[https://zenodo.org/records/5017066] as part of the work Physics based machine learning for sub-cellular segmentation in living cells. We follow the instructions provided in the PPT for fine-tuning the model on the real images after pre-training on dataset created using 3DOSL. 

# Evaluation 
The code used for calculating the metrics of Iou, dice score and betti error is available in the file `evaluation.ipynb`. Before running the code, download the Betti-error (repository)[https://github.com/nstucki/Betti-matching/tree/8fc23e64c035916e2548fb70be3ab08c3860c3a6] and place it in the same folder as the evaluation.ipynb file. 