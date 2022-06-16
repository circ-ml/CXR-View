# CXR-View: Deep learning to estimate view position from chest radiographs


## Overview
CXR-View takes a chest radiograph image as input and outputs probabilities that the radiograph was taken in 1) a posterior-anterior view, 2) anterior-posterior view, or 3) a lateral view

This repo contains data intended to promote reproducible research. It is not for clinical care or commercial use. 

## Installation
This inference code was tested on Ubuntu 18.04.3 LTS, conda version 4.8.0, python 3.7.7, pytorch 1.10.0, CUDA 11.6, fastai 2.5.3 and cadene pretrained models 0.7.4. 

Inference can be run on the GPU or CPU, and should work with ~4GB of GPU or CPU RAM. For GPU inference, a CUDA 11 capable GPU is required.

For the model weights to download, Github's large file service must be downloaded and installed: https://git-lfs.github.com/ 

This example is best run in a conda environment:

```bash
git lfs clone https://github.com/circl-ml/CXR-View
cd location_of_repo
conda env create -n CXR_View -f environment.yml
conda activate CXR_View
python get_views.py dummy_dataset/ models/CXR_View_Predictor output/output.csv --gpu=3
```

The GPU number must correspond to a working GPU on your system. If you do not have a GPU, pass --gpu=None


Dummy image files are provided in `dummy_datasets/test_images/;`. Weights for the CXR-View model are in `models/CXR_View_Predictor.pth`.
CXR-View is a densenet121 architecture 

## Datasets
The CXR-View model was developed using 80% of the PADCHEST, NIH Chest X-ray 14, CheXpert, and MIMIC-CXR databases. It was evaluated in the remaining 20% of these databases.
The model achieved an overall accuracy of 99% with 99% accuracy on PA/AP Views and 99.6% accuracy on Lateral views. 

The `data` folder provides the image filenames and the CXR-View estimates for the testing dataset from PADCHEST, NIH Chest X-Ray 14, CheXpert, and MIMIC-CXR.
Due to data use agreements we are not able to share these datasets, and recommend that interested researchers download these datasets directly from their respective sources.

## Image processing
All datasets were used in their native file formats except PADCHEST. PADCHEST DICOMS were converted to .tif using dcmtk v3.6.1 and then to PNGs with a 
minimum dimension of 512 pixels using ImageMagick.

```bash
for x in *.dcm; do dcmj2pnm -O +ot +G +Wh 2 $x "${x%.dcm}".tif; done;
mogrify -path destination_for_NLST_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
```

## Acknowledgements
I thank Stanford, NIH, Hospital de Universidad San Juan, and physionet.org/MIT for their support and access to these datasets. I would also like to thank the fastai and Pytorch communities as well as the National Academy of Medicine for their support of this work. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


