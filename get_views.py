"""Main testing script for the composite outcome experiment. Purpose is to determine whether using composite outcomes improves DL performance for prognosis
<image_dir> - Directory where images to run the model on are located
<model_path> - Absolute or relative file path to the .pth model file (or the prefix excluding the _0 for an ensemble model)
<output_file> - Absolute or relative file path to where the output dataframe should be written

Usage:
  run_model.py <image_dir> <model_path> <output_file> [--gpu=GPU] 
  run_model.py (-h | --help)
Examples:
  run_model.py /path/to/images /path/to/model /path/to/write/output.csv
Options:
  -h --help                    Show this screen.
  --gpu=GPU                    Which GPU to use? [Default:None]
"""


import warnings
warnings.simplefilter(action='ignore')
import sys
#sys.path.insert(0,'../../utils/')
#sys.path.insert(0,'../../segmentation/')
#sys.path.insert(0,'../../autoclassifier/')
#sys.path.insert(0,'../../autoclassifier/fastai2/')


import os
from docopt import docopt
import pandas as pd

import pretrainedmodels
from sklearn.metrics import *
import math
import time


num_workers = 16
if __name__ == '__main__':

    arguments = docopt(__doc__)
  
    ###Grab image directory
    image_dir = arguments['<image_dir>']
    
    #Set model path 
    mdl_path = arguments['<model_path>']
    
    

    bs = 32
    val_bs = 32
    
    if(arguments['--gpu'] is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments['--gpu']
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import fastai
    from fastai.vision.all import *

    ###set model architecture
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f))] 
    ###Results
    output_df = pd.DataFrame(columns = ['File','Dummy','Prediction'])
        
    output_df['File'] = files
    
    #Create dummy variable
    output_df['Dummy'] = np.random.randint(0,2,len(files))


    #Organize dataset and trick fastai into thinking that we are training on Dummy to create imagedataloaders
    col = 'Dummy'
    output_df['valid_col'] = np.repeat(True,output_df.shape[0])
    final_df = output_df.append(output_df.iloc[output_df.shape[0]-1,:],ignore_index=True)
    final_df.valid_col[final_df.shape[0]-1] = False

    final_df = final_df.append(final_df.iloc[final_df.shape[0]-1,:],ignore_index=True)
    final_df.Dummy[final_df.shape[0]-1] = 1 - final_df.Dummy[final_df.shape[0]-1]


    #Categorical variable being predicted
    block = CategoryBlock


    
    #PA, AP, or Lateral
    out_nodes = 3
    
    #Model uses 320 x 320 pixels
    size =  320
        
    #Create dataloaders
    imgs = ImageDataLoaders.from_df(df=final_df,path=image_dir,label_col=col,y_block=block,bs=bs,val_bs=val_bs,valid_col="valid_col",item_tfms=Resize(size),batch_tfms=[Normalize.from_stats(*imagenet_stats)])

    #Model is a densenet121 CNN
    mdl = fastai.vision.models.densenet121

    #Create CNN leaner based on these configs
    learn = cnn_learner(imgs, mdl,n_out=out_nodes)

    #To fix file path formatting
    learn.model_dir = "."

    path = learn.path
    learn.path = Path(".")

    #Load specified model from file
    learn.load(mdl_path)
            
    learn.path = path
    #Generate predictions    
    preds,y = learn.get_preds(ds_idx=1,reorder=False)
        

         
    ###output predictions as column with model name
    views = ['AP','Lateral','PA']
    for yy in range(out_nodes):
        tmp = preds[:,yy]
        output_df['Prediction_' + views[yy]] = np.array(tmp)

    output_df.to_csv(arguments['<output_file>'])
