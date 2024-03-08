# This software is Copyright © 2024 The Regents of the University of California. 
# All Rights Reserved. Permission to copy, modify, and distribute this software and 
# its documentation for educational, research and non-profit purposes, without fee, and 
# without a written agreement is hereby granted, provided that the above copyright notice, 
# this paragraph and the following three paragraphs appear in all copies. Permission to 
# make commercial use of this software may be obtained by contacting:
# Office of Innovation and Commercialization
# 9500 Gilman Drive, Mail Code 0910
# University of California
# La Jolla, CA 92093-0910
# innovation@ucsd.edu
# This software program and documentation are copyrighted by The Regents of the University of California. 
# The software program and documentation are supplied “as is”, without any accompanying services 
# from The Regents.The Regents does not warrant that the operation of the program will be 
# uninterrupted or error-free.The end-user understands that the program was developed for 
# research purposes and is advised not to rely exclusively on the program for any reason.
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, 
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, 
# ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF 
# THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT 
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
# THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA 
# HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.


#!/usr/bin/python3

# Author: Vaghef Ghazvinian, mghazvinian@ucsd.edu
# Affiliation: CW3E, Scripps, UCSD


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
import datetime
from datetime import datetime
from utils_ANN import *
from models import *
warnings.filterwarnings('ignore')
import inspect
import itertools
import numpy as np
import pandas as pd
print(tf.__version__)
import multiprocessing




def DL_GeneratePQPF(ilead): 
    print("leadtime = ",ilead)
    nlocs, obs_locs, obs_row, obs_col, obs_lats, obs_lons = get_mask()
    fcst_dates = get_fcst_dates(ilead)
    len_verif = len(fcst_dates)
    predictions = np.zeros((len_verif,nlocs,3),dtype=np.float32)
    print('#################################################')
    print('Start training\n') 
    n_features = predictors.shape[-1] 
    grid_params = {
    'embedding_size': [10,25],
    'embedding_dim': [nlocs],
    'par_reg':[1e-4,1e-3]
    'n_features': [n_features],
    'batch_size': [10000,50000],
    'hidden_nodes': [[10,10]],
    'lr' : [0.05],
    'conv': [True],
    'activation': ['softplus'], 
    'optimizer': ['Adam']
    # NOTE: can add more grid params here.
    # format is "param_name":  values
    # note that each param has to be a named argument in either of build
    # or train functions defined 
    }
    keys, values = zip(*grid_params.items())
    all_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for day_id in tqdm(range(len_verif),total=len_verif):
        print(day_id)
        train_predictors, train_targets  = get_train_data(pd.to_datetime(fcst_dates(day_id).strftime("%Y%m%d"),ilead))
        verif_predictors, verif_targets, loc_train = get_verif_data(pd.to_datetime(fcst_dates(day_id).strftime("%Y%m%d"),ilead))
        validation_predictors, validation_targets = get_validation_data(pd.to_datetime(fcst_dates(day_id).strftime("%Y%m%d"),ilead))
        train_predictors_reshaped = train_predictors.reshape(-1,train_predictors.shape[-1])
        train_targets_reshaped = train_targets.reshape(-1)
        validation_predictors_reshaped = validation_predictors.reshape(-1,validation_predictors.shape[-1])
        validation_targets_reshaped = validation_targets.reshape(-1)
        verif_predictors_reshaped = verif_predictors.reshape(-1,verif_predictors.shape[-1])
        verif_targets_reshaped = verif_targets.reshape(-1)
        loc_train =  np.repeat((np.arange(0,nlocs)[np.newaxis,:]),train_predictors.shape[0],axis=0).reshape(-1,1)
        loc_valid =  np.repeat((np.arange(0,nlocs)[np.newaxis,:]),validation_predictors.shape[0],axis=0).reshape(-1,1)
        loc_verif =  np.repeat((np.arange(0,nlocs)[np.newaxis,:]),1,axis=0).reshape(-1,1)

        train_predictors_all = np.concatenate((train_predictors_reshaped,loc_train),axis=1)
        validation_predictors_all = np.concatenate((validation_predictors_reshaped,loc_valid),axis=1)
        verif_predictors_all = np.concatenate((verif_predictors_reshaped,loc_verif),axis=1)
        tuning_params = []
        val_loss = []
        preds = []   

        for param_permutation in all_permutations:
            params_name = '; '.join(['{0}={1}'.format(k, v) for k,v in param_permutation.items()])
            function_arguments = inspect.getfullargspec(build_csgd_embed_model).args
            arg_overwrite = {k: param_permutation[k] for k in param_permutation if k in function_arguments}
            tf.keras.backend.clear_session()
            np.random.seed(123)
            tf.random.set_seed(123)
            model = build_csgd_embed_model(**arg_overwrite)
            model.summary()
            train_function_arguments = inspect.getfullargspec(train_csgd_embed_model).args
            arg_overwrite = {k: param_permutation[k] for k in param_permutation if k in train_function_arguments}
            history = train_csgd_embed_model(model,train_predictors_all, train_targets, validation_predictors_all, validation_targets,               
                **arg_overwrite
            )
            tuning_params.append(params_name)            
            val_loss.append(min(history["val_loss"]))
            verif_x = {}
            verif_x['location'] = verif_predictors_all[:,-1].astype(np.int32)
            verif_x['numericals'] = np.delete(verif_predictors_all,-1,axis=1).astype(np.float32)
            preds.append(model.predict(verif_x))
        indexx_min_loss = np.nanargmin(val_loss)
        predictions[day_id,:,:] =  preds[indexx_min_loss]    
                   
    outfile_dir = "/home/mghazvinian/cw3e_DL/preds_ann_day"+str(ilead)
    np.savez(outfile_dir,predictions=predictions)
    return




  

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=6)
    pool.map(DL_GeneratePQPF,[1,2,3,4,5,6])



   



