#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:00:25 2020

@author: gongzhengxin  zhouming
"""
import numpy as np
from dnnbrain.dnn.core import Mask
from load_data import load_stimuli, load_voxels
from dnnbrain.dnn.models import AlexNet
import pandas as pd
from dnnbrain.dnn.base import MultivariatePredictionModel

class Prepare_Dataset:
    
    def __init__(self,dnn):
        '''
        parameters
        ----------
        dnn[DNN]
        '''
        self.dnn = dnn
    
    def gen_feature(self,stimuli, layer=None, chn=None, mask=None):
        """
        Generate stimuli features
        
        Parameters
        ----------
        dnn[DNN]
        stimuli[ndarray]: shape(n_stim, n_chn, height, width)
        layer[list]: its elements are different layer names
        chn[str|list]: channels that all the layers share
            if chn is str, it must be 'all' which means all channels
            if chn is list, its elements are serial numbers of channels
            default is 'all'
        mask[dict]: storing the layer and chn info 
            layer is the key name and chn is the value
            Their requirements are the same as above
            Note: if using mask, please do not using 'layer' and 'chn' parameters
            
        Return:
        ---------
        features[list]: length equal to layer num
            each element is a activation ndarray for corresponding layer
        fmap_num[int]: num of feature map
        """
        # initialize some params
        fmap_num = 0
        features = []
        dmask = Mask()
        
        # start computing
        if mask is not None:   
            if any([layer, chn]):
                raise AssertionError('Do not define layer and chn if you use mask!')
            else:
                for layer_name in mask.keys():
                    dmask.set(layer_name, channels=mask[layer_name])
                    act = self.dnn.compute_activation(stimuli, dmask).get(layer_name)
                    # handle problems of too many feature maps
                    if act.shape[1] > 1024:
                        varr = np.var(act, axis=0).squeeze()
                        index = np.argsort(-varr)[:1024]
                        act = act[:,index,:,:]
                    fmap_num += act.shape[1]
                    features.append(act)
                    dmask.clear()
        else:
            if layer is None:
                raise AssertionError('layer is required')
            else:    
                for layer_name in layer:
                    dmask.set(layer_name, channels=chn)
                    act = dnn.compute_activation(stimuli, dmask).get(layer_name)
                    # handle problems of too many feature maps
                    if act.shape[1] > 1024:
                        varr = np.var(act, axis=0).squeeze()
                        index = np.argsort(-varr)[:1024]
                        act = act[:,index,:,:]
                    fmap_num += act.shape[1]
                    features.append(act)
                    dmask.clear()
        return features, fmap_num
    
    def gen_rf(self,features,x,y,sigma):
        """
        parameters:
        -----------
        features: list, store of featrue maps from DNN or else
        x,y: float, location of rf center
        sigma: float, rf size
        
        return:
        -----------
        rfs: list, rf masks of each map in features  
        """
        # initilize receptve feilds list
        rfs = []
        
        # generate rf for each feature map
        for i, f in enumerate(features):
            
            # make FC layers can be operate in the same way
            if len(f.shape)==2:
                f = f.reshape(f.shape+(1,1))
                
            # read the basic shape information
            n_sample = f.shape[0] # num of data sample
            n_dimen = f.shape[1] # num of feature demension
            fmap_size = f.shape[2] # resolution of feature map
            
            # generate gaussian mask 
            rf = self.gauss(x,y,fmap_size,sigma) # single mask
            rf = np.tile(rf,(n_dimen*n_sample,1)).reshape(n_sample,n_dimen,fmap_size,fmap_size) # rf tensor
            
            # construct  rf list
            rfs += [rf,]
        
        return rfs
    
    def gauss(self,xi,yi,fmap_size, sigma):
        """
        parameters
        ----------
        xi,yi: float, center location of rf
        fmap_size: int, size of feature map
        sigma: float, gaussian radius 
        
        return:
        ----------
        mask: ndarray, gaussian mask 
        """
        # initialize mask
        mask = np.zeros((fmap_size, fmap_size))
        # localize center
        center = fmap_size//2
        
        # check sigma if negetive use a default para depend on resolution
        if sigma<=0:
            sigma = ((fmap_size-1)*0.5-1)*0.3+0.8 # opCV default
        
        # compute the mask
        s = sigma**2 # var
        coef = 1/(2*np.pi*s) # coefficient
        sum_val =  0 # initialize sum for normalize
        # denote the mask matrix
        for i in range(fmap_size):
            for j in range(fmap_size):
                x, y = center - i, j - center # to formulize direction
                mask[i, j] = coef*np.exp(-((x-xi)**2+(y-yi)**2)/2*s)
                sum_val += mask[i, j] 
        # normalize
        mask = mask/sum_val
        
        return mask


#demo 
# dnn info    
dnn = AlexNet()
layer = ['fc1']#, 'conv3', 'conv4', 'conv5']
chn = 'all'
mask = {'conv1':'all', 'fc2':'all'}
file_path = '/nfs/e3/natural_vision/vim1/data_set_vim1/'



# get stim data
stimuli_lowrez, stimuli_hirez, trn_size = load_stimuli(file_path, npx=224, npc=3)
data_size = len(stimuli_hirez)
val_size = data_size - trn_size
trn_stim_data = stimuli_hirez[:trn_size]
val_stim_data = stimuli_hirez[trn_size:]
#plt.imshow(stimuli_hirez[0].transpose(1,2,0))

#generate features
BE = Prepare_Dataset(dnn)
features, fmap_num = BE.gen_feature(stimuli_hirez, mask=mask)

#get voxel data
subject = 'S1'
roi_names = ['other', 'V1', 'V2', 'V3', 'V3a', 'V3b', 'V4', 'LO']
voxel_data, voxel_roi, voxel_idx = load_voxels(file_path, subject, voxel_subset=range(3400, 3401))
nv = voxel_data.shape[1]
trn_voxel_data = voxel_data[:trn_size]
val_voxel_data = voxel_data[trn_size:]

scores = []
#rf info
x = [-9,-1,1,9]#,1]
y = [-9,-1,1,9]#,1]
sigma = [0.7,1.2]#,8]
#generate fe-stim data
for x_item in x:
    for y_item in y:
        for sigma_item in sigma:
            rfs = BE.gen_rf(features, x_item, y_item, sigma_item)
            #merge rf and feature            
            data_all = np.zeros((features[0].shape[0], 1))
            num = len(rfs)
            for idx in range(num):
                data = np.multiply(rfs[idx], features[idx])
                data = np.sum(data, axis=(2,3)).squeeze()
                data_all = np.concatenate((data_all, data), axis=1)
            data_all = np.delete(data_all, 0, axis=1)

            # start encoding
            #bren = BrainEncoder(voxel_data, 'mul')
            #encode_info = bren.encode_dnn(data_all)
            mpm = MultivariatePredictionModel('glm')
            mpm.set(cv=2)
            mpm.set_scoring('correlation')
            encode_info = mpm.predict(data_all[:trn_size], voxel_data[:trn_size])
            scores += [np.append(np.mean(encode_info['score']),np.array([x_item,y_item,sigma_item]))] 

# sort the rf parameter according to corelation 
df=pd.DataFrame(scores).sort_values(by=0,axis=0,ascending=False)
# initialzie model dict
Models=dict()
# fetch the best model
score,X,Y,Sigma = df.iloc[0,0],df.iloc[0,1],df.iloc[0,2],df.iloc[0,3]
rfs = BE.gen_rf(features, X, Y, Sigma)
# generate dataset for best model training
data_all = np.zeros((features[0].shape[0], 1))
num = len(rfs)
for idx in range(num):
    data = np.multiply(rfs[idx], features[idx])
    data = np.sum(data, axis=(2,3)).squeeze()
    data_all = np.concatenate((data_all, data), axis=1)
data_all = np.delete(data_all, 0, axis=1)
# Regress model
encode_info = mpm.predict(data_all[:trn_size], voxel_data[:trn_size])
# get model parameter and evaluation by test set
encoder = encode_info['model']
paras = encoder[0].coef_.T
pred = data_all[trn_size:].dot(paras)
corr = np.corrcoef(pred,voxel_data[trn_size:].reshape(len(pred)))[0,1]
Models['m1'] = {'where':[X,Y,Sigma],
                  'what':paras,
                  'CV_score':score,
                  'Test_score':corr}



















