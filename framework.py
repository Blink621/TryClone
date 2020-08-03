#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:51:26 2020

@author: zhouming
"""

class EncodingModel():
    pass

class DecodingModel():
    """
    """
    def __init__(self):
        pass
        
    def set_task(self, task):
        """
        Only 'classfication', 'identification', 'Reconstruction' are supported
        """

class DNNEM(EncodingModel):
    """
    Encoding model combining deep neural networks
    """
    def __init__(self, hrf, grid, loss, drop_out):
        """
        Parameters:
        ----------
        hrf : str
            HRF type
        grid : array
            range of the x, y and sigma
        loss : str
            Loss type.
        drop_out : float
            Value of frop out
        cv : int
            Number of cross validation 
        """
        pass
    
    def fit(bold, feat, design, fmask):
        """
        ???
        
        Parameters:
        ----------
        bold : BrainMap
            Brain map object
        feat : Feature
            Feature object(DNN feature or feature generator)
        design : Design
            Design object. (Design matrix and stimulus image array)
        fmask : ?
            ???
        """
        pass
    
    def make_spatial_kernel(feat):
        """
        Generate spatial masks of equal physical size on each feature channel
        
        Parameters:
        ----------
        feat : Feature
            Feature object(DNN feature or feature generator)
        """
        pass
    
    def make_feature_vector():
        pass
        
    def make_model(self):
        pass

    def fit_model(self):
        pass
    
    
class DNNDM(DecodingModel):
    pass

class NaiveEM(EncodingModel):
    pass

class BrainMap:
    pass

class Feature:
    pass

class Design:
    pass

class NaiveDM(DecodingModel):
    pass    