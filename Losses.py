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

#Author: Vaghef Ghazvinian, mghazvinian@ucsd.edu
#Affiliation: CW3E, Scripps, UCSD



import os
import sys
import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras.backend as k
import tensorflow as tf
import math
import numpy as np










def crps_cens(y_true, y_pred):  
    """
    Author: Vaghef Ghazvinian: mghazvinian@ucsd.edu
    see: Ghazvinian et al. 2021;2022;2024
    This function sets the loss function: crps for censored shifted gamma distribution.
    see the closed form expression in Scheuerer and Hamill 2015
    use small offset values as in Scheuerer and Hamill 2015
    for shift, mu and sigma for better convergance
    return: mean value of crps over batch 
    """                
    shift2 = y_pred[:, 0]
    shift = -shift2 - 0.00049
    mu = y_pred[:, 1]+ 0.0005
    sigma = y_pred[:, 2] + 0.0182
    obs=y_true[:, 0]
    shape = k.square(mu / sigma)
    scale = (k.square(sigma)) / mu
    rate = 1 / scale
    c_bar = (-1 * shift) / scale
    y_bar = (obs - shift)/scale
    F_k_c = tf.math.igamma(shape, 1. * c_bar)
    F_kp1_c = tf.math.igamma(shape+1., 1. * c_bar)
    F_2k_2c = tf.math.igamma(2. * shape, 1. * 2. *c_bar)
    B_05_kp05 = tf.vectorized_map(lambda x: k.exp(tf.math.lbeta([0.5,x])), shape+0.5)
    F_k_y = tf.math.igamma(shape, 1. * y_bar) 
    F_kp1_y = tf.math.igamma(shape+1., 1. * y_bar)
    c1=y_bar*(2.*F_k_y-1.)
    c2=shape*(2.*F_kp1_y-1.+k.square(F_k_c)-2.*F_kp1_c*F_k_c)
    c3=c_bar*k.square(F_k_c)
    c4=(shape/float(math.pi))*B_05_kp05*(1.-F_2k_2c)
    crps=c1-c2-c3-c4        
    CRPS = crps * scale
    return k.mean(CRPS)        






 