
import tensorflow as tf
import numpy as np
import helper as hp
from wwModel_helper import *



def model_SRCNN( X, v_scope='model_SRCNN', reuse=False):
    print(hp.who_am_i());
    autonumber("",reset = True);
    
    with tf.variable_scope(v_scope,reuse=reuse):
        
        z = resize_biliear( X, 4);
        
        z = conv2d( relu( bias( z)), [5,5,64]);
        z = conv2d( relu( bias( z)), [5,5,32]);
        z = conv2d( relu( bias( z)), [9,9,1]);
        
    return z;

def model_FSRCNN( X, v_scope='model_FSRCNN', reuse=False):
    print(hp.who_am_i());
    autonumber("",reset = True);
    
    with tf.variable_scope(v_scope,reuse=reuse):
        
        z = conv2d( X, [5,5,64]);
        
        z = conv2d( relu(bias(z)), [1,1,32]);
        
        z = conv2d( relu(bias(z)), [3,3,32]);
        z = conv2d( relu(bias(z)), [3,3,32]);
        z = conv2d( relu(bias(z)), [3,3,32]);
        z = conv2d( relu(bias(z)), [3,3,32]);
        
        z = conv2d( relu(bias(z)), [1,1,64]);
        
        z = conv2d_t( relu( bias(z)), [9,9,1], 4);
        
    return z;

def model_DRRN( X, RS, v_scope='model_DRRN', reuse=False):
    print(hp.who_am_i());
    
    autonumber("",reset=True);
    
    with tf.variable_scope(v_scope,reuse=reuse):
        
        z = resize_biliear( X, 4);
        GRP = z; # Global Residual Pass
        
        with tf.variable_scope('res_block',reuse=reuse):
            z = conv2d( relu(bias( z)), [5,5,128]);
            LRP = z; # Local Residual Pass
            
            with tf.variable_scope('res_unit',reuse=reuse):
                autonumber("",reset=True);
                z = conv2d( relu(bias( z)), [3,3,128]);
                z = conv2d( relu(bias( z)), [3,3,128]);
                z = z+LRP*RS;
            for i in range(8):
                with tf.variable_scope('res_unit',reuse=True):    
                    autonumber("",reset=True);
                    z = conv2d( relu(bias( z)), [3,3,128]);
                    z = conv2d( relu(bias( z)), [3,3,128]);
                    z = z+LRP*RS;
                
        z = conv2d( relu(bias( z)), [3,3,1]);
        z = z+GRP*RS;
        
    return z;