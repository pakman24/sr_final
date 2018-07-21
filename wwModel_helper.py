import tensorflow as tf;
import numpy as np;
        

def autonumber(name,readonly=False,reset=False):
    if( reset == True):
        del autonumber.name_count;
        autonumber.name_count = {'empty':0};
        return 'OK';
    
    if( False == readonly ):
        if( name in autonumber.name_count ):
            autonumber.name_count[name] += 1;
        else:
            autonumber.name_count[name] = 0;
        
    return autonumber.name_count[name];
autonumber.name_count = {'empty':0};

def autonamenumber(name,readonly=False):
    return name+'_'+str(autonumber(name,readonly));

def concat(xs,_axis=3):
    return tf.concat(xs, axis=_axis)
    
def bias(x, name='bias'):
    size_out = x.shape[3].value   
    name = autonamenumber(name);
    
    B = tf.get_variable( name=name,
                         shape = [size_out],
                         initializer = tf.random_normal_initializer(0.0, 0.002));
                        
#    B = tf.Variable( tf.constant( 0, dtype=tf.float32,
#                                 shape=[size_out] ),
#                     name=name);
    return x+B

#def conv2d(x,shape_filter,shape_stride=[1,1,1,1],padding='SAME',stddev=0.001):
#    W = tf.Variable( tf.random_normal( shape_filter, mean=0.0, stddev=stddev ))
#    return tf.nn.conv2d( x, W, shape_stride, padding)
def conv2d(x, shape, stride=1, atrous=1, name='conv2d', padding='same', reuse=False):
    if( len(shape)==2 ):
        shape.append(0);
    if( shape[2]==0):
        shape[2] = x.shape[3].value;
#    if( name=='conv2d' ):
#        name = name + '_' + str(autonumber(name));
    name = autonamenumber(name);
   
    print('[ww] conv2d: k:(%d,%d,%d,%d), s:(%d), atrous:(%d), name:(%s)'%
          (shape[0],shape[1],x.shape[3],shape[2],stride,atrous,name));     
        
    return tf.layers.conv2d( x,
                            shape[2],
                            (shape[0],shape[1]),
                            strides = (stride,stride),
                            padding = padding,
                            dilation_rate=(atrous,atrous),
                            use_bias = False,
                            kernel_initializer = 
                                tf.random_normal_initializer(0.0,0.02),
                            name=name,
                            reuse=reuse);
                            
                            
def conv2d_t(x, shape, stride=2, name='conv2d_t'):
    if( len(shape)==2 ):
        shape.append(0);
    if( shape[2]==0 ):
        shape[2] = x.shape[3].value;
    name = autonamenumber(name);
       
    print('[ww] conv2d_t: k:(%d,%d,%d,%d), s:(%d), name:(%s)'%
          (shape[0],shape[1],x.shape[3],shape[2],stride,name));    
          
    return tf.layers.conv2d_transpose( 
            x,
            shape[2],
            (shape[0],shape[1]),
            strides=(stride,stride),
            padding = 'same',
            use_bias = False,
            kernel_initializer=
                tf.random_normal_initializer(0.0,0.002),
            name=name);
        

#def conv2d_t(x,shape_filter,shape_output,shape_stride=[1,2,2,1],padding='SAME',stddev=0.001):
#    [H,W,C,N] = shape_filter 
#    weight = tf.Variable( tf.random_normal(shape_filter, stddev=stddev, dtype=np.float32))
##            weight = tf.Variable( tf.random_normal(shape_filter, stddev=0.031, dtype=np.float32))
#    return tf.nn.conv2d_transpose(x,weight,tf.stack(shape_output),shape_stride, 
#                                  padding=padding)
    
def conv2d_tf(x,shape_kernel,atrous_rate=1):
    if( shape_kernel[2]==0):
        shape_kernel[2] = x.shape[3].value;
    return tf.layers.conv2d(x,
                     shape_kernel[2],
                     (shape_kernel[0],shape_kernel[1]),
                     strides = (1,1),
                     padding = 'same',
                     dilation_rate=(atrous_rate,atrous_rate),
                     use_bias = False,
                     kernel_initializer = tf.random_normal_initializer(
                             mean=0.0, stddev= 0.1)
## 꼭 필요한지 의문   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001)
                     )
def conv2d_t_tf(x,shape_kernel,stride=2):
    return tf.layers.conv2d_transpose(x,
                     shape_kernel[2],
                     (shape_kernel[0],shape_kernel[1]),
                     strides=(stride,stride),
                     padding = 'same',
                     use_bias = False,
                     kernel_initializer = tf.random_normal_initializer(
                             mean=0.0, stddev = 0.001)
                     )
                     
def async_conv2d_tf(x,shape_kernel):
    [H,W,C] = shape_kernel
    x2 =   conv2d_tf( x,  [H,1,C])
    return conv2d_tf( x2, [1,W,C])

def relu(x,name='relu'):
    name = autonamenumber(name);
    
    return tf.nn.relu(x,name=name);

def lrelu(x,alpha=0.001):
    return tf.maximum( x, x*alpha) 
                 
def prelu( x):
#            alpha = tf.Variable( tf.random_normal(  tf.shape(x)[3], mean=0.1, stddev=0.0))
    alpha = simple_weight( [x.shape[3].value], mean=0.1, stddev = 0.001 )
    return tf.maximum( 0.0, x ) + alpha * tf.minimum( 0.0, x )

def crelu(x):
    return concat([tf.maximum(  x, 0.0),
                   tf.maximum( -x, 0.0)]);

def max_pool(x,ksize=3,stride=2,name='max_pool'):
    ksize = [1,ksize,ksize,1];
    stride = [1,stride,stride,1];
    name = autonamenumber(name);
    
    return tf.nn.max_pool( x, 
                          ksize, 
                          strides=stride, 
                          padding = 'SAME', 
                          name=name);
#    return tf.nn.max_pool( x, ksize=[1,3,3,1],
#                           strides=[1,stride,stride,1],
#                           padding = 'SAME');
    
def avg_pool(x,ksize=3,stride=2,name='avg_pool'):
    ksize = [1,ksize,ksize,1];
    stride = [1,stride,stride,1];
    name = autonamenumber(name);
    
    return tf.nn.avg_pool(x,
                          ksize,
                          stride,
                          padding='SAME',
                          name=name);

def weighted_diff(y,gt):
    we = tf.abs(gt)
    we = tf.greater(we,1)
    we = tf.cast(we,tf.float32)
    return tf.multiply((gt-y),we)

def l2_loss(x):
    return tf.reduce_mean( x**2 )

def l1_loss(x):
    return tf.reduce_mean( tf.abs( x ))


def simple_weight(shape_weight,mean=0.0,stddev=0.00001):
#            return tf.Variable( tf.random_normal(shape_weight,stddev=stddev, dtype=np.float32))
    return tf.Variable( tf.random_normal(shape_weight,mean=mean,stddev=stddev, dtype=np.float32))

def resize_biliear( x, scale, name='bilinear'):
    name = autonamenumber(name);
    
    HW = tf.shape(x)[1:3];
        
    return tf.image.resize_bilinear( x, HW*scale, name=name);
               
         
            

if( __name__ == "__main__" ):
    
    if( not 'sess' in locals() ):
        sess = tf.InteractiveSession();
    
    # 레이어 테스트를 위해서...
    X = np.zeros([5,5,1,1]);
    X = tf.constant(X);
    
    
    Z = conv2d(X,[3,3,0]);
    print(sess.run([Z]));
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    