# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#==============================================================================
# conda env tensorflow35
# git pakman24 sr_final
# sr 다른 방법 다 따라하기
#=================================================


#TODO
# () 로더 스레드, 시작전에 확인해서 닫기
# (해결)모델이름과 실험이름을 다르게 해야된다.
# 학습에 사용되는 파라메터를 모아서 한번에 저장, 불러오는 기능
# (해결)텐서보드 활용
# (다시)진짜 그림을 복원해 보는 코드 필요
# (다시)중간저장, 불러오기
# (다시)PSNR확인 시스템. 특정 위치의.


#%%
# =============================================================================
# import
# =============================================================================

import copy;
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import glob
import helper as hp
from helper import tic, toc, toc_print
import wwModel
from wwModel_gen import *
import DatasetManager



np.random.seed(0)
dset = DatasetManager.Dataset() # 데이터 로드 전용 모듈








#%%
###############################################################################
## 훈련 기본값 세팅

    # 기본 세팅
class TRAIN_SETTING:
    name = "DEFAULT_NAME" # 결과저장시 사용되는 이름

    gpu_mem_ratio = 0.9;    # GPU 메모리 사용량 정의
    max_epoch = 1000;         # 학습 종료 에폭

    # SR 세팅
    sr_scale = 4       # 몇배 확대하는건가?

    # 입력영상
    size_batch = 128;  # 미니배치의 크기
    height_batch = 64; # 훈련영상의 높이
    width_batch =  64; # 훈련영상의 폭

    # 에폭세팅
    num_batch_in_a_epoch = int(2048/TRAIN_SETTING.size_batch)
                                        # 1 epoch는 몇개 batch로 구성되는가.

    # 학습률
    learning_rate = 0.001;    # 학습률

    # 이름 간단하게
TSET = TRAIN_SETTING();

## 훈련 기본값 세팅
###############################################################################






#%%
###############################################################################
# 데이터 경로
class DIR:
    root_db = 'G:/sr_image'
    image_HR = root_db + '/DIV2K_train_HR';                # 학습데이터-정답
    image_LR = root_db + '/DIV2K_train_LR_bicubic/X4';     # 학습데이터-입력

# 데이터 경로
###############################################################################



#%%
###############################################################################
# 텐서보드 로그 디랙터리 설정
from datetime import datetime

# 매번 새로운 로그 디렉터리를 사용
now = datetime.utcnow().strftime("%Y%m%d-%H%M%S");
root_logdir = "tf_logs";
logdir = "{}/run-{}-{}/".format(root_logdir,TSET.name,now);

# 매번 같은 로그 디렉터리를 사용
#logdir = "tf_logs/";

# 텐서보드 로그 디랙터리 설정
###############################################################################





#%% 
    

##############################################################################
# 데이터로드

#if(  사) 살아있는 스레드가 있다면 닫고.

if( True or not 'flag_load_n_restruct' in locals() ):
    
    print('load files list')
    query_HR = os.path.normcase( os.path.join(root_HR, '*.*'))
    query_LR = os.path.normcase( os.path.join(root_LR, '*.*'))
        
    list_HR = glob.glob(query_HR);
    list_LR = glob.glob(query_LR);
    print('complete:load files list');
    
    # 몇장 로드해야 되는가?
    num_img = len(list_HR);
    
    # list save
    dset._1_set_list_dataset(list_LR,list_HR,
                          TSET.height_batch,TSET.width_batch,exp,TSET.size_batch,
                          TSET.sr_scale,4);
    dset._2_start_load_thread();
    
    flag_load_n_restruct = True;
    
# 데이터로드
##############################################################################



#%% 
###############################################################################
## 모델 설정
print('## reset default graph() ')
tf.reset_default_graph()    

# 입출력
in_low      = tf.placeholder( tf.float32, [None,None,None, 3], name='in_low');
in_lowv     = tf.placeholder( tf.float32, [None,None,None, 1], name='in_lowv');
gt_high     = tf.placeholder( tf.float32, [None,None,None, 3], name='gt_high');
gt_highv    = tf.placeholder( tf.float32, [None,None,None, 1], name='gt_highv');
op_rs       = tf.placeholder( tf.float32, [], name='rs_factor');

# 오토엔코더
ef_gt_highv = model_AE_encoder(gt_highv, op_rs, "model_AE_encoder", reuse=False);
    # encoded featrue of ground truth
out_highv_ae = model_AE_decoder(ef_gt_highv, op_rs, "model_AE_decoder", reuse=False);

# 초해상도 모델
out_highv = model_sr_2times( in_lowv, op_rs, "model_sr", reuse=False);
ef_out_highv = model_AE_encoder( out_highv, op_rs, "model_AE_encoder", reuse=True);

# 로스 - 오토엔코더
loss_ae = tf.reduce_mean( tf.abs(out_highv_ae - gt_highv));

# 로스 - 초해상도
loss_sr_1 = tf.reduce_mean( tf.abs(gt_highv-out_highv));
loss_sr_ae = tf.reduce_mean( tf.abs( ef_gt_highv - ef_out_highv ));    
#loss_sr = loss_sr_1 + loss_sr_2 + loss_sr_ae;
#    loss_sr = loss_sr_ae;
loss_sr=loss_sr_1;
#    loss_sr=loss_sr_1+loss_sr_ae;

# mean psnr
mpsnr_ = tf.reduce_mean( tf.image.psnr( gt_highv, out_highv, 1.0));
    
## 모델 설정
###############################################################################

#%% 
###############################################################################
## 훈련 준비                

# 트레이닝 변수 묶기
t_vars = tf.trainable_variables();
ae_vars =           [var for var in t_vars if 'model_AE_encoder' in var.name];
ae_vars = ae_vars + [var for var in t_vars if 'model_AE_decoder' in var.name]
sr_vars = [var for var in t_vars if 'model_sr' in var.name];

# 트레이닝 스탭
train_AE = tf.train.AdamOptimizer(0.001).minimize(loss_ae,var_list=ae_vars);
#train_AE = tf.train.GradientDescentOptimizer(0.001).minimize(loss_ae,var_list=ae_vars);
train_sr = tf.train.AdamOptimizer(0.001).minimize(loss_sr,var_list=sr_vars);

# 텐서 보드 준비
tf.summary.scalar('loss_ae',loss_ae);
tf.summary.scalar('loss_sr',loss_sr);
tf.summary.scalar('loss_sr_ae',loss_sr_ae);
tf.summary.scalar('mean_psnr',mpsnr_);
tf.summary.image('low_v', in_lowv,3);
tf.summary.image('gt_highv',gt_highv,3);
tf.summary.image('out_highv',out_highv,3);
tf.summary.image('out_highv_ae',out_highv_ae,3);

merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logdir);

# GPU메모리 사용량 제한
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = exp.gpumem_ratio;

# 세션 시작
if( 'sess' in locals() ): #이메 세션이 있다면 닫음
    sess.close();
    del sess;
sess = tf.InteractiveSession(config=config)
saver = tf.train.Saver(max_to_keep=99999999) #원래는 0넣으면 무제한이어야 되는데 버그가 있어서 큰수 삽입

                      
# 이어하기 기능
history_trnloss=[]
history_vldloss=[]
history_psnr=[]
if(0==exp.continue_start_epoch):
    # 처음부터 시작
    sess.run(tf.global_variables_initializer())
else:
    # 이어하기
    ss = './checkpoint/model-%s-%d'%(exp.train_name,exp.continue_start_epoch)
#    ss = './checkpoint/model-%s-best-%d'%(train_name,continue_start_epoch)
    saver.restore(sess,ss)

## 훈련 준비
###########################################################################
#%% 


###########################################################################
# 에폭 시작
print("start training");

epoch = 0
init_time = time.time()
mmax_mpsnr = 0;
for epoch in range(exp.continue_start_epoch,exp.max_epoch):
    
#    print('[%d]'%(epoch),end='')
    
    start_time = time.time()
    
    
    ##########
    # 모델 저장
    ##########
    if( epoch%10==0 and not epoch==0): 
        save_path = saver.save(sess, "./checkpoint/model-%s"%(exp.train_name),global_step=epoch)
        print("Model saved in file: %s" % save_path)
    
               
        
    ##########
    # 레지듀얼 스케일 조절
    ##########
    rs_factor = 1.0;
    if( epoch < 100 ):
        rs_factor = epoch/100;
        
    
    ##########
    # 트레이닝
    ##########    
    train_time = tic();
    gpu_time = 0;
    batch_time = 0;
    loss= 0.0
    num_batch = exp.num_batch_in_a_epoch
    batch_trn = [];
        
    for i in range(num_batch):
                  
        batch_st = time.time()
        
        # 배치 가져오기
        batch_trn = dset.get_batch();
        
        st = tic();
        # 학습 - 오토엔코더
#            [summary, _] = sess.run([ merged,train_AE],
#                feed_dict={ in_lowv      : batch_trn.arr4d_LR_v,
        [_] = sess.run([ train_AE],
            feed_dict={ in_lowv      : batch_trn.arr4d_LR_v,
                        gt_highv     : batch_trn.arr4d_HR_v,
                        op_rs        : rs_factor });
        
        # 학습 - generator
        [summary, _] = sess.run([merged, train_sr],
            feed_dict={ in_lowv      : batch_trn.arr4d_LR_v,
                        gt_highv     : batch_trn.arr4d_HR_v,
                        op_rs        : rs_factor} );

        gpu_time += toc(st);
        
        batch_time += time.time()
        
        # 배치 학습상황 출력
        print('\r batch: %4d/%d'%(i,num_batch),end='');
        
        
    print('  \r',end='')
    print('  \r',end='')
    train_time = toc(train_time);
    fps = (exp.num_batch_in_a_epoch*exp.size_batch)/(train_time);

    ########
    # 상황출력
    ########
    print('epoch:%5d, fps:%7.2f'%(epoch,fps));
    
                
        
    #########
    # 텐서보드
    #########
    summary_writer.add_summary(summary, epoch)

summary_writer.close();
