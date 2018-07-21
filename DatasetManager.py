# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:13:59 2017

@author: woojin
"""

# 인덱스는 [0,1) 형식으로 표기, 사


 
import numpy as np
import cv2
import helper as hp
import threading
import queue

class Batch:
    
    # float32
    arr4d_LR = 0;
    arr4d_LR_v = 0; # v channel.
    arr4d_HR = 0;
    arr4d_HR_v = 0;
    arr4d_HR_re = 0; # rehigh, LR->HR by interpolation.
    arr4d_HR_rev = 0;
    
    # uint8
    arr4d_HR_in = 0;
    arr4d_LR_in = 0;
    
    def __init__(self):
        self.arr4d_LR = 0;
        self.arr4d_LRv = 0;
        self.arr4d_HR = 0;
        self.arr4d_HRv = 0;
        self.arr4d_HR_re = 0; # rehigh, LR->HR by interpolation.
        self.arr4d_HR_rev = 0;
        
        self.arr4d_HR_in = 0;
        
    def _1_input(self, arr4d_LR_in, arr4d_HR_in):
#        self.set_gt_in = copy.deepcopy(set_gt)
        self.arr4d_HR_in = arr4d_HR_in;
        self.arr4d_LR_in = arr4d_LR_in;
        
    def _2_prepare(self,bool_augment):   
        (N,H_HR,W_HR,C) = self.arr4d_HR_in.shape;
        (_,H_LR,W_LR,_) = self.arr4d_LR_in.shape;
        
        # mem. alloc. 
        self.arr4d_HR       = np.zeros([N,H_HR,W_HR,3], np.float32  );
        self.arr4d_HR_v     = np.zeros([N,H_HR,W_HR,1], np.float32  );
        self.arr4d_LR       = np.zeros([N,H_LR,W_LR,3], np.float32  );
        self.arr4d_LR_v     = np.zeros([N,H_LR,W_LR,1], np.float32  );
        self.arr4d_HR_re    = np.zeros([N,H_HR,W_HR,3], np.float32  );
        self.arr4d_HR_rev   = np.zeros([N,H_HR,W_HR,1], np.float32  );
#        self.set_rehigh  = np.zeros([N,H,W,3], np.float32  )
#        self.set_rehigh_v= np.zeros([N,H,W,1], np.float32  )
        
        # rooop
        for i in range(N):
            # slice
            HR_uint8 = self.arr4d_HR_in[i,:,:,:];
            LR_uint8 = self.arr4d_LR_in[i,:,:,:];
            LRHR = [LR_uint8,HR_uint8];
            
            # online augmentation
            if( bool_augment ):                
                LRHR = \
                    Batch.augment_random_flip_rot(LRHR);
    #            (low,gt) = Batch.augment_random_hue(low,gt)    # 역효과.
            LR = hp.imgUint82Float(LRHR[0]);
            HR = hp.imgUint82Float(LRHR[1]);
#            if( bool_augment ):
#                gt = Batch.augment_random_brightness(gt, -0.1, 0.1 )
#                gt = Batch.augment_random_contrast( gt,   0.9, 1.1 )
#                gt = Batch.augment_random_saturation(gt, -0.1, 0.1 )
            
            
#            LR = Batch.augment_random_blur(LR,5);
#            LR = Batch.augment_random_gnoise(LR,0.1);
#            hp.Imgshow(LR,'d');
            
            
            # rehigh
            rehigh = cv2.resize( LR, (W_HR,H_HR), 
                                 interpolation=cv2.INTER_CUBIC);
                                
            # v 채널 영상
            LR_v = hp.RGB2Y( LR )
            HR_v = hp.RGB2Y( HR )
            rehigh_v = hp.RGB2Y( rehigh )
            
            # gt gxgy
#            (gt_gx, gt_gy) = Batch.gxgy_image(gt)
#            (gt_gx_v, gt_gy_v) = Batch.gxgy_image(gt_v)
#            gt_gx_v = hp.ForceRank3(gt_gx_v)
#            gt_gy_v = hp.ForceRank3(gt_gy_v)
#            
#            print(np.max(low_v),np.min(low_v))
            

#            hp.Imgshow(LR);
#            hp.Imgshow(HR);
#            time.sleep(100);
            
            # 저장
            self.arr4d_LR     [i,:,:,:] = LR;
            self.arr4d_LR_v   [i,:,:,:] = LR_v;
            self.arr4d_HR     [i,:,:,:] = HR;
            self.arr4d_HR_v   [i,:,:,:] = HR_v;
            self.arr4d_HR_re  [i,:,:,:] = rehigh;
            self.arr4d_HR_rev [i,:,:,:] = rehigh_v;
                       
        return;
                       
    ###########################################################################
    # internal functions
    ###########################################################################
                          
            
    # output은 따로  함수로 만들 필요가 없네.        
#    def _3_output(self):
    
    def gxgy_image(img):
#        img = hp.imgUint82Float(cv2.imread('my_low.png'))
        
        mx = (np.asarray([0.0,0.0,0.0, -1.0,1.0,0.0, 0.0,0.0,0.0],np.float32)).reshape([3,3])
        my = (np.asarray([0.0,-1.0,0.0, 0.0,1.0,0.0, 0.0,0.0,0.0],np.float32)).reshape([3,3])
        
        gx = cv2.filter2D( img, cv2.CV_32F, mx)
        gy = cv2.filter2D( img, cv2.CV_32F, my)
        
#        hp.Imgshow(gx+0.5)
#        hp.Imgshow(gy+0.5)
        
        return gx, gy
    
    
    def downscale_2x(img):
        (h,w,c) = img.shape
        img = cv2.GaussianBlur(img,(7,7),2.25/3.0)
        img = cv2.resize(img,(hp.intfloor(w/2),hp.intfloor(h/2)),interpolation=cv2.INTER_CUBIC)
        return img

    def downscale_x4(img):
        (h,w,c) = img.shape
        h4 = int(h/4)
        w4 = int(w/4)
        img_tmp = img[0:h4*4,0:w4*4,:]
        
        (h,w,c) = img_tmp.shape
        img_tmp = cv2.GaussianBlur(img_tmp,(11,11),4.5/3.0)
        img_tmp = cv2.resize(img_tmp,(int(w/2),int(h/2)),interpolation=cv2.INTER_CUBIC)
        img_tmp = cv2.resize(img_tmp,(int(w/4),int(h/4)),interpolation=cv2.INTER_CUBIC)
        return img_tmp    
    
    def downscale_x4__(img):
        (h,w,c) = img.shape
        h4 = int(h/4)
        w4 = int(w/4)
        img_tmp = img[0:h4*4,0:w4*4,:]
        
        (h,w,c) = img_tmp.shape
        img_tmp = cv2.resize(img_tmp,(int(w/4),int(h/4)),interpolation=cv2.INTER_AREA)
        return img_tmp    
        
    
    def augment_resize(low,gt):
#        low = downscale_2x(low)
#        gt  = downscale_2x(gt)
        return (low,gt)
        
        
    def range_hue_float(img_1ch):
        img = img_1ch
        img[img<0] = img[img<0]+360.0
        img[img>360] = img[img>360]-360.0
        return img

    def augment_random_hue(low,gt):
        low_hsv = cv2.cvtColor(np.float32(low)/255, cv2.COLOR_RGB2HSV)
        gt_hsv  = cv2.cvtColor(np.float32( gt)/255, cv2.COLOR_RGB2HSV)
                            
        rnum = (np.random.rand()-0.5)*360
               
        low_h = low_hsv[:,:,0];
        gt_h = gt_hsv[:,:,0]
        
        low_h = Batch.range_hue_float( low_h + rnum )
        gt_h  = Batch.range_hue_float( gt_h + rnum )
        
        low_hsv[:,:,0] = low_h
        gt_hsv[:,:,0]  = gt_h
        
        low_rgb = np.uint8( cv2.cvtColor(low_hsv, cv2.COLOR_HSV2RGB )*255)
        gt_rgb  = np.uint8( cv2.cvtColor( gt_hsv, cv2.COLOR_HSV2RGB )*255)
        
        return low_rgb, gt_rgb
    
#    def augment_random_rot90(low,gt):
#        rnum = np.random.randint(0,4)
#        if( rnum is 1 ):
#            low = cv2.ro

    def augment_random_flip_rot(set_images):
        
        def __fliplr(aimg):
            aimg  = np.fliplr(aimg)
            return(aimg)
            
        def __flipud(aimg):
            aimg  = np.flipud(aimg)
            return(aimg)
            
        def __transpose(aimg):
            aimg [:,:,0] = np.transpose( aimg[:,:,0])
            aimg [:,:,1] = np.transpose( aimg[:,:,1])
            aimg [:,:,2] = np.transpose( aimg[:,:,2])
            return(aimg)
        
        num_images = len(set_images);
        rnum = np.random.randint(0,8)
        for i in range(num_images):
            img = set_images[i];
            
            if( rnum is 1 ):
                (img) = __fliplr(img)
            if( rnum is 2 ):
                (img) = __flipud(img)
            if( rnum is 3 ):
                (img) = __fliplr(img)
                (img) = __flipud(img)
            if( rnum is 4 ):
                (img) = __transpose(img)
            if( rnum is 5 ):
                (img) = __fliplr(img)
                (img) = __transpose(img)
            if( rnum is 6 ):
                (img) = __flipud(img)
                (img) = __transpose(img)
            if( rnum is 7 ):
                (img) = __fliplr(img)
                (img) = __flipud(img)
                (img) = __transpose(img)
                
            set_images[i] = img;
            
        return set_images;
    
     

    # 밝기, 밝게어둡게 하거나 밝기를 이동시킨다고 보면 되겠지
    def augment_random_brightness(img,lower=-0.1,upper=+0.1):
        aug_brightness = lower + np.random.rand()*(upper-lower)
        img = img + aug_brightness
        return img
        
    
    # 대조비, 평균 명도 주변으로 늘리거나 줄이거나.
    def augment_random_contrast(img,lower=0.9,upper=+1.1):
        aug_contrast = lower + np.random.rand()*(upper-lower)
        
        mmean = np.mean(img)
        img = (img-mmean) * aug_contrast + mmean
        return img      
              
          
    #채도.
    def augment_random_saturation(img,lower=-0.1,upper=+0.1):
        
        HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL )
        [H,S,V] = hp.split_3ch_to_1ch(HSV)
        
        aug_saturation = lower + np.random.rand()*(upper-lower)
        S = S + aug_saturation
        
        HSV = hp.merge_1ch_to_3ch(H,S,V)
        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB_FULL)
        
        return img   
    
    # 랜덤 가우시안 노이즈
    def augment_random_gnoise(img,max_sig=0.03):
        sig = np.random.rand()*max_sig;
        (h,w,c) = img.shape;
        gnoise = np.random.randn(h,w,c).astype(np.float32);
        img = img+gnoise*sig;
        
        img[img<0.0] = 0.0;
        img[img>1.0] = 1.0;
        return img
        
    # 랜덤 광학블러
    def augment_random_blur(img,max_ksize):
        ksize = np.random.rand()*max_ksize;
        ksize = int(ksize);
        ksize = 1+ksize+(ksize%2)
        img = cv2.GaussianBlur(img, (ksize,ksize), 0);
#        hp.Imgshow(img,'ddd');
        return img;
        
    
###############################################################################
#
# 데이터 관리 클래스
# 다중 스레드를 이용하여 데이터 로드와 관리
#
###############################################################################    
class Dataset:
    
    
    list_LR = 0;
    list_HR = 0;
    itable = 0;
    qu_data = 0;
    threads = [];
    flag_thread_off = [];
    height_low = 0;
    width_low = 0;
    sscale = 0;
    size_batch = 0;
    augumentation = False;
    img_duplication = 0;
        
    #############################################
    # (1)
    # 
    # 가장 먼저 불리는 함수, 생성자, 초기화 함수 역할
    #############################################
    def _1_set_list_dataset(self, list_LR, list_HR,
                         height_low, width_low, size_batch, 
                         sscale, augumentation, img_duplication):
        self.list_LR = list_LR;
        self.list_HR = list_HR;
        self.itable = list(range(len(self.list_HR)));
        self.height_low = height_low;
        self.width_low = width_low;
        self.size_batch = size_batch;
        self.sscale = sscale;
        self.augumentation = augumentation;
        self.img_duplication = img_duplication;
        
    #############################################
    # (2)
    # 
    # 두번째로 불려야 되는 함수
    # 쓰레드를 시작함
    #############################################
    def _2_start_load_thread(self):
        
        print('))) start thread');
        self.qu_data = queue.Queue(8);
        
        for i in range(4):
            th = threading.Thread(target=self.run_online_load,
                                  args = [self.qu_data,i]);
            self.threads.append(th);
            self.flag_thread_off.append(False);
            th.start();

       
    #############################################
    # (3)
    # 
    # 마지막으로 불려야 되는 함수
    # 쓰레드를 종료함
    # 전혀 구현되어 있지 않
    #############################################
    def _3_stop_load_thread(self):
        for i in range( len(self.flag_thread_off)):
            self.flag_thread_off[i] = True;
            
        for i in range( len(self.threads) ):
            th = self.threads[i];
            if( []==th):
                skip=1;
            else:
                th.join();
                print('thread finised');
        return;
    
    #############################################
    # (-)
    # 
    # 쓰레드 본체
    # qu_data에 계속 데이터를 쑤셔넣는다.
    # limit에 도달하면 기다린다.
    #############################################
    def run_online_load(self,qu_data,th_num):
#        qu_data = queue.Queue();
        
        # name changes to make simple code
        N = self.size_batch;
        H_LR = self.height_low;
        W_LR = self.width_low;
        S = self.sscale;
        H_HR = H_LR * S;
        W_HR = W_LR * S;
            
        th = threading.current_thread();
        print('))) thread %s ready\n'%(th.name),end='');
                
        flag_thread_on = True;
        while(flag_thread_on):
            
            #print
#            th = threading.current_thread();
#            print('im alive - %s %d\n'%(th.name,self.img_duplication),end='');
            
            # mem alloc.
            arr4d_HR = np.zeros([N,H_HR,W_HR,3],dtype=np.uint8);
            arr4d_LR = np.zeros([N,H_LR,W_LR,3],dtype=np.uint8);
            
            # 배치만큼 데이터 로드
            for i in range(self.size_batch): 
                
                # 같은 이미지를 몇번 재사용, 속도 문제로
                # img_duplication으로 조절 가능.
                if (i%self.img_duplication)==0:   
                    # high resolution image, low resolution image.
                    (HRI,LRI) = self.zzz_load_image_with_check();
                
                # 패치 샘플링
                (H_LRI,W_LRI,_) = LRI.shape;
                ry = np.random.randint(0,H_LRI - H_LR+1);
                rx = np.random.randint(0,W_LRI - W_LR+1);
                croped_LR = LRI [ ry  : ry  +H_LR,   rx  : rx  +W_LR];
                croped_HR = HRI [ ry*S: ry*S+H_HR,   rx*S: rx*S+W_HR];
                arr4d_HR[i,:,:,:] = croped_HR;
                arr4d_LR[i,:,:,:] = croped_LR;
                
            
            # batch 
            a_batch = Batch();
            a_batch._1_input( arr4d_LR, arr4d_HR );
            if( self.augumentation == True):
                a_batch._2_prepare( bool_augment=True); 
            else:
                a_batch._2_prepare( bool_augment=False); 
            
            # 큐 채워 넣기
            # 지금 특별히 중요치는 않기에... 5분 동안 큐가 비워지지 않으면 자동종료
            flag_data_pushing = True;
            try_count = 0;
            while(flag_data_pushing):
                if( try_count >= 60*5 ):
                    flag_thread_on = False;
                    break;
                    
                if( self.flag_thread_off[th_num] ):
                    flag_thread_on = False;
                    break;
                
                try:
                    self.qu_data.put(a_batch,timeout=1);
                    flag_data_pushing = False;
                except:
                    nothing = 0;
                    
                try_count = try_count+1;
                
            
        print('))) thread %s end\n'%(th.name),end='');
        return;
    
    
    

    #############################################
    # (-)
    # 
    # 인터페이스 함수, 데이터 꺼내기.
    #############################################    
    def get_batch(self):
        return self.qu_data.get();

    ###########################################################################
    # internal functions
    ###########################################################################
    def zzz_load_image_with_check(self):
        
        # load image
        while(True):
            ridx = np.random.randint(low=0,
                                     high=len(self.list_HR),
                                     size=1);  
            idx = self.itable[ridx[0]];
            
            # high resolution image, low resolution image.
            HRI  = hp.BGR2RGB(cv2.imread(self.list_HR[idx]));
            LRI  = hp.BGR2RGB(cv2.imread(self.list_LR[idx]));
            
            # 영상이 패치보다 작으면 스킵
            (H_LRI,W_LRI,_) = LRI.shape;
            (H_HRI,W_HRI,_) = HRI.shape;
            if( H_LRI>(self.height_low*self.sscale) and
                W_LRI>(self.width_low *self.sscale) ):
                break;
                
        #
        return (HRI,LRI);


    