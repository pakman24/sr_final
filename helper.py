import cv2 
import numpy as np
import matplotlib.pyplot as plt
import traceback
import time

def tic():
    tic.prev_time = time.time();
    return tic.prev_time; 
tic.prev_time = time.time();

def toc(prev_time = 0):
    if prev_time==0:
        prev_time = tic.prev_time;
    return time.time()-prev_time;
    
def toc_print(prev_time=0):
    if prev_time == 0:
        prev_time = tic.prev_time;
    print('%.3f초입니다.'%(time.time()-prev_time));

def who_am_i():
   stack = traceback.extract_stack()
   filename, codeline, funcName, text = stack[-2]

   return funcName

def Range(mmin,array,mmax):
    array[array<mmin] = mmin
    array[array>mmax] = mmax
    return array

def RotateImage(img,degree=360,scale=1.0):
    (h,w,c) = img.shape
    ch = h/2
    cw = w/2
    rot_mat = cv2.getRotationMatrix2D( (cw,ch), degree, scale)
    return cv2.warpAffine( img, rot_mat, dsize=(w,h), borderMode = cv2.BORDER_CONSTANT )

def shave(img,in_pixel):
    sub = img[in_pixel:-in_pixel,in_pixel:-in_pixel,:]
    return sub

#서울대 평가함수
#imdff = imdff(:);
#rmse = sqrt(mean(imdff.^2));
#psnr = 20*log10(255/rmse);
def PSNR(imgA, imgB, max_value=255.0):
    mse = np.mean( (imgA-imgB)**2 )
    if( mse == 0 ):
        return max_value
    return 20 * np.log10( max_value / np.sqrt(mse) )
#    return 20*np.log10(max_value)-10*np.log10(mse)



def PadImage(img,psize,mode='padding'):
    if(mode=='padding'):
        return cv2.copyMakeBorder(img,psize,psize,psize,psize,borderType=cv2.BORDER_REPLICATE)
    elif(mode=='remove'):
        (h,w,c) = img.shape
        return img[psize:h-psize,psize:w-psize,:]
    
    
def intfloor(number):
    return int(np.floor(number))

def downscale_2x(img):
    (h,w,c) = img.shape
    img = cv2.GaussianBlur(img,(7,7),2.25/3.0)
    img = cv2.resize(img,(int(w/2),int(h/2)),interpolation=cv2.INTER_CUBIC)
    return img
    
def downscale_x4(img):
    (h,w,c) = img.shape
    h4 = intfloor(h/4)
    w4 = intfloor(w/4)
    img_tmp = img[0:h4*4,0:w4*4,:]
    
    (h,w,c) = img_tmp.shape
    img_tmp = cv2.GaussianBlur(img_tmp,(11,11),4.5/3.0)
    img_tmp = cv2.resize(img_tmp,(intfloor(w/2),intfloor(h/2)),interpolation=cv2.INTER_CUBIC)
    img_tmp = cv2.resize(img_tmp,(intfloor(w/4),intfloor(h/4)),interpolation=cv2.INTER_CUBIC)
    return img_tmp

def Imgshow(img_in,title='image'):
    print(img_in.ndim)
    if( img_in.ndim is 2 ):
        img = np.repeat( np.expand_dims(img_in,2), repeats=3, axis=2);
    elif( (img_in.ndim is 3) and (img_in.shape[2] is 1) ):
        print('d')
        img = np.repeat( img_in, repeats=3, axis=2);
    elif( (img_in.ndim is 3) and (img_in.shape[2] is 3) ):
        img = img_in
                   
    print('dd',img.shape)
    plt.imshow(img)
    plt.show();
    plt.pause(0.01);
    print(title);
         
def imgUint82Float(img):
    return np.float32(img)/256

def imgFloat2Uint8(img):
    return np.uint8(range__(0,img*256,255))
    
def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def RGB2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#YCbCr<->RGB
#Y = (299*R + 587*G + 114*B)/1000
#Cb = 0.5643*(B - Y) + 128
#Cr = 0.7132*(R - Y) + 128
#R = (1000*Y + 1402*(Cr-128))/1000
#G = (1000*Y - 714*(Cr-128) - 334(Cb-128))/1000
#B = (1000*Y + 1772*(Cb-128))/1000
#출처: http://darkpgmr.tistory.com/66 [다크 프로그래머]
def merge_1ch_to_3ch(A,B,C):
    A = ForceRank3(A)
    B = ForceRank3(B)
    C = ForceRank3(C)
    return np.concatenate((A,B,C),axis=2)

def split_3ch_to_1ch(ABC):
    A = ForceRank3(ABC[:,:,0])
    B = ForceRank3(ABC[:,:,1])
    C = ForceRank3(ABC[:,:,2])
    return (A,B,C)

def range__(mmin,A,mmax):
    A[A>mmax] = mmax
    A[A<mmin] = mmin
    return A

def RGB2YCbCr(img_uint8_or_float32):
    img_in = img_uint8_or_float32
    if( img_uint8_or_float32.dtype == np.uint8 ):
        img_in = imgUint82Float(img_in)
    R = img_in[:,:,0]
    G = img_in[:,:,1]
    B = img_in[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = 0.5643*(B-Y)+0.5
    Cr = 0.7132*(R-Y)+0.5
    YCbCr = merge_1ch_to_3ch(Y,Cb,Cr)
    
    if( img_uint8_or_float32.dtype == np.uint8 ):
        return imgFloat2Uint8(YCbCr)
    elif( img_uint8_or_float32.dtype == np.float32 ):
        return YCbCr

def YCbCr2RGB(img_uint8_or_float32):
    img_in = img_uint8_or_float32
    if( img_uint8_or_float32.dtype == np.uint8 ):
        img_in = imgUint82Float(img_in)
    Y = img_in[:,:,0]
    Cb= img_in[:,:,1]
    Cr= img_in[:,:,2]
    R = Y + 1.402*(Cr-0.5)
    G = Y - 0.714*(Cr-0.5) - 0.334*(Cb-0.5)
    B = Y + 1.772*(Cb-0.5)
    RGB = merge_1ch_to_3ch(R,G,B)
    
    if( img_uint8_or_float32.dtype == np.uint8 ):
        return imgFloat2Uint8(RGB)
    elif( img_uint8_or_float32.dtype == np.float32 ):
        return RGB

def RGB2Y(img_uint8_or_float32):
    YCbCr = RGB2YCbCr(img_uint8_or_float32)
    Y = ForceRank3(YCbCr[:,:,0])
    return Y

def chageY(RGB,Y):
    YCbCr = RGB2YCbCr(RGB)
    (Y_,Cb,Cr) = split_3ch_to_1ch(YCbCr)
    YCbCr      = merge_1ch_to_3ch(Y,Cb,Cr)
    out = YCbCr2RGB(YCbCr)
    return out
         

def ForceRank3(A):
    if( A.ndim!= 3 ):
        return np.expand_dims(A,2)
    else:
        return A

def AugumentNoise(imgs,aug_factor,sig_noise=6,mmin=0,mmax=255):
    [H,W,C] = imgs[0].shape
    imgs_new = []
    for i, aimg in enumerate(imgs):
        for j in range(aug_factor):    
            aaimg = aimg + (np.random.rand(H,W,C)-0.5)*sig_noise*2
            imgs_new.append(aaimg)
    
    imgs_new = np.minimum( np.maximum(mmin,imgs_new),mmax )
    imgs_new = np.uint8(imgs_new)
    return imgs_new

def AugumentCopy(imgs,aug_factor):
    imgs_new = []
    for i, aimg in enumerate(imgs):
        for j in range(aug_factor):
            imgs_new.append(aimg)
    return imgs_new

def AugumentRotate(imgs,labels,rot_range=[-10,0,10]):
    [H,W,C] = imgs[0].shape
    imgs_new = []
    labs_new = []
    
    for i, aimg in enumerate(imgs):
        alab = labels[i]
        for j,rot_ang in enumerate(rot_range):
#            cv2.getRotationMatrix2D(center angle, scale)
#            ang = (np.random.rand()-0.5)*10
                       
            M = cv2.getRotationMatrix2D((W/2,H/2),rot_ang,1)
            aaimg = cv2.warpAffine(aimg, M, dsize=(W,H), borderMode=cv2.BORDER_REFLECT)
            aalab = cv2.warpAffine(alab, M, dsize=(W,H), borderMode=cv2.BORDER_REFLECT)
            imgs_new.append(aaimg)
            labs_new.append(ForceRank3(aalab))
            
    return (imgs_new, labs_new)

def AugumentRandomSample(imgs,labs,size_samples,dsize_H,dsize_W):
    # dsize : (H,W)
    
    [H,W,C] = imgs[0].shape
    imgs_new = []
    labs_new = []
    
    for i, aimg in enumerate(imgs):
        alab = labs[i]
        for j in range(size_samples):
            H2 = H-dsize_H
            W2 = W-dsize_W
            randy = np.random.randint(0,H2)
            randx = np.random.randint(0,W2)
            subimg = aimg[randy:randy+dsize_H, randx:randx+dsize_W, :]
            sublab = alab[randy:randy+dsize_H, randx:randx+dsize_W, :]
            
            imgs_new.append(subimg)
            labs_new.append(sublab)
            
    return (imgs_new,labs_new)

def AugumentCenterCrop(imgs,dsize_H,dsize_W):
    [H,W,C] = imgs[0].shape
    cx = int(np.floor(W/2))
    cy = int(np.floor(H/2))
    dHhf = int(np.floor(dsize_H/2))
    dWhf = int(np.floor(dsize_W/2))
    imgs_new = []
    
    for i, aimg in enumerate(imgs):
        aaimg = aimg[ cy-dHhf:cy+dHhf, cx-dWhf:cx+dHhf,:]
        imgs_new.append(aaimg)
        
    return (imgs_new)

    
if( __name__ == "__main__" ):
    
    tic();
    time.sleep(1);
    toc_print();
    
    
    