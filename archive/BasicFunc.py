# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:38:14 2018

@author: xuzhiqin
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
#isShowPic=1
Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

# >>> used function
def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=1,isShowPic=0):
    if isax==1:
        pltm.legend(fontsize=18)
        # plt.title(y_name,fontsize=14)
#        ax.set_xlabel('step',fontsize=18)
#        ax.set_ylabel('loss',fontsize=18)
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic:
        pltm.show() 
    else:
        pltm.close()
        
def plot_w(w_tmp,range_val=[0,0],bin_num=50,fntmp='wdis',w_tmp0=[],iseps=0,isGauss=1):
    len_w=len(w_tmp)
    row=int(np.sqrt(len_w))
    col=int(len_w/row)
    w_dis=[]
    
    plt.figure()
    for i_l in range(len_w):
        w_dis_i=[]
        val_vec=np.reshape(w_tmp[i_l],[-1])
        if range_val[0]==range_val[1]:
            range_val0=[np.max(val_vec),np.min(val_vec)]
        else:
            range_val0=range_val
        plt.subplot(row,col,i_l+1)
        ax=plt.gca()
        aa_hist=np.histogram(val_vec,bins=50,range=(range_val0[0],range_val0[1]))
        xx_hist=aa_hist[1][0:-1]
        yy_hist=aa_hist[0]/np.max(aa_hist[0])
        plt.plot(xx_hist,yy_hist,'b',label='DNN')
        w_dis_i.append(yy_hist)
        if isGauss:
            ind_hist=yy_hist>np.max(yy_hist)/4
            pcoe=np.polyfit(xx_hist[ind_hist],np.log(yy_hist[ind_hist]),deg=2)
            hist_fit=np.exp(pcoe[0]*xx_hist**2+pcoe[1]*xx_hist+pcoe[2])
            plt.plot(xx_hist,hist_fit,'r--',label='Gauss')
            w_dis_i.append(hist_fit)
        
        if len(w_tmp0)==len(w_tmp):
            val_vec0=np.reshape(w_tmp0[i_l],[-1])
            aa_hist0=np.histogram(val_vec0,bins=50,range=(range_val0[0],range_val0[1]))
            plt.plot(xx_hist,aa_hist0[0]/np.max(aa_hist0[0]),'g--',label='ini')
            w_dis_i.append(aa_hist0[0]/np.max(aa_hist0[0]))
        ax.set_ylim([-0.2,1.1])
        
        ax.set_yticks([])
#                ax.set_yscale('log')
        if i_l<len_w-1:
            plt.axis('off') 
        if i_l==0:
#            plt.title('epoch=%s'%(i))
            plt.legend(ncol=3,loc=3) 
        w_dis.append(w_dis_i)
    mySaveFig(plt, fntmp,iseps=iseps,isax=0)   
    return xx_hist,w_dis
    

        
def getWeightNorm(WW):
    combWW=[]
    for tmp in WW:
        combWW=np.concatenate([combWW,np.reshape(tmp,[-1])])
    mean_W=np.mean(np.abs(combWW))
    std_W =np.std(np.abs(combWW))
    return mean_W,std_W


        
def getWeightNormLayer(WW): 
    mean_W=[]
    std_W=[]
    for tmp in WW:
        mean_W.append(np.mean(np.abs(tmp)))
        std_W.append(np.std(np.abs(tmp)))
    return mean_W,std_W


def getWeightNormL2(WW):
    combWW=[]
    for tmp in WW:
        combWW=np.concatenate([combWW,np.reshape(tmp,[-1])])
    L2_W=np.sqrt(np.sum(np.square(combWW)))
    return L2_W


def getWeightSpecNorm(WW):
    s=0 
    Spec=[]
    for tmp in WW:
        if np.ndim(tmp)>1:
            u, s, vh=np.linalg.svd(tmp,full_matrices=True)
            Spec.append(np.max(np.abs(s)))
        else:
            Spec.append(np.linalg.norm(tmp,2))
    return Spec

# >>> Used function >>>
#############################################
def SelectPeakIndex(FFT_Data, endpoint=True):
    D1 = FFT_Data[1:-1]-FFT_Data[0:-2]
    D2 = FFT_Data[1:-1]-FFT_Data[2:]
    D3 = np.logical_and(D1>0,D2>0)
    tmp=np.where(D3==True)
    sel_ind=tmp[0]+1
    if endpoint:
        if FFT_Data[0]-FFT_Data[1]>0:
            sel_ind=np.concatenate([[0],sel_ind])
        if FFT_Data[-1]-FFT_Data[-2]>0:
            Last_ind=len(FFT_Data)-1
#             print(Last_ind)
            sel_ind=np.concatenate([sel_ind,[Last_ind]])
    return sel_ind

def GetFreq(x_range,x_size):
    
    Fs=x_size/x_range
#     Freq_len=int(x_size/2+1)
    Freq=np.linspace(0,Fs,num=x_size)
    return Freq




def unique_entropy(data):
    b=np.unique(data,axis=0,return_index=True, return_inverse=True, return_counts=True)
    prob_b = b[3]/np.sum(b[3])
    I = - np.dot(prob_b, np.log2(prob_b))
    return b, prob_b, I
def Cond_entropy_YT(b_res, Y_bin_true):
    I_Y_Cond_T_i=np.zeros([len(b_res[0])],dtype=np.float32)
    prob_i_res = b_res[3]/np.sum(b_res[3])
    for i_res in range(len(b_res[0])):
        ind = np.where(b_res[2]==i_res)
        Sel_Y = Y_bin_true[ind]
        b, prob_b, tmpI=unique_entropy(Sel_Y)
        I_Y_Cond_T_i[i_res] =  tmpI * prob_i_res[i_res]
        
    I_Y_Cond_T = np.sum(I_Y_Cond_T_i)
    return I_Y_Cond_T

# >>> used function
def my_fft(data,freq_len=40,x_input=np.zeros(10),kk=0,min_f=0,max_f=np.pi/3,isnorm=1):
    second_diff_input=np.mean(np.diff(np.diff(np.squeeze(x_input))))
    if abs(second_diff_input)<1e-10 :
        datat=np.squeeze(data)
        datat_fft = np.fft.fft(datat) 
        ind2=range(freq_len)
        fft_coe=datat_fft[ind2]
        if isnorm==1:
            return_fft=np.absolute(fft_coe) 
        else:
            return_fft=fft_coe
    else:
        return_fft=get_ft_multi(x_input,data,kk=kk,freq_len=freq_len,min_f=min_f,max_f=max_f,isnorm=isnorm)
    return return_fft

#NU DFT
def get_ft_multi(x_input,data,kk=0,freq_len=100,min_f=0,max_f=np.pi/3,isnorm=1):
    # x_input: sample x dim; y_input: sample x y_dim; kk: x_dim x k_sample
    n=x_input.shape[1]
    if np.max(abs(kk))==0:
        k = np.linspace(min_f,max_f,num=freq_len,endpoint=True)
        kk = np.matmul(np.ones([n,1]),np.reshape(k,[1,-1])); 
    tmp=np.matmul(np.transpose(data), np.exp(-1J * (np.matmul(x_input, kk))))
    if isnorm==1:
        return_fft=np.absolute(tmp) 
    else:
        return_fft=tmp
    return np.squeeze(return_fft)


def my_fft_ori(data):
    
    datat=np.squeeze(data)
    datat_fft = np.fft.fft(datat) 
    return datat_fft

def tfsigderi(x):
    return tf.exp(-x)/(1+tf.exp(-x))**2

def sigderi(x):
    return np.exp(-x)/(1+np.exp(-x))**2


def add_layer(x,input_dim = 1,output_dim = 1,astddev=0.05,bstddev=0.05,ActFuc=0,seed=0,norm=False, name_scope='hidden'):
    if seed==0:
        seed=time.time()
    tf.set_random_seed(seed)
        
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable(
            name='ua_w'
            , shape=[input_dim, output_dim]
            , initializer=tf.random_normal_initializer(stddev=astddev)
        )
        ua_b = tf.get_variable(
            name='ua_b'
            , shape=[output_dim]
            , initializer=tf.random_normal_initializer(stddev=bstddev)
        ) 
        z0=tf.matmul(x, ua_w) + ua_b
#        z = tf.layers.batch_normalization(z0, training=norm)
#        z=z0
        if norm: # 判断是否是Batch Normalization层
            z = tf.layers.batch_normalization(z0, training=norm)
        else:
            z=z0
#        kz=R_variable['ActFuc_kz']
        if ActFuc==1:
            output_z = tf.nn.tanh(z)
            print('tanh')
        elif ActFuc==3:
            output_z = tf.sin(z)
            print('sin')
        elif ActFuc==0:
            output_z = tf.nn.relu(z)
            print('relu')
        elif ActFuc==4:
            output_z = z**50
            print('z**50')
        elif ActFuc==5:
            output_z = tf.nn.sigmoid(z)
            print('sigmoid')
        elif ActFuc==6:
            output_z = tfsigderi(z)
            print('sigmoid deri')
        L2Wight= tf.nn.l2_loss(ua_w) 
        return output_z,ua_w,ua_b,L2Wight

# Our UA function

# >>> used function
def univAprox(x0, hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1,astddev=0.05,bstddev=0.05,ActFuc=0,seed=0,norm=False):
    if seed==0:
        seed=time.time()
    # The simple case is f: R -> R 
    hidden_num = len(hidden_units)
    #print(hidden_num)
    add_hidden = [input_dim] + hidden_units;
    if norm:
        x = tf.layers.batch_normalization(x0, training=norm)
    else:
        x=x0
#    if norm: # 为第一层进行BN
#        fc_mean, fc_var = tf.nn.moments(x, axes=[0])
#        scale = tf.Variable(tf.ones([1]))
#        shift = tf.Variable(tf.zeros([1]))
#        epsilon = 0.001
# 
#        ema = tf.train.ExponentialMovingAverage(decay=0.5)
# 
#        def mean_var_with_update():
#            ema_apply_op = ema.apply([fc_mean, fc_var])
#            with tf.control_dependencies([ema_apply_op]):
#                return tf.identity(fc_mean), tf.identity(fc_var)
# 
#        mean, var = mean_var_with_update()
#        x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
    #tf.assign(output,x)
    output=x
    w_Univ=[]
    b_Univ=[]
    w_std_Univ=[]
    b_std_Univ=[]
    L2w_all=0
    for i in range(hidden_num):
        input_dim = add_hidden[i]
        output_dim = add_hidden[i+1]
        print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
        name_scope = 'hidden' + np.str(i+1)
        output,ua_w,ua_b,L2Wight_tmp=add_layer(output,input_dim,output_dim,
                                               astddev,bstddev, ActFuc,seed, norm, name_scope)
        w_Univ.append(ua_w)
        b_Univ.append(ua_b)
        L2w_all=L2w_all+L2Wight_tmp
        
    ua_we = tf.get_variable(
            name='ua_we'
            , shape=[hidden_units[hidden_num-1], output_dim_final]
            , initializer=tf.random_normal_initializer(stddev=astddev)
        )
    ua_be = tf.get_variable(
            name='ua_be'
            , shape=[1,output_dim_final]
            , initializer=tf.random_normal_initializer(stddev=bstddev)
        )
    z = tf.matmul(output, ua_we)+ua_be
    w_Univ.append(ua_we)
    b_Univ.append(ua_be)

    return z,w_Univ,b_Univ,L2w_all
