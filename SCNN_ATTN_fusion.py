# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:21:36 2022

@author: User01
"""

from SCNN_ATTN_Ts_module import SCNN_ATTN_Ts
import matlab.engine
import scipy.io as io
import glob
import os
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
import joblib
import pickle

comp = 4
total_cm = np.zeros((5,5))
for numm in range(1,11,1):
    
    # if numm == 3:
    #     continue

    ii = str(numm)
    data_dir = 'E:/data2/'+ii
    
       
    # weight calculation
    file = glob.glob(os.path.join('E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/teenager/', "*.mat"))
    data=io.loadmat(file[numm-1])
    # print(file[numm-1])
    xx = data['xx']
    try:
        y_true = data['y_5']
    except:
        y_true = data['yy']
    y_true = y_true.reshape(-1,1)
    e = (len(y_true)-9)/32
    y_true = y_true[0:32*int(e)+9]
    x_temp = xx[:10,:]
    x_temp = x_temp.tolist()
    x_temp = matlab.double(x_temp)
    
    # feature extraction
    eng = matlab.engine.start_matlab()
    x0 = eng.FE(x_temp)
    eng.quit()
    x0 = np.array(x0)
    scaler = pickle.load(open('E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/scaler.pkl', 'rb'))
    X_std = scaler.transform(x0)
    
    #L1
    # data_source = io.loadmat("E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/meanstd.mat")
    # mean = data_source['mean']
    # std = data_source['std']
    # X_std = (x0 - np.ones((len(x0),1)) * mean) 
    # for ll in range(X_std.shape[-1]):
    #     if ll == 4:
    #         continue
    #     X_std[:,ll] = X_std[:,ll] / std[0,ll]
    # X_std[:,4] = x0[:,4]
    
    X_pca = PCA(n_components=comp).fit_transform(X_std)     
    matrix = io.loadmat("E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/trans_marx3.mat")
    trans_marx = matrix['trans']
    X_pca = np.dot(X_std, trans_marx)
        
    knn = joblib.load('E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/knn2.pkl')
    
    
    #L2 
    # matrix = io.loadmat("E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/trans_marx.mat")
    # trans_marx = matrix['trans']
    
    # X_std = StandardScaler().fit_transform(x0.T)
    # X_pca = np.dot(X_std, trans_marx)
    
    # knn = joblib.load('E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/knn.pkl')
    
    # L2+L1
    # matrix = io.loadmat("E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/trans_marx2.mat")
    # trans_marx = matrix['trans']
    
    # X_pre = StandardScaler().fit_transform(x0.T)
    # data_source = io.loadmat("E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/meanstd2.mat")
    # mean = data_source['mean']
    # std = data_source['std']
    # X_std = (X_pre.T - np.ones((len(x0),1)) * mean) 
    # for ll in range(X_std.shape[-1]):
    #     # if ll == 4:
    #     #     continue
    #     X_std[:,ll] = X_std[:,ll] / std[0,ll]
    # # X_std[:,4] = x0[:,4]
    # X_pca = PCA(n_components=comp).fit_transform(X_std)
    # knn = joblib.load('E:/Lab/EOG_Code/infant_sleep - tf20-2/input prepare/true_ouput/multi-crowd/knn3.pkl')
    
    prob_knn = knn.predict_proba(X_pca)
    prob_knn_mean = np.mean(prob_knn, axis=0)
    
    print('Neo prob: {}, Teen prob: {}, Adu prob: {}, \n'.format(prob_knn_mean[0], prob_knn_mean[1], prob_knn_mean[2]))
    people = {0:'Neonate', 1:'Teenager', 2:'Adult'}
    print('Most probable group of people: {}'.format(people[np.argmax(prob_knn_mean)]))

    
    model_dir_neo = 'E:/Lab/EOG_Code/infant_sleep - tf20-2/network/model/multi-group/CNN_ATTN_Ts/CHILD/C4/10/fold'+ii+'/sequence_learning/s_model_weights.h5'
    model_dir_teen = 'E:/Lab/EOG_Code/infant_sleep - tf20-2/network/model/multi-group/CNN_ATTN_Ts/CHAT/C4/10/fold'+ii+'/sequence_learning/s_model_weights.h5'
    model_dir_adult = 'E:/Lab/EOG_Code/infant_sleep - tf20-2/network/model/multi-group/CNN_ATTN_Ts/MASS/C4/10/fold'+ii+'/sequence_learning/s_model_weights.h5'
    
    # probabilitiy acquire
    prob_neo = SCNN_ATTN_Ts(model_dir_neo, data_dir)
    prob_teen = SCNN_ATTN_Ts(model_dir_teen, data_dir)
    prob_adu = SCNN_ATTN_Ts(model_dir_adult, data_dir)
    
    # if numm == 10:
    #     prob_knn_mean[1] += 0.8
    
    # a = np.zeros(3)
    # a[np.where(prob_knn_mean == prob_knn_mean.max())] = 1
    
    
    prob = prob_knn_mean[0]*prob_neo + prob_knn_mean[1]*prob_teen + prob_knn_mean[2]*prob_adu
    # prob = a[0]*prob_neo+a[1]*prob_teen+a[2]*prob_adu
    
    y_pred = np.argmax(prob, axis=1)
    # y_pred.numpy()
    y_pred = y_pred.reshape(-1,1)
    train_cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average="macro") 
    print ("train: (acc={:.3f},f1={:.3f})".format(acc, f1))
    print(train_cm)
    total_cm += train_cm

print(total_cm)
    
        
    