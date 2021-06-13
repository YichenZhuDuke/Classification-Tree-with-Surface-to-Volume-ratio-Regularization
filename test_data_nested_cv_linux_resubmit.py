# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:09:08 2020

This file is used to test the performance of SVR-Tree and other Tree based imbalanced
classification methods. 

@author: acezy
"""

'''to execute inside python, type: exec(open('test_data_nested_cv_linux.py').read())'''


import sys
''' Please append your own path here. '''
sys.path.append('D:/Research/AUC_IB/code2020')
sys.path.append('D:/Research/AUC_IB/data')
sys.path.append('/home/grad/yz486/ImbalanceData/code2021')
sys.path.append('/home/grad/yz486/ImbalanceData/data')
import numpy as np
import pandas as pd
import Tree
import multiprocessing as multip
import time
import sampler



t0 = time.time()

''' Reading data sets. Please choose data_name from: 
    'vechicle', 'pima', 'abalone', 'satimage', 'wine', 'glass', 'page', 'yeast', 'shuttle', 'segment', 'vowel', 'ecoli', 'penbased'  '''
   
data_name = 'titanic'

if data_name == 'vehicle':
    '''vehicle data'''
    da = [None] * 9
    filenames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    for i in range(9):
        da[i] = pd.read_table('/home/grad/yz486/ImbalanceData/data/xa'+filenames[i]+'.dat', sep=' ', header=None, error_bad_lines=False, \
                              keep_default_na=False)    ## read file in linux
    da_vehicle = pd.concat(da)
    da_vehicle['class'] = np.array(da_vehicle[18] == 'van', dtype=int)
    da_vehicle = da_vehicle.drop([18], axis=1)
    da = da_vehicle.values
    Xall = da[:,0:18]
    Xall = Xall.astype('float')
    for i in range(18):
        Xall[:,i] = Xall[:,i] / np.max(Xall[:,i])
    Yall = da[:,18]
    n, d = np.shape(Xall)
    times = 2
elif data_name == 'pima':
    '''Pima data'''
    da_pima = pd.read_csv('/home/grad/yz486/ImbalanceData/data/diabetes.csv')
    da = da_pima.values
    Xall = da[:,0:8]
    Yall = da[:,8]
    Yall = Yall.astype(int)
    expan = 1.01
    for i in range(8):
        Xall[:,i] = Xall[:,i] / np.max(Xall[:,i]) / expan
    n, d = np.shape(Xall)
    times = 1
elif data_name == 'wine':
    '''Wine data'''
    da_wine = pd.read_csv('/home/grad/yz486/ImbalanceData/data/winequality-red.csv', sep=';')
    da = da_wine.values
    Xall = da[:,0:11]
    Yall = np.zeros(np.shape(Xall)[0])
    Yall[np.flatnonzero(da[:,11]>=7)] = 1
    n, d = np.shape(Xall)
    times = 5
elif data_name == 'abalone':
    '''Abalone data'''
    da_abalone = pd.read_csv('/home/grad/yz486/ImbalanceData/data/abalone.data', sep=',')
    da_abalone = da_abalone.drop(columns='M')
    da = da_abalone.values
    Xall = da[:,0:7]
    n1_ind = np.flatnonzero(da[:,7] == 18)
    n0_ind = np.flatnonzero(da[:,7] == 9)
    all_ind = np.concatenate((n1_ind, n0_ind))
    Xall = da[all_ind, 0:7]
    Yall = np.zeros(len(all_ind))
    Yall[0:len(n1_ind)] = 1
    n, d = np.shape(Xall)
    times = 15
elif data_name == 'satimage':
    '''For Satimage data'''
    da_a = pd.read_table('/home/grad/yz486/ImbalanceData/data/sat_train.txt', sep=' ', header=None, error_bad_lines=False, \
                              keep_default_na=False)
    da_a = da_a.values
    Xa = da_a[:,0:36]
    Ya = da_a[:,36]
    Ya = np.int_(Ya==4)
    da_b = pd.read_table('/home/grad/yz486/ImbalanceData/data/sat_test.txt', sep=' ', header=None, error_bad_lines=False, \
                              keep_default_na=False)
    da_b = da_b.values
    Xb = da_b[:,0:36]
    Yb = da_b[:,36]
    Yb = np.int_(Yb==4)
    Xall = np.vstack((Xa, Xb))
    Yall = np.concatenate((Ya, Yb))
    n, d = np.shape(Xall)
    times = 8
elif data_name == 'page':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/page-blocks0.dat', sep=',', header=None)
    da = da0.values
    Xall = da[:,0:10]
    n1_ind = np.flatnonzero(da[:,10] == ' positive')
    Yall = np.zeros(np.shape(da)[0],dtype=int)
    Yall[n1_ind] = 1
    n, d = np.shape(Xall)
    times = int((n-len(n1_ind))/len(n1_ind)) - 1
elif data_name == 'yeast':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/yeast4.dat', sep=',', header=None)
    da = da0.values
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' positive')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1        
elif data_name == 'segment':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/segment0.dat', sep=',', header=None)
    da = np.delete(da0.values, 2, axis=1)      ## remove the second feature as it contains no information
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' positive')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1        
elif data_name == 'ecoli':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/ecoli.dat', sep=',', header=None)
    da = np.delete(da0.values, 3, axis=1)       ## remove the third feature as it contains little information 
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' pp')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1           
elif data_name == 'glass2':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/glass2.dat', sep=',', header=None)
    da = da0.values      
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' positive')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1     
elif data_name == 'phoneme':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/phoneme.dat', sep=',', header=None)
    da = da0.values      
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d])
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1           
elif data_name == 'titanic':
    da0 = pd.read_table('/home/grad/yz486/ImbalanceData/data/titanic.dat', sep=',', header=None)
    da = da0.values      
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d]==1)
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1     

    
#%%
'''Experiment Functions'''
def performance_stats(TP, FP, FN, TN):
    TPR = TP / (TP+FN)
    if TP+FP > 0:
        precision = TP / (TP+FP)
    else:
        precision = 0
    accuracy = (TP+TN) / (TP+FP+FN+TN)
    TNR = TN / (TN+FP)
    G_mean = np.sqrt(TPR*TNR)
    if precision > 0:
        F_measure = 2*TPR*precision / (TPR+precision)
    else:
        F_measure = 0
    return np.array([accuracy, precision, TPR, TNR, F_measure, G_mean])

def divide(n, m):
    '''Function to divide n samples into m roughly equal folds.'''
    n_low = n // m
    out = np.ones(m,dtype=int) * n_low
    remain = n % m
    out[0:remain] = out[0:remain] + 1
    return out

def id_divide(ids, m):
    '''Function to divide id_seq into m roughly equal folds.'''
    n = len(ids)
    n_in_folds = divide(n, m)
    id_lst = [None]*m
    loc = 0
    for i in range(m):
        loc_new = loc + n_in_folds[i]
        id_lst[i] = ids[loc:loc_new]
        loc = loc_new
    return id_lst  

def experiment_runner(ids, Xall, Yall, n_cv, n_cv_out, pen_lst, weight, c0, alpha_lst):
    X = Xall[ids,:]
    Y = Yall[ids]
    n = len(Y)
    n1 = np.int_(np.sum(Y))
    n0 = n - n1
    n1_divide = divide(n1, n_cv_out)
    n0_divide = divide(n0, n_cv_out)
    id_1 = np.flatnonzero(Y)            ## all variables starting with "id" are indices of X and Y (not Xall or Yall)
    id_0 = np.flatnonzero(Y==0)
    TPFP_svr = np.zeros(2)
    TPFP_svr_select = np.zeros(2)
    TPFP_duplicate = np.zeros(2)
    TPFP_SMOTE = np.zeros(2)
    TPFP_BSMOTE = np.zeros(2)
    TPFP_ADASYN = np.zeros(2)    
    
    for test_fold_no in range(n_cv_out):    
        id_1test_start = np.int_(np.sum(n1_divide[0:test_fold_no]))
        id_0test_start = np.int_(np.sum(n0_divide[0:test_fold_no]))
        id_1test = id_1[id_1test_start:(id_1test_start+n1_divide[test_fold_no])]
        id_1train = np.delete(id_1, np.arange(id_1test_start,id_1test_start+n1_divide[test_fold_no]))
        id_0test = id_0[id_0test_start:(id_0test_start+n0_divide[test_fold_no])]
        id_0train = np.delete(id_0, np.arange(id_0test_start,id_0test_start+n0_divide[test_fold_no]))
        n1train = len(id_1train)
        n1test = len(id_1test)
        n0train = len(id_0train)
        n0test = len(id_0test)
        id_1train_cv = id_divide(id_1train, n_cv)
        id_0train_cv = id_divide(id_0train, n_cv)
        id_cv = [None] * n_cv
        for k in range(n_cv):
            id_cv[k] = np.concatenate((id_1train_cv[k], id_0train_cv[k]))
        id_train = np.concatenate((id_1train, id_0train))
        Xtrain = X[id_train,:]
        Ytrain = Y[id_train]
        id_test = np.concatenate((id_1test, id_0test))
        Xtest = X[id_test,:]
        Ytest = Y[id_test]
        
        '''SVR tree'''
        F_lst = np.zeros(len(pen_lst))
        for j in range(len(pen_lst)):
            TP = 0
            FP = 0
            for k in range(n_cv):
                id_cv_copy = id_cv.copy()
                del id_cv_copy[k]
                id_temp = np.concatenate(id_cv_copy)
                Xtrain_temp = X[id_temp,:]
                Ytrain_temp = Y[id_temp]
                Xtest_temp = X[id_cv[k],:]
                Ytest_temp = Y[id_cv[k]]
                tr_svr = Tree.tree()
                tr_svr.fit_sv(Xtrain_temp, Ytrain_temp, pen_lst[j], weight=weight, feature_select=False, maximal_leaves=2*np.sqrt(n*2/3))
                Y_pred_temp = tr_svr.predict(Xtest_temp)
                TP += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp==0)])
            if TP > 0:
                tpr = TP / n1train
                precision = TP / (TP+FP)
                F_lst[j] = 2*tpr*precision / (tpr+precision)
        para_id = np.argmax(F_lst)
        tr_svr = Tree.tree()
        tr_svr.fit_sv(Xtrain, Ytrain, pen_lst[para_id], weight=weight, feature_select=False, maximal_leaves=2*np.sqrt(n*2/3))
        Y_pred = tr_svr.predict(Xtest)    
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        FP = np.sum(Y_pred[np.flatnonzero(Ytest==0)])
        TPFP_svr = TPFP_svr + np.array([TP, FP])
            
        '''SVR tree with feature selection'''
        F_lst = np.zeros(len(pen_lst))
        for j in range(len(pen_lst)):
            TP = 0
            FP = 0
            for k in range(n_cv):
                id_cv_copy = id_cv.copy()
                del id_cv_copy[k]
                id_temp = np.concatenate(id_cv_copy)
                Xtrain_temp = X[id_temp,:]
                Ytrain_temp = Y[id_temp]
                Xtest_temp = X[id_cv[k],:]
                Ytest_temp = Y[id_cv[k]]
                tr_svr_select = Tree.tree()
                tr_svr_select.fit_sv(Xtrain_temp, Ytrain_temp, pen_lst[j], weight=weight, feature_select=True, c0=c0, maximal_leaves=2*np.sqrt(n*2/3))
                Y_pred_temp = tr_svr_select.predict(Xtest_temp)
                TP += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp==0)])
            if TP > 0:
                tpr = TP / n1train
                precision = TP / (TP+FP)
                F_lst[j] = 2*tpr*precision / (tpr+precision)
        para_id = np.argmax(F_lst)
        tr_svr_select = Tree.tree()
        tr_svr_select.fit_sv(Xtrain, Ytrain, pen_lst[para_id], weight=weight, feature_select=True, c0=c0, maximal_leaves=2*np.sqrt(n*2/3))
        Y_pred = tr_svr_select.predict(Xtest)    
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        FP = np.sum(Y_pred[np.flatnonzero(Ytest==0)])
        TPFP_svr_select = TPFP_svr_select + np.array([TP, FP])
    
        '''CART with duplicate samples'''
        F_lst = np.zeros(len(alpha_lst))
        TP_lst = np.zeros(len(alpha_lst))
        FP_lst = np.zeros(len(alpha_lst))
        for k in range(n_cv):
            id_cv_copy = id_cv.copy()
            del id_cv_copy[k]
            id_temp = np.concatenate(id_cv_copy)
            Xtrain_temp = X[id_temp,:]
            Ytrain_temp = Y[id_temp]
            Xtest_temp = X[id_cv[k],:]
            Ytest_temp = Y[id_cv[k]]
            sampler_now = sampler.sampler(times, 'duplicate')
            Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain_temp, Ytrain_temp)
            tr_cart = Tree.tree()
            tr_cart.fit(Xtrain_res, Ytrain_res)
            treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
            alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
            for j in range(len(alpha_lst)):
                i = 0
                while alpha_prune_lst[i+1] <= alpha_lst[j]:
                    i = i + 1
                Y_pred_temp = treelst[i].predict(Xtest_temp)
                TP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp==0)])
        for j in range(len(alpha_lst)):
            if TP_lst[j] > 0:
                tpr = TP_lst[j] / n1train
                precision = TP_lst[j] / (TP_lst[j]+FP_lst[j])
                F_lst[j] = 2*tpr*precision / (tpr+precision)   
        para_id = np.argmax(F_lst)
        tr_cart = Tree.tree()
        tr_cart.fit(Xtrain, Ytrain)
        treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
        alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[para_id]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        FP = np.sum(Y_pred[np.flatnonzero(Ytest==0)])
        TPFP_duplicate = TPFP_duplicate + np.array([TP, FP])
        
        '''CART with SMOTE'''
        F_lst = np.zeros(len(alpha_lst))
        TP_lst = np.zeros(len(alpha_lst))
        FP_lst = np.zeros(len(alpha_lst))
        for k in range(n_cv):
            id_cv_copy = id_cv.copy()
            del id_cv_copy[k]
            id_temp = np.concatenate(id_cv_copy)
            Xtrain_temp = X[id_temp,:]
            Ytrain_temp = Y[id_temp]
            Xtest_temp = X[id_cv[k],:]
            Ytest_temp = Y[id_cv[k]]
            sampler_now = sampler.sampler(times, 'SMOTE')
            Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain_temp, Ytrain_temp)
            tr_cart = Tree.tree()
            tr_cart.fit(Xtrain_res, Ytrain_res)
            treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
            alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
            for j in range(len(alpha_lst)):
                i = 0
                while alpha_prune_lst[i+1] <= alpha_lst[j]:
                    i = i + 1
                Y_pred_temp = treelst[i].predict(Xtest_temp)
                TP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp==0)])
        for j in range(len(alpha_lst)):
            if TP_lst[j] > 0:
                tpr = TP_lst[j] / n1train
                precision = TP_lst[j] / (TP_lst[j]+FP_lst[j])
                F_lst[j] = 2*tpr*precision / (tpr+precision)   
        para_id = np.argmax(F_lst)
        tr_cart = Tree.tree()
        tr_cart.fit(Xtrain, Ytrain)
        treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
        alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[para_id]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        FP = np.sum(Y_pred[np.flatnonzero(Ytest==0)])
        TPFP_SMOTE = TPFP_SMOTE + np.array([TP, FP])
    
        '''CART with B-SMOTE'''
        F_lst = np.zeros(len(alpha_lst))
        TP_lst = np.zeros(len(alpha_lst))
        FP_lst = np.zeros(len(alpha_lst))
        for k in range(n_cv):
            id_cv_copy = id_cv.copy()
            del id_cv_copy[k]
            id_temp = np.concatenate(id_cv_copy)
            Xtrain_temp = X[id_temp,:]
            Ytrain_temp = Y[id_temp]
            Xtest_temp = X[id_cv[k],:]
            Ytest_temp = Y[id_cv[k]]
            sampler_now = sampler.sampler(times, 'BSMOTE')
            Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain_temp, Ytrain_temp)
            tr_cart = Tree.tree()
            tr_cart.fit(Xtrain_res, Ytrain_res)
            treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
            alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
            for j in range(len(alpha_lst)):
                i = 0
                while alpha_prune_lst[i+1] <= alpha_lst[j]:
                    i = i + 1
                Y_pred_temp = treelst[i].predict(Xtest_temp)
                TP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp==0)])
        for j in range(len(alpha_lst)):
            if TP_lst[j] > 0:
                tpr = TP_lst[j] / n1train
                precision = TP_lst[j] / (TP_lst[j]+FP_lst[j])
                F_lst[j] = 2*tpr*precision / (tpr+precision)   
        para_id = np.argmax(F_lst)
        tr_cart = Tree.tree()
        tr_cart.fit(Xtrain, Ytrain)
        treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
        alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[para_id]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        FP = np.sum(Y_pred[np.flatnonzero(Ytest==0)])
        TPFP_BSMOTE = TPFP_BSMOTE + np.array([TP, FP])
     
        '''CART with ADASYN'''
        F_lst = np.zeros(len(alpha_lst))
        TP_lst = np.zeros(len(alpha_lst))
        FP_lst = np.zeros(len(alpha_lst))
        for k in range(n_cv):
            id_cv_copy = id_cv.copy()
            del id_cv_copy[k]
            id_temp = np.concatenate(id_cv_copy)
            Xtrain_temp = X[id_temp,:]
            Ytrain_temp = Y[id_temp]
            Xtest_temp = X[id_cv[k],:]
            Ytest_temp = Y[id_cv[k]]
            sampler_now = sampler.sampler(times, 'ADASYN')
            Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain_temp, Ytrain_temp)
            tr_cart = Tree.tree()
            tr_cart.fit(Xtrain_res, Ytrain_res)
            treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
            alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
            for j in range(len(alpha_lst)):
                i = 0
                while alpha_prune_lst[i+1] <= alpha_lst[j]:
                    i = i + 1
                Y_pred_temp = treelst[i].predict(Xtest_temp)
                TP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp)])
                FP_lst[j] += np.sum(Y_pred_temp[np.flatnonzero(Ytest_temp==0)])
        for j in range(len(alpha_lst)):
            if TP_lst[j] > 0:
                tpr = TP_lst[j] / n1train
                precision = TP_lst[j] / (TP_lst[j]+FP_lst[j])
                F_lst[j] = 2*tpr*precision / (tpr+precision)   
        para_id = np.argmax(F_lst)
        tr_cart = Tree.tree()
        tr_cart.fit(Xtrain, Ytrain)
        treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
        alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[para_id]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        TP = np.sum(Y_pred[np.flatnonzero(Ytest)])
        FP = np.sum(Y_pred[np.flatnonzero(Ytest==0)])
        TPFP_ADASYN = TPFP_ADASYN + np.array([TP, FP])
    
    results_svr = performance_stats(TPFP_svr[0], TPFP_svr[1], n1-TPFP_svr[0], n0-TPFP_svr[1])
    results_svr_select = performance_stats(TPFP_svr_select[0], TPFP_svr_select[1], n1-TPFP_svr_select[0], n0-TPFP_svr_select[1])
    results_duplicate = performance_stats(TPFP_duplicate[0], TPFP_duplicate[1], n1-TPFP_duplicate[0], n0-TPFP_duplicate[1])
    results_SMOTE = performance_stats(TPFP_SMOTE[0], TPFP_SMOTE[1], n1-TPFP_SMOTE[0], n0-TPFP_SMOTE[1])
    results_BSMOTE = performance_stats(TPFP_BSMOTE[0], TPFP_BSMOTE[1], n1-TPFP_BSMOTE[0], n0-TPFP_BSMOTE[1])
    results_ADASYN = performance_stats(TPFP_ADASYN[0], TPFP_ADASYN[1], n1-TPFP_ADASYN[0], n0-TPFP_ADASYN[1])
    
    return (results_svr, results_svr_select, results_duplicate, results_SMOTE, results_BSMOTE, results_ADASYN)
        


#%%
'''Linux exepriments'''
seednum = 40
np.random.seed(seednum)
nexps = 20     ## number of nested cross-validation experiments
n1 = len(np.flatnonzero(Yall))
n0 = n - n1
id_permutes = np.zeros((nexps, n),dtype=int)
for i in range(nexps):
    id_permutes[i,:] = np.random.permutation(n)
id_filename = data_name+'_seed'+str(seednum)+'_ids'
np.save(id_filename, id_permutes) 
n_cv = 5
n_cv_out = 3
train_ratio = 1 - 1/n_cv_out
n1train = np.int_(n1*train_ratio)
n0train = np.int_(n0*train_ratio)

## The below alpha_lst is for common datasets
alpha_lst = np.array([0, 1/256, 1/128, 1/64, 1/32, 1/16, 0.125, 0.177, 0.25, 0.35, 0.5, 0.71, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024, 1450, 2048, 2896, 4096]) * 10**(-3) * (n*train_ratio)**(-1/3)
## The below alpha_lst is for satimage datasets
# alpha_lst = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 15000, 30000, 60000, 120000]) * 10**(-3) * (n0train+n1train)**(-1/3)
pen_lst = np.array([0, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024]) * 10**(-3) * (n0train+n1train)**(-1/3)
weight=times+1
c0=4

inputs = [None]*nexps
print('Run experiments for '+str(data_name))
t1 = time.time()
print('head time: '+str(t1-t0))
for i in range(nexps):
    inputs[i] = (id_permutes[i,:], Xall, Yall, n_cv, n_cv_out, pen_lst, weight, c0, alpha_lst)
with multip.Pool(processes=nexps) as pool:
    outputs = pool.starmap(experiment_runner, inputs)    
t2 = time.time()
print('main programs time: '+str(t2-t1))
outputs_filename = data_name+'_seed'+str(seednum)+'_outputs'
np.save(outputs_filename, outputs)

results_mat_svr = np.zeros((nexps, 6))
results_mat_svr_select = np.zeros((nexps, 6))
results_mat_duplicate = np.zeros((nexps, 6))
results_mat_SMOTE = np.zeros((nexps, 6))
results_mat_BSMOTE = np.zeros((nexps, 6))
results_mat_ADASYN = np.zeros((nexps, 6))
for i in range(nexps):
    output_now = outputs[i]
    results_mat_svr[i,:] = output_now[0]
    results_mat_svr_select[i,:] = output_now[1]
    results_mat_duplicate[i,:] = output_now[2]
    results_mat_SMOTE[i,:] = output_now[3]
    results_mat_BSMOTE[i,:] = output_now[4]
    results_mat_ADASYN[i,:] = output_now[5]

mean_summary = np.zeros((6,6))
std_summary = np.zeros((6,6))
'''The resulst, by orders, are: accuracy, precision, TPR, TNR, F_measure, G_mean'''
mean_summary[0,:] = np.mean(results_mat_svr, axis=0)
mean_summary[1,:] = np.mean(results_mat_svr_select, axis=0)
mean_summary[2,:] = np.mean(results_mat_duplicate, axis=0)
mean_summary[3,:] = np.mean(results_mat_SMOTE, axis=0)
mean_summary[4,:] = np.mean(results_mat_BSMOTE, axis=0)
mean_summary[5,:] = np.mean(results_mat_ADASYN, axis=0)

std_summary[0,:] = np.std(results_mat_svr, axis=0)
std_summary[1,:] = np.std(results_mat_svr_select, axis=0)
std_summary[2,:] = np.std(results_mat_duplicate, axis=0)
std_summary[3,:] = np.std(results_mat_SMOTE, axis=0)
std_summary[4,:] = np.std(results_mat_BSMOTE, axis=0)
std_summary[5,:] = np.std(results_mat_ADASYN, axis=0)

t3 = time.time()
print('tail time: '+str(t3-t2))

mean_filename = data_name+'_seed'+str(seednum)+'_mean'
np.save(mean_filename, mean_summary)
std_filename = data_name+'_seed'+str(seednum)+'_std'
np.save(std_filename, std_summary)

''' Print the key performance measure stats and average number of features selected. '''
print(mean_summary)
print(std_summary)





