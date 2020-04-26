# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:09:08 2020

@author: Yichen Zhu
"""
import sys
sys.path.append('D:/Research/AUC_IB/code2020')
sys.path.append('D:/Research/AUC_IB/data')
sys.path.append('/home/grad/yz486/ImbalanceData')
sys.path.append('/home/grad/yz486/ImbalanceData/data')
import numpy as np
import pandas as pd
import Tree
import multiprocessing as multip
import time
import sampler



t0 = time.time()

def performance_stats(TPR, TNR, n1ratio):
    '''Input column vector TPR, TNR, return column vector accuracy, precision, TPR(recall), 
            f-measure, g-mean, algorithm mean'''
    n0ratio = 1 - n1ratio
    accuracy = TPR*n1ratio + TNR*n0ratio
    normal_ind = np.flatnonzero(TPR)
    precision = np.zeros(len(TPR))
    precision[normal_ind] = TPR[normal_ind]*n1ratio / (TPR[normal_ind]*n1ratio + (1-TNR[normal_ind])*n0ratio)
    F_measure = np.zeros(len(TPR))
    F_measure[normal_ind] = 2*TPR[normal_ind]*precision[normal_ind] / (TPR[normal_ind]+precision[normal_ind])
    G_mean = np.sqrt(TNR*TPR)
    performances = np.vstack((accuracy, precision, TPR, F_measure, G_mean))
    return performances.transpose()

d_redundant = 10


###################################################################################
''' Reading data sets. Choose which data set you want to test on and only uncomment
that corresponding paragraphs. Current this file is testing on "glass" data. '''

#'''vehicle data'''
#da = [None] * 9
#filenames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
#for i in range(9):
#    da[i] = pd.read_table('/home/grad/yz486/ImbalanceData/data/xa'+filenames[i]+'.dat', sep=' ', header=None, error_bad_lines=False, \
#                          keep_default_na=False)    ## read file in windows
##    da[i] = pd.read_table('/home/grad/yz486/ImbalanceData/data/xa'+filenames[i]+'.dat', sep=' ', header=None, error_bad_lines=False, \
##                          keep_default_na=False)    ## read file in linux
#da_vehicle = pd.concat(da)
#da_vehicle['class'] = np.array(da_vehicle[18] == 'van', dtype=int)
#da_vehicle = da_vehicle.drop([18], axis=1)
#da = da_vehicle.values
#Xall = da[:,0:18]
#Xall = Xall.astype('float')
#Yall = da[:,18]
#n, d = np.shape(Xall)
#times = 3


#'''Pima data'''
#da_pima = pd.read_csv('/home/grad/yz486/ImbalanceData/data/diabetes.csv')
#da = da_pima.values
#Xall = da[:,0:8]
#Yall = da[:,8]
#Yall = Yall.astype(int)
#n, d = np.shape(Xall)
#times = 2


'''Abalone data'''
da_abalone = pd.read_csv('/home/grad/yz486/ImbalanceData/data/abalone.data', sep=',')
#da_abalone = pd.read_csv('D:/Research/AUC_IB/data/abalone.data', sep=',')
da_abalone = da_abalone.drop(columns='M')
da = da_abalone.values
Xall = da[:,0:7]
n1_ind = np.flatnonzero(da[:,7] == 18)
n0_ind = np.flatnonzero(da[:,7] == 9)
all_ind = np.concatenate((n1_ind, n0_ind))
Xall = da[all_ind, 0:7]
n, d = np.shape(Xall)
X_redundant = np.random.rand(n, d_redundant)
Xall = np.hstack((Xall, X_redundant))
Yall = np.zeros(len(all_ind))
Yall[0:len(n1_ind)] = 1
n, d = np.shape(Xall)
times = 10


#'''For Satimage data'''
#da_a = pd.read_table('/home/grad/yz486/ImbalanceData/data/sat_train.txt', sep=' ', header=None, error_bad_lines=False, \
#                          keep_default_na=False)
#da_a = da_a.values
#Xa = da_a[:,0:36]
#Ya = da_a[:,36]
#Ya = np.int_(Ya==4)
#da_b = pd.read_table('/home/grad/yz486/ImbalanceData/data/sat_test.txt', sep=' ', header=None, error_bad_lines=False, \
#                          keep_default_na=False)
#da_b = da_b.values
#Xb = da_b[:,0:36]
#Yb = da_b[:,36]
#Yb = np.int_(Yb==4)
#Xall = np.vstack((Xa, Xb))
#Yall = np.concatenate((Ya, Yb))
#n, d = np.shape(Xall)
#times = 8


#'''Wine data'''
#da_wine = pd.read_csv('/home/grad/yz486/ImbalanceData/data/winequality-red.csv', sep=';')
#da = da_wine.values
#Xall = da[:,0:11]
#n, d = np.shape(Xall)
#X_redundant = np.random.rand(n, d_redundant)
#Xall = np.hstack((Xall, X_redundant))
#n, d = np.shape(Xall)
#Yall = np.zeros(np.shape(Xall)[0])
#Yall[np.flatnonzero(da[:,11]>=7)] = 1
#times = 5


#'''Glass data'''
#da_glass = pd.read_csv('/home/grad/yz486/ImbalanceData/data/glass.data', sep=',')
##da_bcwe = da_bcw.drop(columns='M')
#da = da_glass.values
#Xall = da[:,1:10]
#n, d = np.shape(Xall)
#X_redundant = np.random.rand(n, d_redundant)
#Xall = np.hstack((Xall, X_redundant))
#n, d = np.shape(Xall)
#n1_ind = np.flatnonzero(da[:,10] == 7)
#Yall = np.zeros(np.shape(da)[0])
#Yall[n1_ind] = 1
#times = 5

'''End of reading data sets. '''
###############################################################################


    
    
    




    
    

'''Experiments codes'''
nexps = 50     ## number of repetitive experiments
n1 = len(np.flatnonzero(Yall))
n0 = n - n1
n1train = np.int_(np.ceil(n1*2/3))
n1test = n1 - n1train
n0train = np.int_(np.ceil(n0*2/3))
n0test = n0 - n0train
## The below alpha_lst is for common datasets
alpha_lst = np.array([0, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024]) * 10**(-3) * (n0train+n1train)**(-1/3)
### The below alpha_lst is for satimage datasets
#alpha_lst = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 15000, 30000, 60000, 120000]) * 10**(-3) * (n0train+n1train)**(-1/3)

pen_lst = np.array([0, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024]) * 10**(-3) * (n0train+n1train)**(-1/3)
weight=times+1
c0=4
print(n, d)

def experiment_runner(seed, Xall, Yall, n1, n0, pen_lst, weight, c0, alpha_lst):
    np.random.seed(seed)
    X1 = Xall[np.flatnonzero(Yall),:]
    X0 = Xall[np.flatnonzero(Yall==0),:]
    n1_permute = np.random.permutation(n1)
    n0_permute = np.random.permutation(n0)
    Xtrain = np.vstack((X1[n1_permute[0:n1train],:], X0[n0_permute[0:n0train],:]))
    Ytrain = np.concatenate((np.ones(n1train), np.zeros(n0train)))
    Xtest = np.vstack((X1[n1_permute[n1train:n1],:], X0[n0_permute[n0train:n0],:]))
    Ytest = np.concatenate((np.ones(n1test), np.zeros(n0test)))    
    Trate_svr = np.zeros((len(pen_lst), 2))
    Trate_svr_feat_select = np.zeros((len(pen_lst), 2))
    Trate_duplicate = np.zeros((len(alpha_lst), 2))
    Trate_SMOTE = np.zeros((len(alpha_lst), 2))
    Trate_BSMOTE = np.zeros((len(alpha_lst), 2))
    Trate_ADASYN = np.zeros((len(alpha_lst), 2))
    num_feats_n = np.zeros((max(len(alpha_lst),len(pen_lst)), 6))   ## records the number of nonredundant features used for: svr, svr_feat_select, duplicate, SMOTE, BSMOTE, ADASYN
    num_feats_r = np.zeros((max(len(alpha_lst),len(pen_lst)), 6))   ## records the number of redundant features
       
    '''SVR tree'''
    for j in range(len(pen_lst)):
        tr_svr = Tree.tree()
        tr_svr.fit_sv(Xtrain, Ytrain, pen_lst[j], weight=weight, feature_select=False, maximal_leaves=2*np.sqrt(n*2/3))
        Y_pred = tr_svr.predict(Xtest)
        Trate_svr[j,0] = 1 - sum(abs((Ytest-Y_pred)[0:n1test])) / n1test   ##True positive rate
        Trate_svr[j,1] = 1 - sum(abs((Ytest-Y_pred)[n1test:(n1test+n0test)])) / n0test   ##True negative rate 
        num_feats_n[j,0] +=sum(tr_svr.feats_usage[0:(d-d_redundant)])
        num_feats_r[j,0] +=sum(tr_svr.feats_usage[(d-d_redundant):d])

    '''SVR tree with feature selection'''
    for j in range(len(pen_lst)):
        tr_svr_fs = Tree.tree()
        tr_svr_fs.fit_sv(Xtrain, Ytrain, pen_lst[j], weight=weight, feature_select=True, c0=c0, maximal_leaves=2*np.sqrt(n*2/3))
        Y_pred = tr_svr_fs.predict(Xtest)
        Trate_svr_feat_select[j,0] = 1 - sum(abs((Ytest-Y_pred)[0:n1test])) / n1test   ##True positive rate
        Trate_svr_feat_select[j,1] = 1 - sum(abs((Ytest-Y_pred)[n1test:(n1test+n0test)])) / n0test   ##True negative rate 
        num_feats_n[j,1] +=sum(tr_svr_fs.feats_usage[0:(d-d_redundant)])
        num_feats_r[j,1] +=sum(tr_svr_fs.feats_usage[(d-d_redundant):d])

    '''CART with duplicate resamples'''
    sampler_now = sampler.sampler(times, 'duplicate')
    Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain, Ytrain)
    tr_cart = Tree.tree()
    tr_cart.fit(Xtrain_res, Ytrain_res)
    treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
    alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
    for j in range(len(alpha_lst)):
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[j]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        Trate_duplicate[j,0] = 1 - sum(abs((Ytest-Y_pred)[0:n1test])) / n1test   ##True positive rate
        Trate_duplicate[j,1] = 1 - sum(abs((Ytest-Y_pred)[n1test:(n1test+n0test)])) / n0test   ##True negative rate 
        treelst[i].compute_feats_usage()
        num_feats_n[j,2] +=sum(treelst[i].feats_usage[0:(d-d_redundant)])
        num_feats_r[j,2] +=sum(treelst[i].feats_usage[(d-d_redundant):d])
        
    '''CART with SMOTE'''
    sampler_now = sampler.sampler(times, 'SMOTE')
    Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain, Ytrain)
    tr_cart = Tree.tree()
    tr_cart.fit(Xtrain_res, Ytrain_res)
    treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
    alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
    for j in range(len(alpha_lst)):
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[j]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        Trate_SMOTE[j,0] = 1 - sum(abs((Ytest-Y_pred)[0:n1test])) / n1test   ##True positive rate
        Trate_SMOTE[j,1] = 1 - sum(abs((Ytest-Y_pred)[n1test:(n1test+n0test)])) / n0test   ##True negative rate 
        treelst[i].compute_feats_usage()
        num_feats_n[j,3] +=sum(treelst[i].feats_usage[0:(d-d_redundant)])
        num_feats_r[j,3] +=sum(treelst[i].feats_usage[(d-d_redundant):d])
        
    '''CART with B-SMOTE'''
    sampler_now = sampler.sampler(times, 'BSMOTE')
    Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain, Ytrain)
    tr_cart = Tree.tree()
    tr_cart.fit(Xtrain_res, Ytrain_res)
    treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
    alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
    for j in range(len(alpha_lst)):
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[j]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        Trate_BSMOTE[j,0] = 1 - sum(abs((Ytest-Y_pred)[0:n1test])) / n1test   ##True positive rate
        Trate_BSMOTE[j,1] = 1 - sum(abs((Ytest-Y_pred)[n1test:(n1test+n0test)])) / n0test   ##True negative rate 
        treelst[i].compute_feats_usage()
        num_feats_n[j,4] +=sum(treelst[i].feats_usage[0:(d-d_redundant)])
        num_feats_r[j,4] +=sum(treelst[i].feats_usage[(d-d_redundant):d])  
        
    '''CART with ADASYN'''
    sampler_now = sampler.sampler(times, 'ADASYN')
    Xtrain_res, Ytrain_res = sampler_now.fit_resample(Xtrain, Ytrain)
    tr_cart = Tree.tree()
    tr_cart.fit(Xtrain_res, Ytrain_res)
    treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
    alpha_prune_lst = np.append(alpha_prune_lst, np.inf)
    for j in range(len(alpha_lst)):
        i = 0
        while alpha_prune_lst[i+1] <= alpha_lst[j]:
            i = i + 1
        Y_pred = treelst[i].predict(Xtest)
        Trate_ADASYN[j,0] = 1 - sum(abs((Ytest-Y_pred)[0:n1test])) / n1test   ##True positive rate
        Trate_ADASYN[j,1] = 1 - sum(abs((Ytest-Y_pred)[n1test:(n1test+n0test)])) / n0test   ##True negative rate 
        treelst[i].compute_feats_usage()
        num_feats_n[j,5] +=sum(treelst[i].feats_usage[0:(d-d_redundant)])
        num_feats_r[j,5] +=sum(treelst[i].feats_usage[(d-d_redundant):d])

    return (Trate_svr, Trate_svr_feat_select, Trate_duplicate, Trate_SMOTE, Trate_BSMOTE, Trate_ADASYN, num_feats_n, num_feats_r)
        

'''The codes that runs the experiments'''
seed_lst = np.random.choice(np.arange(10**6),nexps,replace=False)
inputs = [None]*nexps
t1 = time.time()
print('head time: '+str(t1-t0))
for i in range(nexps):
    inputs[i] = (seed_lst[i], Xall, Yall, n1, n0, pen_lst, weight, c0, alpha_lst)
with multip.Pool(processes=nexps) as pool:
    outputs = pool.starmap(experiment_runner, inputs)    
t2 = time.time()
print('main programs time: '+str(t2-t1))
''' '''
     
Trate_mat_svr = np.zeros((nexps, len(pen_lst), 2))
Trate_mat_svr_feat_select = np.zeros((nexps, len(pen_lst), 2))
Trate_mat_duplicate = np.zeros((nexps, len(alpha_lst), 2))
Trate_mat_SMOTE = np.zeros((nexps, len(alpha_lst), 2))
Trate_mat_BSMOTE = np.zeros((nexps, len(alpha_lst), 2))
Trate_mat_ADASYN = np.zeros((nexps, len(alpha_lst), 2))
num_feats_n = np.zeros((max(len(alpha_lst),len(pen_lst)), 6))   ## records the number of featured used for: svr, svr_feat_select, duplicate, SMOTE, BSMOTE, ADASYN
num_feats_r = np.zeros((max(len(alpha_lst),len(pen_lst)), 6))
for i in range(nexps):
    output_now = outputs[i]
    Trate_mat_svr[i,:,:] = output_now[0]
    Trate_mat_svr_feat_select[i,:,:] = output_now[1]
    Trate_mat_duplicate[i,:,:] = output_now[2]
    Trate_mat_SMOTE[i,:,:] = output_now[3]
    Trate_mat_BSMOTE[i,:,:] = output_now[4]
    Trate_mat_ADASYN[i,:,:] = output_now[5]
    num_feats_n = num_feats_n + output_now[6]
    num_feats_r = num_feats_r + output_now[7]

Trate_lst_svr = np.mean(Trate_mat_svr, axis=0)
Trate_lst_svr_feat_select = np.mean(Trate_mat_svr_feat_select, axis=0)
Trate_lst_duplicate = np.mean(Trate_mat_duplicate, axis=0)
Trate_lst_SMOTE = np.mean(Trate_mat_SMOTE, axis=0)
Trate_lst_BSMOTE = np.mean(Trate_mat_BSMOTE, axis=0)
Trate_lst_ADASYN = np.mean(Trate_mat_ADASYN, axis=0)
num_feats_n = num_feats_n / nexps
num_feats_r = num_feats_r / nexps

'''select tuning parameter by the highest f-measure'''
select_ind = np.int_(np.zeros(6))
performances_mat = np.zeros((6,5))
n1ratio = sum(Yall) / len(Yall)

performances_now = performance_stats(Trate_lst_svr[:,0], Trate_lst_svr[:,1], n1ratio)
select_ind[0] = np.argmax(performances_now[:,3])
performances_mat[0,:] = performances_now[select_ind[0],:]

performances_now = performance_stats(Trate_lst_svr_feat_select[:,0], Trate_lst_svr_feat_select[:,1], n1ratio)
select_ind[1] = np.argmax(performances_now[:,3])
performances_mat[1,:] = performances_now[select_ind[1],:]

performances_now = performance_stats(Trate_lst_duplicate[:,0], Trate_lst_duplicate[:,1], n1ratio)
select_ind[2] = np.argmax(performances_now[:,3])
performances_mat[2,:] = performances_now[select_ind[2],:]

performances_now = performance_stats(Trate_lst_SMOTE[:,0], Trate_lst_SMOTE[:,1], n1ratio)
select_ind[3] = np.argmax(performances_now[:,3])
performances_mat[3,:] = performances_now[select_ind[3],:]

performances_now = performance_stats(Trate_lst_BSMOTE[:,0], Trate_lst_BSMOTE[:,1], n1ratio)
select_ind[4] = np.argmax(performances_now[:,3])
performances_mat[4,:] = performances_now[select_ind[4],:]

performances_now = performance_stats(Trate_lst_ADASYN[:,0], Trate_lst_ADASYN[:,1], n1ratio)
select_ind[5] = np.argmax(performances_now[:,3])
performances_mat[5,:] = performances_now[select_ind[5],:]

num_feats_select = np.zeros((6,2))
for k in range(6):
    num_feats_select[k,0] = num_feats_n[select_ind[k], k]
    num_feats_select[k,1] = num_feats_r[select_ind[k], k]

t3 = time.time()
print('tail time: '+str(t3-t2))

print(performances_mat)
print(num_feats_select)















    
    
    
    
    
    
'''Some records'''
# For pima data, weight=4, the best pen seems to lie in (17,70)*10**(-4), which is (0.017, 0.071)*n**(-1/3)
# For Satimage data, the recommended pen_lst is:
#    pen_lst = np.array([0, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024]) * 10**(-3) * len(Ytrain)**(-1/3)
# For future data sets, we recommend using
#    pen_lst = np.array([0, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024]) * 10**(-3) * len(Ytrain)**(-1/3)

# We recommend using c0 = 4. c0 between 1 and 10 are all reasonable choices. The results are not sensible to c0 values.
    
    
    
# Next step plan: set pen_lst as recommended above. For each data set, randomly select 2/3 samples as training, 1/3 as testing. Run the SVR-Tree 
# for all the pen in pen_lst on the training set, output test TPR, TNR. Repeat at least 10 times. Do the same thing for other CART based methods.

    
    
    
