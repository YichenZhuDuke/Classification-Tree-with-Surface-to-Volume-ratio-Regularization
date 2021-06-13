# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:48:54 2021

@author: acezy
"""
import os
import numpy as np
import pandas as pd
import arff 
import subprocess
import Tree



#%% read data and save as arff file
''' Reading data sets. Please choose data_name from: 
    'vechicle', 'pima', 'abalone', 'satimage', 'wine', ''page', 'yeast', 'segment', 'ecoli', 'glass2', 'phoneme', 'titanic'   '''

## KEEL datasets used:
## page-blocks0, yeast4, shuttle-c0-vs-c4, segment0, vowel0
    
    
    
data_name = 'ecoli'

if data_name == 'vehicle':
    '''vehicle data'''
    da = [None] * 9
    filenames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    for i in range(9):
        da[i] = pd.read_table('D:/Research/AUC_IB/data/xa'+filenames[i]+'.dat', sep=' ', header=None, error_bad_lines=False, \
                              keep_default_na=False)    ## read file in windows
    #    da[i] = pd.read_table('/home/grad/yz486/ImbalanceData/data/xa'+filenames[i]+'.dat', sep=' ', header=None, error_bad_lines=False, \
    #                          keep_default_na=False)    ## read file in linux
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
    da_pima = pd.read_csv('D:/Research/AUC_IB/data/diabetes.csv')
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
    da_wine = pd.read_csv('D:/Research/AUC_IB/data/winequality-red.csv', sep=';')
    da = da_wine.values
    Xall = da[:,0:11]
    Yall = np.zeros(np.shape(Xall)[0])
    Yall[np.flatnonzero(da[:,11]>=7)] = 1
    n, d = np.shape(Xall)
    times = 5
elif data_name == 'abalone':
    '''Abalone data'''
    da_abalone = pd.read_csv('D:/Research/AUC_IB/data/abalone.data', sep=',')
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
    da_a = pd.read_table('D:/Research/AUC_IB/data/sat_train.txt', sep=' ', header=None, error_bad_lines=False, \
                              keep_default_na=False)
    da_a = da_a.values
    Xa = da_a[:,0:36]
    Ya = da_a[:,36]
    Ya = np.int_(Ya==4)
    da_b = pd.read_table('D:/Research/AUC_IB/data/sat_test.txt', sep=' ', header=None, error_bad_lines=False, \
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
    da0 = pd.read_table('D:/Research/AUC_IB/data/page-blocks0.dat', sep=',', header=None)
    da = da0.values
    Xall = da[:,0:10]
    n1_ind = np.flatnonzero(da[:,10] == ' positive')
    Yall = np.zeros(np.shape(da)[0],dtype=int)
    Yall[n1_ind] = 1
    n, d = np.shape(Xall)
    times = int((n-len(n1_ind))/len(n1_ind)) - 1
elif data_name == 'yeast':
    da0 = pd.read_table('D:/Research/AUC_IB/data/yeast4.dat', sep=',', header=None)
    da = da0.values
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' positive')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1        
elif data_name == 'segment':
    da0 = pd.read_table('D:/Research/AUC_IB/data/segment0.dat', sep=',', header=None)
    da = np.delete(da0.values, 2, axis=1)      ## remove the second feature as it contains no information
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' positive')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1        
elif data_name == 'ecoli':
    da0 = pd.read_table('D:/Research/AUC_IB/data/ecoli.dat', sep=',', header=None)
    da = np.delete(da0.values, 3, axis=1)       ## remove the third feature as it contains little information 
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' pp')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1           
elif data_name == 'glass2':
    da0 = pd.read_table('D:/Research/AUC_IB/data/glass2.dat', sep=',', header=None)
    da = da0.values      
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d] == ' positive')
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1     
elif data_name == 'phoneme':
    da0 = pd.read_table('D:/Research/AUC_IB/data/phoneme.dat', sep=',', header=None)
    da = da0.values      
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d])
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1           
elif data_name == 'titanic':
    da0 = pd.read_table('D:/Research/AUC_IB/data/titanic.dat', sep=',', header=None)
    da = da0.values      
    n, d = np.shape(da)
    d = d-1
    Xall = da[:,0:d]
    n1_ind = np.flatnonzero(da[:,d]==1)
    Yall = np.zeros(n,dtype=int)
    Yall[n1_ind] = 1
    times = int((n-len(n1_ind))/len(n1_ind)) - 1     

#%%   
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

def HD_experiment_runner(ids, Xall, Yall, n_cv, n_cv_out, criterion='1'):
    X = Xall[ids,:]
    Y = Yall[ids]
    n = len(Y)
    n1 = np.int_(np.sum(Y))
    n0 = n - n1
    n1_divide = divide(n1, n_cv_out)
    n0_divide = divide(n0, n_cv_out)
    id_1 = np.flatnonzero(Y)            ## all variables starting with "id" are indices of X and Y (not Xall or Yall)
    id_0 = np.flatnonzero(Y==0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
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
        
        ntrain = n1train+n0train
        ntest = n1test+n0test
        Datatrain = np.concatenate((Xtrain, np.reshape(Ytrain,(ntrain,1))), axis=1)
        Datatrain = Datatrain.tolist()
        for i in range(ntrain):
            Datatrain[i][d] = int(Datatrain[i][d])
        names = [None]*(d+1)
        for i in range(d):
            names[i] = ('x'+str(i), 'REAL')
        names[d] = ('response',['0','1'])
        obj = {'description': 'None',
           'relation': 'weather',
           'attributes': names,
           'data': Datatrain}
        dumpstr = arff.dumps(obj)
        trainfile = open('train.arff', 'w')
        trainfile.write(dumpstr)
        trainfile.close()        
        Datatest = np.concatenate((Xtest, np.reshape(Ytest,(ntest,1))), axis=1)
        Datatest = Datatest.tolist()
        for i in range(ntest):
            Datatest[i][d] = int(Datatest[i][d])
        names = [None]*(d+1)
        for i in range(d):
            names[i] = ('x'+str(i), 'REAL')
        names[d] = ('response',['0','1'])
        obj = {'description': 'None',
           'relation': 'weather',
           'attributes': names,
           'data': Datatest}
        dumpstr = arff.dumps(obj)
        testfile = open('test.arff', 'w')
        testfile.write(dumpstr)
        testfile.close()
        
        command = ['java', '-cp', 'D:\ProgramFiles\Weka-3-8-5\weka.jar', 'weka.classifiers.trees.HoeffdingTree', \
                   '-L', criterion, \
                   '-t', 'D:/Research/AUC_IB/code2021/train.arff', \
                   '-T', 'D:/Research/AUC_IB/code2021/test.arff', '-o']
        HDobj = subprocess.run(command, stdout=subprocess.PIPE)
        out = HDobj.stdout.decode()
        outs = out.split()
        TP += int(outs[-5])
        FN += int(outs[-6])
        FP += int(outs[-11])
        TN += int(outs[-12])
    
    results_HD = performance_stats(TP, FP, FN, TN)
    return results_HD




#%%
seednum = 40
id_filename = data_name+'_seed'+str(seednum)+'_ids.npy'
id_permutes = np.load('D:/Research/AUC_IB/code2021/'+id_filename)

nexps = np.shape(id_permutes)[0]
n_cv_out = 3
n_cv = 5
os.chdir('D:/Research/AUC_IB/code2021/')

results_mat_HD = np.zeros((nexps, 6))
for i in range(nexps):
    ids = id_permutes[i,:]
    results_mat_HD[i,:] = HD_experiment_runner(ids, Xall, Yall, n_cv, n_cv_out, criterion='1')

if np.mean(results_mat_HD[:,5]) < 10**(-2):
    print('zero positive or zero negative under naive bayes criterion')
    for i in range(nexps):
        ids = id_permutes[i,:]
        results_mat_HD[i,:] = HD_experiment_runner(ids, Xall, Yall, n_cv, n_cv_out, criterion='0')
    if np.mean(results_mat_HD[:,5]) < 10**(-2):
        print('zero positive or zero negative under majority voting')
        for i in range(nexps):
            ids = id_permutes[i,:]
            results_mat_HD[i,:] = HD_experiment_runner(ids, Xall, Yall, n_cv, n_cv_out, criterion='0')  
        
mean_HD = np.mean(results_mat_HD, axis=0)
std_HD = np.std(results_mat_HD, axis=0)    
    
print('HDDT for '+str(data_name))
print(mean_HD)
print(std_HD)

mean_filename = data_name+'_seed'+str(seednum)+'_mean.npy'
mean_load = np.load('D:/Research/AUC_IB/code2021/'+mean_filename)
mean_all = np.zeros((7,6))
mean_all[0:6,:] = mean_load
mean_all[6,:] = mean_HD

std_filename = data_name+'_seed'+str(seednum)+'_std.npy'
std_load = np.load('D:/Research/AUC_IB/code2021/'+std_filename)
std_all = np.zeros((7,6))
std_all[0:6,:] = std_load
std_all[6,:] = std_HD

mean_all_filename = data_name+'_seed'+str(seednum)+'_mean_all'
np.save(mean_all_filename, mean_all)
std_all_filename = data_name+'_seed'+str(seednum)+'_std_all'
np.save(std_all_filename, std_all)

ranks_all = np.zeros((7,6),dtype=int)
for i in range(6):
    sort_args = np.argsort(mean_all[:,i])
    rank = 7
    for j in sort_args:
        ranks_all[j,i] = rank
        rank -= 1
ranks_all_filename = data_name+'_seed'+str(seednum)+'_ranks_all'
np.save(ranks_all_filename, ranks_all)



        
    


#%% test range of alpha in CART
# tr_cart = Tree.tree()
# tr_cart.fit(Xall, Yall)
# treelst, alpha_prune_lst, tot_leaf_lst = tr_cart.Prune()
# print(alpha_prune_lst)
# n_cv_out = 3
# train_ratio = 1 - 1/n_cv_out
# alpha_lst = np.array([0, 1/256, 1/128, 1/64, 1/32, 1/16, 0.125, 0.177, 0.25, 0.35, 0.5, 0.71, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 16, 22, 32, 44, 64, 89, 128, 179, 256, 358, 512, 716, 1024, 1450, 2048, 2896, 4096]) * 10**(-3) * (n*train_ratio)**(-1/3)
# print(alpha_lst)

    
#%% Compute overall average rankings
# data_name_lst = ['pima', 'vehicle', 'segment', 'wine', 'satimage', 'glass2', 'abalone', 'yeast', 'titanic', 'ecoli', 'page', 'phoneme']
# k = len(data_name_lst)
# ranks_tensor = np.zeros((7,6,k))
# for kk in range(k):
#     data_name_load = data_name_lst[kk]
#     ranks_tensor[:,:,kk] = np.load(data_name_load+'_seed'+str(seednum)+'_ranks_all.npy')
# ranks_tensor[2,:,8] = ranks_tensor[2,:,8]-0.5 ## same tier correction
# ranks_tensor[3,:,8] = ranks_tensor[3,:,8]+0.5
# ranks_mean = np.zeros((7,6))
# for i in range(7):
#     for j in range(6):
#         ranks_mean[i,j] = np.mean(ranks_tensor[i,j,:])
# print(ranks_mean)
        
    
    
    