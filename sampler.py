# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:04:47 2019

This package perform oversampling on datasets. Supported methods are:
Duplicated oversampling, SMOTE, Borderline-SMOTE, ADASYN.

@author: Yichen Zhu
"""
import numpy as np

class sampler():
    def __init__(self, times=1, method='duplicate', mBSMOTE=10):
        '''
        Initiation function for sampler class.
        
        Parameters
        ----------
        times: integer
            Number of times the original data will be resampled. Default value is 1.
        method: 'duplicate' or 'SMOTE' or 'BSMOTE' or 'ADASYN'.
            Which method is used for resampling. Default value is 'duplicate'.        
        '''
        self.times = int(times)
        self.method = method
        self.mBSMOTE = mBSMOTE
    
    @staticmethod
    def neighbor_ind(X, s):
        '''Find the s-nearest neighbors of each sample, return with a n\times s 
        matrix containing indices of nearest neighbors.
        '''
        n = np.shape(X)[0]
        Dmat = np.zeros((n, n))
        for i in range(n-1):
            for j in range(i+1,n):
                Dmat[i,j] = np.dot(X[i,:]-X[j,:], X[i,:]-X[j,:])
                Dmat[j,i] = Dmat[i,j]
        E = np.zeros((n, s), dtype=int)
        for i in range(n):
            di = np.core.records.fromarrays(np.array([Dmat[i,:], np.array(range(n))]), names='distance, index')
            di = np.sort(di, order='distance')  
            E[i,:] = di['index'][0:s]
        return E  
    
    @staticmethod
    def neighbor_ind_half(X, Xplus, s):
        '''Find the s-nearest neighbors from Xplus for each sample in X, return 
        with a n\times s matrix containing indices of nearest neighbors.
        '''
        n1 = np.shape(X)[0]
        n = np.shape(Xplus)[0]
        Dmat = np.zeros((n1, n))
        for i in range(n1):
            for j in range(n):
                Dmat[i,j] = np.dot(X[i,:]-Xplus[j,:], X[i,:]-Xplus[j,:])
        E = np.zeros((n1, s), dtype=int)
        for i in range(n1):
            di = np.core.records.fromarrays(np.array([Dmat[i,:], np.array(range(n))]), names='distance, index')
            di = np.sort(di, order='distance')  
            E[i,:] = di['index'][0:s]
        return E  
        
    def fit_resample(self, X, Y):
        '''
        Main function to return resampled data. For how many times the dataset will be 
        resampled and which method is used for resampled, see parameters of initiation
        function.
        
        Parameters
        ----------
        X: ndarray of shape n \times d
            The features of orignial data.
        Y: ndarry or list of length n
            The outcome variable of original data.
            
        Returns
        -------
        val: ndarray
            The features of data after oversampling.
        '''
        if self.times <= 0:
            return (X, Y)
        n, d = np.shape(X)
        c1_ind = np.flatnonzero(Y)
        n1 = len(c1_ind)
        X1 = X[c1_ind, :]
        if self.method == 'duplicate':
            X1_res = np.zeros((n1*self.times, d))
            for ti in np.arange(self.times):
                X1_res[(ti*n1):((ti+1)*n1),:] = X1.copy()
            X_res = np.vstack((X, X1_res))
            Y_res = np.append(Y, np.ones(n1*self.times))
        elif self.method == 'SMOTE':
            if self.times <= 5:
                E = self.neighbor_ind(X1, 5)    
                X1_res = np.zeros((n1*self.times, d))
                for i in range(n1):
                    X_ne = X1[E[i,np.random.choice(5,self.times,replace=False)],:]
                    unif_r = np.random.random(self.times)
                    X_syn = np.dot(np.diag(unif_r), X_ne-X1[i,:]) + X1[i,:]
                    X1_res[(i*self.times):((i+1)*self.times),:] = X_syn
            else:
                E = self.neighbor_ind(X1, 5)    
                X1_res = np.zeros((n1*self.times, d))
                for i in range(n1):
                    X_ne = X1[E[i,np.random.choice(5,self.times,replace=True)],:]
                    unif_r = np.random.random(self.times)
                    X_syn = np.dot(np.diag(unif_r), X_ne-X1[i,:]) + X1[i,:]
                    X1_res[(i*self.times):((i+1)*self.times),:] = X_syn
            X_res = np.vstack((X, X1_res))
            Y_res = np.append(Y, np.ones(n1*self.times))
        elif self.method == 'BSMOTE':
            m = self.mBSMOTE
            Eall = self.neighbor_ind_half(X1, X, m)
            minor_num = np.zeros(n1)
            for i in range(n1):
                minor_num[i] = int(sum(Y[Eall[i,:]]))
            ss = 0
            for j in range(m):
                ss = ss + len(np.flatnonzero(minor_num == j))
                if ss >= n1 / 2:
                    break
            c1_border_ind = np.flatnonzero(minor_num <= j)
            n1_border = len(c1_border_ind)
            X1_border = X1[c1_border_ind]
            if self.times <= 5:
                E = self.neighbor_ind_half(X1_border, X1, 5)
                X1_res = np.zeros((n1_border*self.times, d))
                for i in range(n1_border):
                    X_ne = X1[E[i,np.random.choice(5,self.times,replace=False)],:]
                    x = X1_border[i,:]
                    unif_r = np.random.random(self.times)
                    X_syn = np.dot(np.diag(unif_r), X_ne-x) + x
                    X1_res[(i*self.times):((i+1)*self.times),:] = X_syn 
            else:
                E = self.neighbor_ind_half(X1_border, X1, 5)
                X1_res = np.zeros((n1_border*self.times, d))
                for i in range(n1_border):
                    X_ne = X1[E[i,np.random.choice(5,self.times,replace=True)],:]
                    x = X1_border[i,:]
                    unif_r = np.random.random(self.times)
                    X_syn = np.dot(np.diag(unif_r), X_ne-x) + x
                    X1_res[(i*self.times):((i+1)*self.times),:] = X_syn                
            X_res = np.vstack((X, X1_res))
            Y_res = np.append(Y, np.ones(n1_border*self.times))
        elif self.method == 'ADASYN':
            m = 5
            Eall = self.neighbor_ind_half(X1, X, m)
            major_rate = np.zeros(n1)
            for i in range(n1):
                major_rate[i] = 1 - sum(Y[Eall[i,:]])/m
            major_rate = major_rate / np.average(major_rate)
            n1res_lst = np.rint(major_rate * self.times)
            n1res_lst = n1res_lst.astype(int)
            n1_res = sum(n1res_lst)
            X1_res = np.zeros((n1_res, d))
            s = 5
            E = self.neighbor_ind(X1, s)
            isum = 0
            for i in range(n1):
                if n1res_lst[i] >= 1:
                    X_ne = X1[E[i,np.random.choice(5,n1res_lst[i],replace=True)],:]
                    unif_r = np.random.random(n1res_lst[i])
                    X_syn = np.dot(np.diag(unif_r), X_ne-X1[i,:]) + X1[i,:]
                    X1_res[isum:(isum+n1res_lst[i]),:] = X_syn 
                    isum = isum + n1res_lst[i]
            X_res = np.vstack((X, X1_res))
            Y_res = np.append(Y, np.ones(n1_res))
        else:
            raise ValueError('There is no such a method.')
            
        return (X_res, Y_res)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
  