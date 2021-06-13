 # -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:36:04 2019

This package defines the data structure tree and provides the fitting and testing of
CART and SVR-Tree methods.

@author: Yichen Zhu
"""

import numpy as np
import collections
import pdb


def data_standardize(X):
    ''' Function to linearly transfer feature matrix to [0,1]^d. '''
    n, d = np.shape(X) 
    border = np.zeros((d,2))
    for j in range(d):
        feat_min = min(X[:,j])
        feat_max = max(X[:,j])
        if feat_max == feat_min:
            raise Exception('feature '+str(j)+' has only one value')
        border_dist = (feat_max-feat_min)/(n-1)*1
        border[j,:] = [feat_min-border_dist, feat_max+border_dist]    
    shifts = - border[:,0]
    multipliers = np.diag(1/(border[:,1]-border[:,0]))
    return np.matmul(X + np.reshape(shifts, (1,d)), multipliers)


class node():
    def __init__(self):
        pass

class tree(node):
    def __init__(self):
        super().__init__()
        self.leaf = True
        self.class_label = None
        self.standardize_para = None
            
    def fit(self, X, Y, weight=1, criterion='gini', \
            min_split_weight=None, min_leaf_weight=None, feats_usage=None):       
        '''
        Function to Fit a CART.
        
        Parameters
        ----------
        X: ndarray of shape n \times d
            Features of data
        Y: ndarry or list of length n
            Response variable of data
        weight: float
            Weight for minority class samples. Should be no less than 1. Default value is 1.
        criterion: 'gini'
            Criterion for computing impurity. Currently only supports 'gini'.
        min_split_weight: float
            The minimal weight for a node to be further partitioned. If not provided, it will
            be the value of parameter "weight".
        min_leaf_weight: float
            The minimal weight of lead nodes. If not provided, the program will set it to be 1.
            
        Returns
        -------
        This function does not directly return any variables. The built tree can be printed by calling
        "self.print()". To predict new data with the built tree, refer to function "predict".        
        '''
        X = np.array(X)
        Y = np.array(Y)
        n, d = np.shape(X)
        self.d = d
        self.criterion = criterion
        if min_split_weight is None:
            self.min_split_weight = weight
        else:
            self.min_split_weight = min_split_weight
        if min_leaf_weight is None:
            self.min_leaf_weight = 1
        else:
            self.min_leaf_weight = min_leaf_weight
        self.weight = weight
        self.n, self.d = np.shape(X)           ## n: number of samples; d: number of features
        self.wn = len(Y) + (weight-1)*sum(Y)
        self.wy = weight*sum(Y)
        self.impu = self.Compute_Impu(self.wy, self.wn)
        self.impu_decr = -1
        if feats_usage is None:
            self.feats_usage = np.zeros(d, dtype=bool)
        ## check stop criterion
        if (self.wn < self.min_split_weight) or (self.wn == self.wy) or (self.wy == 0):
            self.leaf = True
            if self.wy/self.wn < 0.5:
                self.class_label = 0
            else:
                self.class_label = 1
            return
        splits = np.zeros((3, self.d))  ##featureid, splitthreshold, impurity decrease
        for i in range(self.d):
            splits[:,i] = np.concatenate((np.array([i]), self.findsplit(X[:,i], Y)))
        splitind = np.int_(splits[0, np.argmin(splits[2:,])])
        self.split = splits[0:2,splitind]
        self.impu_decr = self.impu - splits[2,splitind]
        if self.impu_decr <= 0: 
            self.leaf = True
            if self.wy/self.wn < 0.5:
                self.class_label = 0
            else:
                self.class_label = 1
            return            
        else:
            self.feats_usage[np.int_(self.split[0])] = True
            self.leaf = False
            leftind = np.flatnonzero(X[:,splitind]<=self.split[1])
            self.left = tree()
            self.left.fit(X[leftind,:], Y[leftind], weight, criterion, min_split_weight, min_leaf_weight)
            self.right = tree()
            self.right.fit(np.delete(X,leftind,0), np.delete(Y,leftind), weight, criterion, \
                           min_split_weight, min_leaf_weight)
            self.feats_usage = np.logical_or(np.logical_or(self.feats_usage, self.left.feats_usage), self.right.feats_usage)
            
    def findsplit(self, x, Y):   ## x is one dimension of X, of length n
        ''' Find the best split in one-dimensional feature x that minimizes impurity. '''
        self.wn = len(Y) + (self.weight-1)*sum(Y)
        self.wy = self.weight*sum(Y)
        wyleft = 0
        wi = 0
        impu = 2
        threshold = 0
        dat = np.core.records.fromarrays(np.array([x, Y]), names='feature, label')
        dat = np.sort(dat, order='feature')
        for i in range(self.min_leaf_weight-1, len(Y)-self.min_leaf_weight):
            wyleft = wyleft + self.weight*dat[i][1]
            wi = wi + 1 + (self.weight-1)*dat[i][1]
            if (dat[i+1][0] != dat[i][0]):
                impu_new = self.Compute_NodeImpu(wyleft, wi, self.wy, self.wn)
                if impu_new < impu:
                    threshold = (dat[i+1][0]+dat[i][0]) / 2
                    impu = impu_new
        return np.array([threshold, impu])     
        
    @staticmethod
    def Overlap_Rec(rec1, rec2):
        upmat = rec1.copy()
        upmat[:,0] = rec2[:,1]
        lowmat = rec1.copy()
        lowmat[:,1] = rec2[:,0]
        rec = rec1.copy()
        rec[:,0] = np.amax(lowmat, axis=1)
        rec[:,1] = np.amin(upmat, axis=1)
        return rec
    
    # @staticmethod
    # def piecewise_fun(rec, overlap, sub_overlap, V, S, epsilon=10**(-6)):
    #     d = np.shape(rec)[0]
    #     ans = [None]*d
    #     for j in range(d):
    #         sub_overlap_j = sub_overlap[j]
    #         slopes_changes = np.zeros((2*len(sub_overlap_j+1),2))
    #         sidelen_j = rec[j,1] - rec[j,0]
    #         for i in len(sub_overlap_j):
    #             slopes_changes[i+i,:] = [sub_overlap_j[i][0], -sub_overlap_j[i][2]]
    #             slopes_changes[i+i+1,:] = [sub_overlap_j[i][1], sub_overlap_j[i][2]]
    #         slopes_changes = slopes_changes[np.argsort(slopes_changes[:,0]),:]
    #         checkpoints = []
    #         slopes = []
    #         value = rec[j,0]
    #         slope_all = (S - V/sidelen_j) / sidelen_j
    #         sl = slope_all
    #         for k in range(len(slopes_changes)): 
    #             if np.abs(slopes_changes[k,0]-value) < epsilon:
    #                 sl += slopes_changes[k,1]
    #             else:
    #                 checkpoints.append(value)
    #                 slopes.append(sl)
    #                 if np.abs(slopes_changes[k,0]-rec[j,1]) < epsilon:
    #                     break
    #         slopes10 = 2*slope - slope_all
    #         intercepts10 = np.zero(len(checkpoints)) 
    #         intercepts10[0] = overlap[]
            
            
                
        
        
    
    def surface_funs(self, rec, label, reclst0, labellst0, epsilon=10**(-12)):  
        ''' Returns all the necessary parameters to compute the change of surface of the whole
        tree once a new partition at rec is made. Currently only working for d>=3.
        This function concerns all surfaces bordering and inside rec.
        '''
        ## Processing all overlapping cells 
        d = np.shape(rec)[0]
        V = np.prod(rec[:,1] - rec[:,0])
        S_faces = np.zeros(d)
        overlap = np.zeros((d, 2))   ## the overlapping surface between rec and other rectangles that are labeled 1, at two faces of feature j
        sub_overlap = [None]*d      ## sub_overlap is a list, with each element as [start, end, sub overlapping surface]
        for j in range(d):
            sub_overlap[j] = []
            S_faces[j] = V / (rec[j,1] - rec[j,0])
        S = np.sum(S_faces) * 2
        ans = [None]*(d+1)
        ## If reclst is empty:
        if len(labellst0) == 0:
            for j in range(d):
                intercepts10 = [2*S_faces[j]]
                slopes10 = [(S - S_faces[j]*2) / (rec[j,1] - rec[j,0])]
                ans[j] = ([rec[j,0]], slopes10, intercepts10, S_faces[j])  
            ans[d] = (0, S)
            return ans
            
        for i in range(len(labellst0)):
            if labellst0[i] == 0:
                continue
            recnow = reclst0[i]    
            contact_feat = -1
            for j in range(d):
                if rec[j,0] == recnow[j,1]:
                    contact_feat = j
                    contact_direct = 0
                    break
                elif rec[j,1] == recnow[j,0]:
                    contact_feat = j
                    contact_direct = 1 
                    break
            if contact_feat == -1:
                continue
            overlap_rec = self.Overlap_Rec(rec, recnow)
            overlap_rec_del = np.delete(overlap_rec, contact_feat, axis=0)
            if np.min(overlap_rec_del[:,1]-overlap_rec_del[:,0]) <= 0:
                continue
            overlap_V = np.prod(overlap_rec_del[:,1] - overlap_rec_del[:,0])
            overlap[contact_feat, contact_direct] += overlap_V
            feats = np.delete(np.arange(d), contact_feat)
            for j in feats:
                sub_overlap[j].append([overlap_rec[j,0], overlap_rec[j,1], overlap_V/(overlap_rec[j,1]-overlap_rec[j,0])])               
        
        ## Compute piecewise linear functions with overlapping information
        s_0 = np.sum(overlap)
        s_1 = S - s_0
        ans[d] = (s_0, s_1)
        for j in range(d):
            sub_overlap_j = sub_overlap[j]
            if len(sub_overlap_j) == 0:
                intercepts10 = [s_0 - overlap[j,0] + S_faces[j] - overlap[j,0] + S_faces[j]]
                slopes10 = [(S - S_faces[j]*2) / (rec[j,1] - rec[j,0])]
                ans[j] = ([rec[j,0]], slopes10, intercepts10, S_faces[j])   
                continue
            slopes_changes = np.zeros((2*len(sub_overlap_j),2))  ## both slopes_changes and slopes depicts slopes overlapping with elements of reclst with label 1
            sidelen_j = rec[j,1] - rec[j,0]
            for i in range(len(sub_overlap_j)):
                slopes_changes[i+i,:] = [sub_overlap_j[i][0], sub_overlap_j[i][2]]
                slopes_changes[i+i+1,:] = [sub_overlap_j[i][1], -sub_overlap_j[i][2]]
            slopes_changes = slopes_changes[np.argsort(slopes_changes[:,0]),:]
            checkpoints = []
            slopes = []
            value = rec[j,0]
            slope_all = (S - S_faces[j]*2) / sidelen_j
            sl = 0
            for k in range(len(slopes_changes)): 
                if np.abs(slopes_changes[k,0]-value) < epsilon:
                    sl += slopes_changes[k,1]
                else:
                    checkpoints.append(value)
                    value = slopes_changes[k,0]
                    slopes.append(sl)
                    sl += slopes_changes[k,1]
                    if np.abs(slopes_changes[k,0]-rec[j,1]) < epsilon:
                        break
            try:                
                if len(checkpoints) == 0:
                    intercepts10 = [s_0 - overlap[j,0] + S_faces[j] - overlap[j,0] + S_faces[j]]
                    slopes10 = [(S - S_faces[j]*2) / (rec[j,1] - rec[j,0])]
                    ans[j] = ([rec[j,0]], slopes10, intercepts10, S_faces[j])   
                    continue
                if np.abs(checkpoints[-1]-value) >= epsilon:
                    checkpoints.append(value)
                    slopes.append(sl)
            except:
                pdb.set_trace()
                debug = checkpoints
            slopes10 = slope_all - 2*np.array(slopes)
            intercepts10 = np.zeros(len(checkpoints)) 
            intercepts10[0] = s_0 - overlap[j,0] + S_faces[j] - overlap[j,0] + S_faces[j]
            for k in range(1,len(checkpoints)):
                if checkpoints[k] < checkpoints[k-1]:
                    print('Error: invalid checkpoints: '+str(checkpoints))
                intercepts10[k] = intercepts10[k-1] + slopes10[k-1]*(checkpoints[k]-checkpoints[k-1])
            ans[j] = (checkpoints, slopes10, intercepts10, S_faces[j])            
        return ans            
                
    def fit_sv(self, X, Y, pen, feature_select=False, c0=1, weight=1, border=None, standardize=False, 
               criterion='gini', min_split_weight=None, min_leaf_weight=None, tol=10**(-10), maximal_leaves=None):       
        '''
        Function to Fit a SVR-Tree.
        
        Parameters
        ----------
        X: ndarray of shape n \times d
            Features of data
        Y: ndarry or list of length n
            Response variable of data
        pen: float
            Penalty parameter of surface-to-volume ratio. We suggest to try values in the
            interval [0.001, 1]\times n^{-1/3}.
        feature_select: boolean
            Whether feature selection steps are enabled. Default value is False.
        c0: float
            c0 parameter is feature selections. If feature_select=False, c0 does not have
            any impacts on this function. Default value is 1.
        weight: float
            Weight for minority class samples. Should be no less than 1. Default value is 1.
        border: ndarray of shape d \times 2
            A hyperrectangle where the features of data lie in. If not provided, the program 
            will automatically compute one. If "border" is provided, "standardized" must by True.
        standardize: boolean
            Whether the features are already standardized. By saying standardized, it means the
            data is already transformed to lie in "border". It is recommanded that users do not 
            mannually input values for both "border" and "standardize", in which case the program
            will automatically pre-process the dataset.        
        criterion: 'gini'
            Criterion for computing impurity. Currently only supports 'gini'.
        min_split_weight: float
            The minimal weight for a node to be further partitioned. If not provided, it will
            be the value of parameter "weight".
        min_leaf_weight: float
            The minimal weight of lead nodes. If not provided, the program will set it to be 1.
        tol: float
            Tolerance for errors in comparison. Default is 10^(-5).
        maximal_leaves:
            Maximal number of leave nodes. If not provided, the program will run until no partition
            can be further accepted.
            
        Returns
        -------
        This function does not directly return any variables. The built tree can be printed by calling
        "self.print()". To predict new data with the built tree, refer to function "predict".        
        '''
        X = np.array(X)
        Y = np.array(Y)
        n, d = np.shape(X)           ## n: number of samples; d: number of features
        self.d = d
        if border == None:
            border = np.zeros((d,2))
            border[:,1] = 1
        if not standardize:
            X = self.data_standardize(X)
        if min_split_weight == None:
            min_split_weight = weight+1
        if min_leaf_weight == None:
            min_leaf_weight = 1
        if maximal_leaves == None:
            maximal_leaves = np.floor(np.sqrt(n))
        wn_all = len(Y) + (weight-1)*sum(Y)
        self.wn = wn_all
        self.wy = weight*sum(Y)
        self.impu = self.Compute_Impu(self.wy, self.wn)
        self.class_label = 1
        self.sign_impu = self.Compute_SignImpu(self.wy, self.wn, self.class_label)
        tree_impu = self.impu
        tree_sign_impu = self.sign_impu

        volume = np.prod(border[:,1] - border[:,0])
        surface = 0
        for j in range(d):
            surface += 2 * volume / (border[j,1]-border[j,0])
        sv_reg_min = self.sv_regular(surface, volume, d)
        risk = tree_impu + pen * sv_reg_min
        self.class_label = int(self.wy/self.wn>=0.5)
        self.rec = border
        self.X = X
        self.Y = Y
        node_que = collections.deque([self])   ## node_que is the queue that stores the nodes to operate, right side in and left side out
        rec_que = collections.deque([border]) 
        label_que = collections.deque([1])        
        reclst_leg = []
        labellst_leg = []
        feats_usage = np.zeros(d, dtype=bool)
        n_operate_nodes = 1

        while len(node_que) > 0 and n_operate_nodes < maximal_leaves:    ## Note surface, volume, tree_impu are attributes of a certain subtree (which contains root) rather than a node
            n_operate_nodes += 1
            node = node_que.popleft()
            rec = rec_que.popleft()
            reclst = list(rec_que)
            reclst.extend(reclst_leg)
            label = label_que.popleft()
            labellst = list(label_que)
            labellst.extend(labellst_leg)
            ans = self.surface_funs(rec, label, reclst, labellst)    ## ans contains information about changes of surface after partitions
            s_0, s_1 = ans[d]
            if label == 1:
                s_origin = s_1
            else:
                s_origin = s_0
            volume0 = volume - label * np.prod(rec[:,1] - rec[:,0])  ## The quantities subtitled by 0 remain unchanged through the next for loop
            # print(volume0, rec)
            if volume0 < -tol:          ## a bug-checking procedure
                pdb.set_trace()
                print('Negative volume0: '+str(volume0))
                raise Exception('Negative volume0: '+str(volume0))
            surface0 = surface
            tree_impu0 = tree_impu - node.impu * node.wn
            tree_sign_impu0 = tree_sign_impu - node.sign_impu * node.wn/wn_all
            
            featureid = -1         ## featureid=-1 means no better partition is found
            feats_reorder = np.append(np.flatnonzero(feats_usage), np.flatnonzero(1-feats_usage))
            node_impu_selected = node.impu
            S_faces = np.zeros(d)
            for j in feats_reorder:
                checkpoints, slope10, intercept10, S_faces[j] = ans[j]
                loc = 0          ## loc is the largest index of checkpoints that are no greater than thre
                wleft = 0
                wyleft = 0
                dat = np.core.records.fromarrays(np.array([node.X[:,j], node.Y]), names='feature, label')
                dat = np.sort(dat, order='feature')
                for sa in range(len(node.Y)-1):    ## sa is short for sample                        
                    wyleft = wyleft + weight*dat[sa][1]
                    wleft = wleft + 1 + (weight-1)*dat[sa][1]
                    try:
                        if wleft < min_leaf_weight or dat[sa][0]-rec[j,0] < tol:
                            pass
                        elif node.wn - wleft < min_leaf_weight or rec[j,1]-dat[sa+1][0]< tol:
                            pass
                    except:
                        pdb.set_trace()
                        print(dat[sa][0], dat[sa+1][0], rec[j,0], rec[j,1])
                    if wleft < min_leaf_weight or dat[sa][0]-rec[j,0] < tol:
                        continue
                    elif node.wn - wleft < min_leaf_weight or rec[j,1]-dat[sa+1][0]< tol:
                        break
                    if (dat[sa+1][0] != dat[sa][0]):
                        thre_new = (dat[sa+1][0]+dat[sa][0]) / 2
                        node_impu_new = self.Compute_NodeImpu(wyleft, wleft, node.wy, node.wn)
                        while loc < len(checkpoints)-1 and checkpoints[loc+1] <= thre_new:
                            loc += 1
                            
                        if feature_select:
                            if feats_usage[j]:
                                node_impu_selected = min(node_impu_selected, node_impu_new)
                            else:
                                if node_impu_selected-node_impu_new < c0*pen*wn_all/node.wn:
                                    continue
                        tree_impu_new = node_impu_new * node.wn / wn_all + tree_impu0

                        
                        tree_sign_impu_new_lst = [tree_sign_impu0]*4
                        surface_new_lst = [0,0,0,0]
                        volume_new_lst = [0,0,0,0]
                        risk_new_lst = [0,0,0,0]
                        child_labels_lst = [[1,1], [0,0], [0,1], [1,0]]
                        
                        '''If both child nodes are labeled 1'''
                        surface_new_lst[0] = surface0 + s_1 - s_origin
                        volume_new_lst[0] = np.prod(rec[:,1] - rec[:,0]) + volume0
                        tree_sign_impu_new_lst[0] = tree_sign_impu_new_lst[0] + node.wn / wn_all * self.Compute_SignNodeImpu(wyleft, wleft, node.wy, node.wn, [1,1])
                        if volume_new_lst[0] <= 0 or surface_new_lst[0] <= 0:
                            svr = sv_reg_min
                        else:
                            svr = self.sv_regular(surface_new_lst[0], volume_new_lst[0], d) 
                        risk_new_lst[0] = tree_sign_impu_new_lst[0] + pen*svr 
                        
                        '''If both child nodes are labeled 0'''
                        surface_new_lst[1] = surface0 + s_0 - s_origin
                        volume_new_lst[1] = volume0
                        tree_sign_impu_new_lst[1] = tree_sign_impu_new_lst[1] + node.wn / wn_all * self.Compute_SignNodeImpu(wyleft, wleft, node.wy, node.wn, [0,0])
                        if volume_new_lst[1] <= 0 or surface_new_lst[1] <= 0:
                            svr = sv_reg_min
                        else:
                            svr = self.sv_regular(surface_new_lst[1], volume_new_lst[1], d) 
                        risk_new_lst[1] = tree_sign_impu_new_lst[1] + pen*svr                    

                        '''If left child is labeled 0 and right child is labeled 1'''
                        surface_new_lst[2] = surface0 + s_0 + s_1 + 2*S_faces[j] - (intercept10[loc] + slope10[loc]*(thre_new-checkpoints[loc])) - s_origin
                        volume_new_lst[2] = volume0 + np.prod(np.delete(rec[:,1],j)-np.delete(rec[:,0],j)) * (rec[j,1]-thre_new)
                        tree_sign_impu_new_lst[2] = tree_sign_impu_new_lst[2] + node.wn / wn_all * self.Compute_SignNodeImpu(wyleft, wleft, node.wy, node.wn, [0,1])
                        if volume_new_lst[2] <= 0 or surface_new_lst[2] <= 0:
                            svr = sv_reg_min
                        else:
                            svr = self.sv_regular(surface_new_lst[2], volume_new_lst[2], d)
                        risk_new_lst[2] = tree_sign_impu_new_lst[2] + pen*svr 

                        '''If left child is labeled 1 and right child is labeled 0'''
                        surface_new_lst[3] = surface0 + intercept10[loc] + slope10[loc]*(thre_new-checkpoints[loc]) - s_origin
                        volume_new_lst[3] = volume0 + np.prod(np.delete(rec[:,1],j)-np.delete(rec[:,0],j)) * (thre_new-rec[j,0])
                        tree_sign_impu_new_lst[3] = tree_sign_impu_new_lst[3] + node.wn / wn_all * self.Compute_SignNodeImpu(wyleft, wleft, node.wy, node.wn, [1,0])
                        if volume_new_lst[3] <= 0 or surface_new_lst[3] <= 0:
                            svr = sv_reg_min
                        else:
                            svr = self.sv_regular(surface_new_lst[3], volume_new_lst[3], d)
                        risk_new_lst[3] = tree_sign_impu_new_lst[3] + pen*svr    
                        
                        argmin = np.argmin(risk_new_lst)
                        
                        if np.min(surface_new_lst) < - tol:
                            print('reclst:', reclst)
                            print('rec:', rec, 'len(reslst):', len(reclst))                            
                            print('slope10:', slope10, 'intercept10:', intercept10)
                            print('Negative surface: '+str(np.min(surface_new_lst))+'  pen: '+str(pen)+'  type: '+str(np.argmin(surface_new_lst)))
                            print('volume0:', volume0, 'surface0:', surface0)
                            print('featureid_now:', j, 'thre_now:', thre_new)
                            
                            pdb.set_trace()
                            raise Exception('Negative surface: '+str(np.min(surface_new_lst))+'  pen:'+str(pen))
                        if np.min(tree_sign_impu_new_lst) < - tol:
                            print('Negative tree signed impurity: '+str(np.min(tree_sign_impu_new_lst)))

                        if risk_new_lst[argmin] < risk:                           
                            thre = thre_new
                            featureid = j
                            child_labels = child_labels_lst[argmin]
                            surface = surface_new_lst[argmin]
                            volume = volume_new_lst[argmin]
                            tree_impu = tree_impu_new   
                            tree_sign_impu = tree_sign_impu_new_lst[argmin]
                            risk = risk_new_lst[argmin]
                            if risk < -tol:
                                print('Negative risk: '+str(risk)+'  pen:'+str(pen))
                                print('signed impu: '+str(tree_sign_impu))
                                print('volume: '+str(volume))
                                print('surface: '+str(surface))
                                pdb.set_trace()
                                raise Exception('Negative risk: '+str(risk)+'  pen:'+str(pen))

            if featureid >= 0:                     ## i.e., a better partition is found
                node.leaf = False
                feats_usage[featureid] = True
                node.split = [featureid, thre]
                node.left = tree()
                node.left.standardize_para = node.standardize_para
                leftind = np.flatnonzero(node.X[:,featureid]<=thre)
                node.left.X = node.X[leftind,]
                node.left.Y = node.Y[leftind]
                node.left.wn = len(node.left.Y) + (weight-1) * sum(node.left.Y)
                node.left.wy = weight * sum(node.left.Y)
                node.left.impu = self.Compute_Impu(node.left.wy, node.left.wn)
                node.left.class_label = child_labels[0]
                node.left.sign_impu = self.Compute_SignImpu(node.left.wy, node.left.wn, node.left.class_label)
                node.left.rec = rec.copy()
                node.left.rec[featureid,1] = thre
                if node.left.wy == 0 or node.left.wy == node.left.wn or node.left.wn < min_split_weight:
                    node.left.leaf = True
                    if node.left.class_label == 1:
                        reclst_leg.append(node.left.rec)
                        labellst_leg.append(1)
                else:
                    node_que.append(node.left)
                    rec_que.append(node.left.rec)
                    label_que.append(node.left.class_label)
                node.right = tree()
                node.right.standardize_para = node.standardize_para
                rightind = np.flatnonzero(node.X[:,featureid]>thre)
                node.right.X = node.X[rightind,]
                node.right.Y = node.Y[rightind]
                node.right.wn = len(node.right.Y) + (weight-1) * sum(node.right.Y)
                node.right.wy = weight * sum(node.right.Y)
                node.right.impu = self.Compute_Impu(node.right.wy, node.right.wn)
                node.right.class_label = child_labels[1]
                node.right.sign_impu = self.Compute_SignImpu(node.right.wy, node.right.wn, node.right.class_label)
                node.right.rec = rec.copy()
                node.right.rec[featureid,0] = thre
                node.right.rec[featureid,0] = thre
                if node.right.wy == 0 or node.right.wy == node.right.wn or node.right.wn < min_split_weight:
                    node.right.leaf = True
                    if node.right.class_label == 1:
                        reclst_leg.append(node.right.rec)
                        labellst_leg.append(1)
                else:
                    node_que.append(node.right)
                    rec_que.append(node.right.rec)
                    label_que.append(node.right.class_label)
            else:
                if node.class_label == 1:
                    reclst_leg.append(node.rec)
                    labellst_leg.append(1)
                    
        self.feats_usage = feats_usage
        return


    def data_standardize(self, X):
        ''' A function of class tree which linearly transfers feature matrix to [0,1]^d. '''
        n, d = np.shape(X) 
        border = np.zeros((d,2))
        for j in range(d):
            feat_min = min(X[:,j])
            feat_max = max(X[:,j])
            if feat_max == feat_min:
                raise Exception('feature '+str(j)+' has only one value')
            border_dist = (feat_max-feat_min)/(n-1)
            border[j,:] = [feat_min-border_dist, feat_max+border_dist]    
        shifts = - border[:,0]
        multipliers = np.diag(1/(border[:,1]-border[:,0]))
        self.standardize_para = (shifts, multipliers)
        return np.matmul(X + np.reshape(shifts, (1,d)), multipliers)  
                    
    @staticmethod
    def sv_regular(surface, volume, d):
        ''' Compute surface-to-volume regularization. '''
        return surface/volume
    
    @staticmethod
    def Compute_Impu(wy, w, criterion='gini'):
        ''' Compute impurity of a node. '''
        return 1 - (wy/w)**2 - ((w-wy)/w)**2 
        
    @staticmethod
    def Compute_SignImpu(wy, w, label, criterion='gini'):
        ''' Compute signed impurity of a node. '''
        if int(wy/w>=0.5) == label:
            return 1 - (wy/w)**2 - ((w-wy)/w)**2
        else:
            return (wy/w)**2 + ((w-wy)/w)**2
        
    @staticmethod
    def Compute_NodeImpu(wyleft, wleft, wy, w, criterion='gini'):
        ''' Compute impurity of a node after a partition. '''
        return 1 - ((wyleft/wleft)**2 + ((wleft-wyleft)/wleft)**2)*wleft/w \
                - (((wy-wyleft)/(w-wleft))**2 + ((w-wleft-wy+wyleft)/(w-wleft))**2)*(w-wleft)/w 
    
    @staticmethod
    def Compute_SignNodeImpu(wyleft, wleft, wy, w, child_labels, criterion='gini'):
        ''' Compute signed impurity of a node after a partition. '''
        impu_left = 1 - (wyleft/wleft)**2 - ((wleft-wyleft)/wleft)**2
        impu_right = 1 - ((wy-wyleft)/(w-wleft))**2 - ((w-wleft-wy+wyleft)/(w-wleft))**2
        if int(wyleft/wleft>=0.5) == child_labels[0]:
            impu_left_sign = impu_left
        else:
            impu_left_sign = 1 - impu_left
        if int((wy-wyleft)/(w-wleft)>=0.5) == child_labels[1]:
            impu_right_sign = impu_right
        else:
            impu_right_sign = 1 - impu_right
        return impu_left_sign*wleft/w + impu_right_sign*(w-wleft)/w  
    
                
    def predict(self, X):    
        '''
        This function return predict class labels for a new data using the tree "self".
        
        Parameters
        ----------
        X: ndarray
            Feature matrix of new data. Must has the same number of features as 
            the training data.
        
        Returns
        -------
        var: ndarray
            One-dimensional array contains the predicted class labels of new data.
        '''
        X = np.array(X)
        d = np.shape(X)[1]
        if not self.standardize_para == None:
            shifts, multipliers = self.standardize_para
            X = np.matmul(X + np.reshape(shifts, (1,d)), multipliers)
        return self.localpredict(X)
    
    def localpredict(self, X): 
        ''' This recursive functions is called by function "predict" to complete 
        its taks of predicting class labels. '''
        if self.leaf:
            return self.class_label * np.ones(np.shape(X)[0],dtype=int)
        else:
            Y = np.zeros(np.shape(X)[0],dtype=int)
            featureid, thre = self.split
            featureid = np.int_(featureid)
            leftind = np.flatnonzero(X[:,featureid]<=thre)
            Y[leftind] = self.left.localpredict(X[leftind,:])
            rightind = np.flatnonzero(X[:,featureid]>thre)
            Y[rightind] = self.right.localpredict(X[rightind,:])
            return Y
            
    def compute_feats_usage(self):
        '''
        This function checks whether each feature is used in partitions. It does 
        not take any input parameters except "self".
        
        Returns
        -------
        var: ndarray
            One-dimensional array of length equal to the number of features used
            for training. The ith element of this array is True if the ith feature
            is used for partitions and False otherwise.
        '''
        feats_usage = np.zeros(self.d, dtype=bool)
        self.feats_usage = self.local_feats_usage(feats_usage) 
    
    def local_feats_usage(self, feats_usage):
        ''' This recursive is called by "compute_feats_usage" to check whether each
        feature is used for partitions. '''
        if self.leaf:
            return feats_usage
        else:
            featureid = np.int_(self.split[0])
            feats_usage[featureid] = True
            feats_usage = self.left.local_feats_usage(feats_usage)
            feats_usage = self.right.local_feats_usage(feats_usage)
            return feats_usage
    
    def print(self, init=True, print_weight=False, print_impu=False):
        '''
        This function print a tree.
        
        Parameters
        ----------
        init: boolean
            Whether the printing is started from root node. If not called by the 
            the function "print" itself, it should always set to be True. Default 
            value is True.
        print_weight: boolean
            Whether to print the weight of training samples in each node. Default
            is False.
        print_impu: boolean
            Whether to print the impurity of training samples in each node. Default
            is False.
        
        Returns
        -------
        This function returns nothing.
        
        Outputs
        -------
        This function will print all the nodes of the tree in a depth-first order.
        '''
        if init:
            self.codename = 'root'
        if self.leaf:
            print(self.codename+':', self.class_label)
            if print_weight:
                print('class 1 weight, total weight:', self.wy, self.wn)
            if print_impu:
                print('impurity:', self.impu)                
        else:
            print(self.codename+':', 'feature '+str(self.split[0])+' <= '+str(self.split[1]))
            if print_weight:
                print('class 1 weight and total weight:', self.wy, self.wn)
            if print_impu:
                print('impurity, impurity_decr:', self.impu, self.impu_decr, self.tot_impudecr, self.alpha)
            self.left.codename = self.codename + '.left'
            self.left.print(False, print_weight, print_impu)
            self.right.codename = self.codename + '.right'
            self.right.print(False, print_weight, print_impu)
         
            
    def copy(self):
        ''' Copy the current tree represented by "self". '''
        copytr = tree()
        copytr.leaf = self.leaf
        copytr.impu = self.impu
        copytr.wn = self.wn
        copytr.wy = self.wy
        if self.leaf:
            copytr.class_label = self.class_label
        else:
            copytr.split = self.split
            copytr.impu_decr = self.impu_decr
            copytr.left = self.left.copy()
            copytr.right = self.right.copy()
        return copytr
        
    def Find_prune(self, totalweight, alpha_min=np.inf, prunetree=[]):
        ''' This function called by "prune" to complete its task of pruning a tree. '''
        if self.leaf:
            self.tot_impudecr = 0
            self.tot_leaf = 1
            return (alpha_min, prunetree)
        else:
            alpha_min, prunetree = self.left.Find_prune(totalweight, alpha_min, prunetree)
            alpha_min, prunetree = self.right.Find_prune(totalweight, alpha_min, prunetree)
            left_prop = self.left.wn / self.wn
            right_prop = self.right.wn / self.wn
            self.tot_impudecr = self.left.tot_impudecr * left_prop + self.right.tot_impudecr * right_prop \
                                    + self.impu_decr
            self.tot_leaf = self.left.tot_leaf + self.right.tot_leaf
            self.alpha = self.tot_impudecr * self.wn / (totalweight * (self.tot_leaf-1))
            if self.alpha < alpha_min:
                alpha_min = self.alpha
                prunetree = [self]
            elif self.alpha == alpha_min:
                prunetree.append(self)
            return (alpha_min, prunetree)
    
    def Prune(self):
        '''
        This function prunes a CART. Users should not use this function to prune
        a SVR-Tree, despite doing that does not yield an error in codes.
        
        Parameters
        ----------
        This function takes no parameters except "self".
        
        Returns
        -------
        var=(treelst, alphalst), tot_leaf_lst): tuple of length 3. each element
        is explained as below:
            treelst: a list of possible optimal CART tree after pruning.
            alphalst: an unidimensional ndarry of the same length as "treelst". Each
                element represents the alpha value when the corresponding element 
                in "treelst" is the optimally pruned CART tree.
            tot_leaf_lst: a list of the same length as "treelst". Each element contains
                the number of leaf nodes of the corresponding element in "treelst".
        '''
        treelst = [self]
        alphalst = [0]
        tot_leaf_lst = []
        tr = self
        while tr.leaf == False:
            tr_next = tr.copy()
            tr_next.d = self.d
            alpha, prunetree = tr_next.Find_prune(totalweight=self.wn)
            alphalst.append(alpha)
            tot_leaf_lst.append(tr_next.tot_leaf)       ## The total leaf value lags one iteration 
            for i in prunetree:
                i.leaf = True
                i.left = None
                i.right = None
                i.class_label = int(i.wy/i.wn >= 0.5)
            treelst.append(tr_next)
            tr = tr_next
        tot_leaf_lst.append(1)                ## a tree with only root node has total leaf 1
        return (treelst, np.array(alphalst), tot_leaf_lst)

        
        
        

            
        
        
        
        
        
        
        
        
        
        
        
    

        
