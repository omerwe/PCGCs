from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import sys
import random
import time
import os
import os.path
import pandas as pd
import itertools
np.set_printoptions(precision=4, linewidth=200)
from sklearn.linear_model import LinearRegression, LogisticRegression
import pcgcs_utils
from pcgcs_utils import print_memory_usage


# def print_memory():
    # import psutil
    # process = psutil.Process(os.getpid())
    # print 'memory usage:', process.memory_info().rss
    
    
def print_sumstats(cov1, u1_0, u1_1, var_t1, cov2, u2_0=None, u2_1=None, var_t2=None, s1=None, sum_s1=None, sum_s1_sqr=None, s2=None, sum_s2=None, sum_s2_sqr=None):
    if (cov1 is None and cov2 is None and sum_s1 is None and sum_s2 is None): return
    print
    print
    print 'summary statistics for subsequent estimation:'
    print '-----------------------------------'
        
    if (cov1 is not None):
        print 'mean Q for study 1 (mean_Q1): %0.4f'%(np.mean((u1_0 + u1_1)**2))
        print 'liability variance explained by covariates (var_t1): %0.4f'%(var_t1)

    if (cov2 is not None):
        print 'mean Q for study 2 (mean_Q2): %0.4f'%(np.mean((u2_0 + u2_1)**2))
        print 'liability variance explained by covariates (var_t2): %0.4f'%(var_t2)
        
        
    if (sum_s1 is not None):
        print 'study 1 genotypes deflation factor (geno1_factor): %0.6f'%((sum_s1 - s1.sum()) / sum_s1)
        print 'study 1 squared genotypes deflation factor  (sqr_geno1_factor): %0.6f'%((sum_s1_sqr - np.sum(s1**2)) / sum_s1_sqr)
        
    if (sum_s2 is not None):
        print 'study 2 genotypes deflation factor (geno2_factor): %0.6f'%((sum_s2 - s2.sum()) / sum_s2)
        print 'study 2 squared genotypes deflation factor  (sqr_geno2_factor): %0.6f'%((sum_s2_sqr - np.sum(s2**2)) / sum_s2_sqr)
        
    print
    print
        

def pcgc_jackknife_sig2g(X, y, numer_sig2g, denom_sig2g, pcgc_coeff=1.0, u0=None, u1=None, window_size=1000):
    
    if (u0 is not None):
        u_sqr = (u0 + u1)**2
        qy = y * (u0+u1)
    else:
        qy = y
        
    if (window_size is None or window_size<0): window_size = X.shape[0]
    
    estimators_arr = np.empty(X.shape[0])
    for i in xrange(0, X.shape[0], window_size):
        X_i = X[i:i+window_size]
        G_i = X_i.dot(X.T) / X.shape[1]
        indices0 = np.arange(G_i.shape[0])
        G_i[indices0, i+indices0]=0
        
        for j in xrange(G_i.shape[0]):
            numer_sig2g_i = numer_sig2g - 2*G_i[j].dot(qy[i+j]*qy)
            if (u0 is None): denom_sig2g_i = denom_sig2g - 2*G_i[j].dot(G_i[j])
            else: denom_sig2g_i = denom_sig2g - 2*G_i[j].dot(G_i[j] * u_sqr[i+j]*u_sqr)         
            estimators_arr[i+j] = numer_sig2g_i / denom_sig2g_i
                   
    estimators_arr /= pcgc_coeff
    sig2g_var = (X.shape[0]-1)/float(X.shape[0]) * np.sum((estimators_arr - estimators_arr.mean())**2)
    return np.sqrt(sig2g_var)
        
    
    

def pcgc_jackknife_corr(X1, X2, y1, y2,
    numer_sig2g1, denom_sig2g1, numer_sig2g2, denom_sig2g2, numer_rho, denom_rho,
    pcgc_coeff1=1.0, pcgc_coeff2=1.0, pcgc_coeff12=1.0, 
    u1_0=None, u1_1=None, u2_0=None, u2_1=None,
    is_same=None, window_size=1000):
    
    if (window_size is None or window_size<0): window_size = X1.shape[0]
    
    if (u1_0 is not None):
        u1_sqr = (u1_0 + u1_1)**2
        u2_sqr = (u2_0 + u2_1)**2
        qy1 = y1 * (u1_0+u1_1)
        qy2 = y2 * (u2_0+u2_1)
    else:
        qy1 = y1
        qy2 = y2
    
    sig2g1_estimators_arr = np.empty(X1.shape[0])
    sig2g2_estimators_arr = np.empty(X2.shape[0])
    rho_estimators_arr = np.empty(X1.shape[0] + X2.shape[0])
    
    
    #exclude individuals from study 1
    for i in xrange(0, X1.shape[0], window_size):
        X1_i = X1[i:i+window_size]
        G_i = X1_i.dot(X1.T) / X1.shape[1]
        indices0 = np.arange(G_i.shape[0])
        G_i[indices0, i+indices0]=0
        
        for j in xrange(G_i.shape[0]):
            numer_sig2g1_i = numer_sig2g1 - 2*G_i[j].dot(qy1[i+j]*qy1)
            if (u1_0 is None): denom_sig2g1_i = denom_sig2g1 - 2*G_i[j].dot(G_i[j])
            else: denom_sig2g1_i = denom_sig2g1 - 2*G_i[j].dot(G_i[j] * u1_sqr[i+j]*u1_sqr)         
            sig2g1_estimators_arr[i+j] = numer_sig2g1_i / denom_sig2g1_i
            
        G_i = X1_i.dot(X2.T) / X1.shape[1]
        G_i[is_same[i:i+window_size]]=0         
        for j in xrange(G_i.shape[0]):
            numer_rho_i = numer_rho - G_i[j].dot(qy1[i+j]*qy2)
            if (u1_0 is None): denom_rho_i = denom_rho - G_i[j].dot(G_i[j])
            else: denom_rho_i = denom_rho - G_i[j].dot(G_i[j] * u1_sqr[i+j]*u2_sqr)         
            rho_estimators_arr[i+j] = numer_rho_i / denom_rho_i
            
            
    #exclude individuals from study 2
    for i in xrange(0, X2.shape[0], window_size):
        X2_i = X2[i:i+window_size]
        G_i = X2_i.dot(X2.T) / X1.shape[1]
        indices0 = np.arange(G_i.shape[0])
        G_i[indices0, i+indices0]=0
        
        for j in xrange(G_i.shape[0]):
            numer_sig2g2_i = numer_sig2g2 - G_i[j].dot(qy2[i+j]*qy2)
            if (u2_0 is None): denom_sig2g2_i = denom_sig2g2 - 2*G_i[j].dot(G_i[j])
            else: denom_sig2g2_i = denom_sig2g2 - 2*G_i[j].dot(G_i[j] * u2_sqr[i+j]*u2_sqr)         
            sig2g2_estimators_arr[i+j] = numer_sig2g2_i / denom_sig2g2_i
            
        G_i = X2_i.dot(X1.T) / X1.shape[1]
        G_i[is_same.T[i:i+window_size]]=0
        for j in xrange(G_i.shape[0]):
            numer_rho_i = numer_rho - G_i[j].dot(qy1[i+j]*qy2)
            if (u1_0 is None): denom_rho_i = denom_rho - G_i[j].dot(G_i[j])
            else: denom_rho_i = denom_rho - G_i[j].dot(G_i[j] * u2_sqr[i+j]*u1_sqr)         
            rho_estimators_arr[X1.shape[0]+i+j] = numer_rho_i / denom_rho_i         
            
                   
    sig2g1_estimators_arr /= pcgc_coeff1
    sig2g2_estimators_arr /= pcgc_coeff2
    rho_estimators_arr /= pcgc_coeff12  
    
    sig2g1_var = (X1.shape[0]-1)/float(X1.shape[0]) * np.sum((sig2g1_estimators_arr - sig2g1_estimators_arr.mean())**2)
    sig2g2_var = (X2.shape[0]-1)/float(X2.shape[0]) * np.sum((sig2g2_estimators_arr - sig2g2_estimators_arr.mean())**2) 
    rho_var    = (rho_estimators_arr.shape[0]-1)/float(rho_estimators_arr.shape[0]) * np.sum((rho_estimators_arr - rho_estimators_arr.mean())**2)
    
    #compute genetic correlation pseudo-values
    sig2g1 = numer_sig2g1 / denom_sig2g1 / pcgc_coeff1
    sig2g2 = numer_sig2g2 / denom_sig2g2 / pcgc_coeff2
    sig2g1_estimators_arr = np.concatenate((sig2g1_estimators_arr, np.ones(X2.shape[0])*sig2g1))
    sig2g2_estimators_arr = np.concatenate((np.ones(X1.shape[0])*sig2g2, sig2g2_estimators_arr))
    corr_estimators_arr = rho_estimators_arr / np.sqrt(sig2g1_estimators_arr * sig2g2_estimators_arr)
    corr_var = (corr_estimators_arr.shape[0]-1)/float(corr_estimators_arr.shape[0]) * np.sum((corr_estimators_arr - corr_estimators_arr.mean())**2)
    
    return np.sqrt(sig2g1_var), np.sqrt(sig2g2_var), np.sqrt(rho_var), np.sqrt(corr_var)
        
        
# # # def permutation_test(G, yyT, is_same, num_perms=10000):
    
    # # # x = G.reshape(-1)
    # # # y = yyT.reshape(-1)
    # # # x = x[~(is_same.reshape(-1))]
    # # # y = y[~(is_same.reshape(-1))]
    
    # # # real_stat = x.dot(y)
    # # # null_stats = np.empty(num_perms)

    # # # for i in xrange(num_perms):
        # # # if (i>0 and i % 100 == 0): print 'finished %d/%d permutations'%(i, num_perms)
        # # # np.random.shuffle(y)
        # # # null_stats[i] = x.dot(y)
        
    # # # pvalue = np.mean(np.abs(null_stats) > np.abs(real_stat))
    # # # if (pvalue < 1.0/num_perms): pvalue = 1.0/num_perms
    # # # return pvalue
    
    
def permutation_test2(X1, y1, X2, y2, G12_issame, is_same1, is_same2, num_perms=10000):

    has_same = (G12_issame.shape[0] > 0)
    c = float(X1.shape[1])
    y1 = y1.copy()
    y2 = y2.copy()
    
    null_stats = np.empty(num_perms)
    z1 = y1.dot(X1)
    z2 = y2.dot(X2)
    real_stat = z1.dot(z2) / c
    if has_same: real_stat -= G12_issame.dot(y1[is_same1] * y2[is_same2])
    for i in xrange(num_perms):
        if (i>0 and i % 100 == 0): print 'finished %d/%d permutations'%(i, num_perms)
        np.random.shuffle(y1)
        np.random.shuffle(y2)
        z1 = y1.dot(X1)
        z2 = y2.dot(X2)
        null_stats[i] = z1.dot(z2) / c
        if has_same: null_stats[i] -= G12_issame.dot(y1[is_same1] * y2[is_same2])
        
    pvalue = np.mean(np.abs(null_stats) > np.abs(real_stat))
    if (pvalue < 1.0/num_perms): pvalue = 1.0/num_perms
    return pvalue

    
    
    
def permutation_test_heritability(X, y, G_diag, num_perms=10000):

    c = float(X.shape[1])
    y = y.copy()
    
    null_stats = np.empty(num_perms)
    z = y.dot(X)
    real_stat = z.dot(z) / c
    real_stat -= G_diag.dot(y**2)
    for i in xrange(num_perms):
        if (i>0 and i % 100 == 0): print 'finished %d/%d permutations'%(i, num_perms)
        np.random.shuffle(y)
        z = y.dot(X)
        null_stats[i] = z.dot(z) / c
        null_stats[i] -= G_diag.dot(y**2)
        
    pvalue = np.mean(np.abs(null_stats) > np.abs(real_stat))
    if (pvalue < 1.0/num_perms): pvalue = 1.0/num_perms
    return pvalue


    
#compute the PCGC denominator with limited memory, by only storing matrices of size (window_size x sample_size)
def pcgc_denom_lowmem(X1, X2, u1_0, u1_1, u2_0, u2_1, is_same=None, window_size=1000):


    print_memory_usage(7)

    denom=0
    if (window_size is None or window_size<0): window_size = X1.shape[0]
    for i in xrange(0, X1.shape[0], window_size):
        G_i = X1[i:i+window_size].dot(X2.T)
        if (is_same is None):
            indices0 = np.arange(G_i.shape[0])
            G_i[indices0, i+indices0]=0
        else: G_i[is_same[i:i+window_size]] = 0         
        u1_0_i = u1_0[i:i+window_size]
        u1_1_i = u1_1[i:i+window_size]
        denom += np.einsum('ij,ij,i,j',G_i,G_i,u1_0_i**2,u2_0**2) + np.einsum('ij,ij,i,j',G_i,G_i,u1_0_i**2,u2_1**2) + np.einsum('ij,ij,i,j',G_i,G_i,u1_1_i**2,u2_0**2) + np.einsum('ij,ij,i,j',G_i,G_i,u1_1_i**2,u2_1**2)
        denom += 2 * (
              np.einsum('ij,ij,i,j->', G_i, G_i, u1_0_i**2,u2_0*u2_1)
            + np.einsum('ij,ij,i,j->', G_i, G_i, u1_0_i*u1_1_i,u2_0**2)
            + np.einsum('ij,ij,i,j->', G_i, G_i, u1_0_i*u1_1_i,u2_0*u2_1)
            + np.einsum('ij,ij,i,j->', G_i, G_i, u1_0_i*u1_1_i,u2_1*u2_0)
            + np.einsum('ij,ij,i,j->', G_i, G_i, u1_0_i*u1_1_i,u2_1**2)
            + np.einsum('ij,ij,i,j->', G_i, G_i, u1_1_i**2, u2_1*u2_0)          
        )
    denom /= X1.shape[1]**2
    return denom

    
#compute the PCGC denominator with limited memory, by only storing matrices of size (window_size x sample_size), without covariates
def pcgc_denom_lowmem_nocov(X1, X2, is_same=None, window_size=1000):

    print_memory_usage(6)

    denom=0
    if (window_size is None or window_size<0): window_size = X1.shape[0]
    for i in xrange(0, X1.shape[0], window_size):
        G_i = X1[i:i+window_size].dot(X2.T)
        if (is_same is None):
            indices0 = np.arange(G_i.shape[0])
            G_i[indices0, i+indices0]=0
        else: G_i[is_same[i:i+window_size]] = 0         
        denom += np.einsum('ij,ij',G_i,G_i)
    denom /= X1.shape[1]**2
    return denom    


def write_sumstats(z, n, snpNames, out_file, compute_p=True):

    #Compute p-values
    t = z / np.sqrt(n)
    t[t>1.0] = 1.0
    t[t<-1.0] = -1.0
    degrees_f = n-2
    TINY = 1.0e-20
    stat = t * np.sqrt(degrees_f / ((1.0-t+TINY) * (1.0+t+TINY)))
    pvals = stats.t.sf(np.abs(stat), degrees_f)*2
    if not compute_p: pvals[:] = np.zeros(pvals.shape[0]) + np.nan

    df = pd.DataFrame(snpNames, columns=['snpid'])
    df['a1'] = ['1']*len(pvals)
    df['a2'] = ['2']*len(pvals)
    df['N'] = [n]*len(pvals)
    df['P'] = pvals
    df['Z'] = z
    
    if (len(out_file) < 4 or out_file[-5:] != '.gzip' or out_file[-3:] != '.gz'): out_file += '.gz' 
    df.to_csv(out_file, sep='\t', index=False, float_format='%0.6e', compression='gzip', na_rep='NA')



def print_preamble():
    print '*********************************************************************'
    print '* PCGC-direct for heritability and genetic correlation estimates'
    print '* Version 1.0.0'
    print '* (C) 2017 Omer Weissbrod'
    print '* Technion - Israel Institute of Technology'
    print '*********************************************************************'
    print


#compute liability variance due to covariates
def varLiab_covar(prev, tau_i, phe):
    var_E_t_given_y = prev * (1-prev) * (tau_i[phe>phe.mean()].mean() - tau_i[phe<phe.mean()].mean())**2
    E_var_t_given_y = prev * np.var(tau_i[phe>phe.mean()]) + (1-prev) * np.var(tau_i[phe<phe.mean()])
    var_t = var_E_t_given_y + E_var_t_given_y
    return var_t



def my_linreg(X,y):
    R = X.T.dot(X)
    XTy = X.T.dot(y)
    L = la.cho_factor(R)
    coef = la.cho_solve(L, XTy)
    return coef
    

#initial computations required for PCGC
def regress_given_PCs(X, cov, PC_indices):

    assert np.all(PC_indices <= cov.shape[1]), 'given PC number cannot be larger than %d'%(cov.shape[1])
    assert np.all(PC_indices > 0)
    assert np.all(~np.isnan(cov))
    assert np.all(~np.isnan(X))
    
    coef = my_linreg(cov[:, PC_indices-1], X)
    X -= cov[:, PC_indices-1].dot(coef)
    
    # linreg = LinearRegression(fit_intercept=False)
    # linreg.fit(cov[:, PC_indices-1], X)
    # X -= linreg.predict(cov[:, PC_indices-1])
    
    
    
    return X

    
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    #parameters for exact computations
    parser.add_argument('--sumstats_only', metavar='sumstats_only', type=int, default=0, help='If set to 1, PCGC-s will only compute summary statistics and print them to files, without estimating variance components (default 0)')   
    parser.add_argument('--bfile1', metavar='bfile1', required=True, help='plink file for study 1')
    parser.add_argument('--bfile2', metavar='bfile2', default=None, help='plink file for study 2')
    parser.add_argument('--pheno1', metavar='pheno1', required=True, help='phenotypes file for study 1')
    parser.add_argument('--pheno2', metavar='pheno2', default=None, help='phenotypes file for study 2')
    parser.add_argument('--covar1', metavar='covar1', default=None, help='covariates file for study 1')
    parser.add_argument('--covar2', metavar='covar2', default=None, help='covariates file for study 2')
    parser.add_argument('--prev1', metavar='prev1', type=float, required=True, help='population prevalence of study 1')
    parser.add_argument('--prev2', metavar='prev2', type=float, default=None, help='population prevalence of study 2')
    parser.add_argument('--extract', metavar='extract', default=None, help='file with list of SNPs to use')
    parser.add_argument('--keep1', metavar='keep1', default=None, help='file with list of individuals to use in study 1')
    parser.add_argument('--keep2', metavar='keep2', default=None, help='file with list of individuals to use in study 2')
    parser.add_argument('--norm', metavar='norm', default=None, help='SNPs normalization method (see help file)')   
    parser.add_argument('--maf', metavar='maf', default=None, help='MAFs file (to be used with "--norm maf" option)')   
    parser.add_argument('--numPCs1', metavar='numPCs1', type=int, default=0, help='#PCs to regress out of dataset 1')   
    parser.add_argument('--numPCs2', metavar='numPCs2', type=int, default=0, help='#PCs to regress out of dataset 2')   
    parser.add_argument('--chr', metavar='chr', type=int, default=None, help='use only SNPs from a specific chromosome')    
    parser.add_argument('--missingPhenotype', metavar='missingPhenotype', default='-9', help='identifier for missing values (default: -9)')
    parser.add_argument('--center', metavar='center', type=int, default=1, help='whether to center SNPs prior to computing kinship (0 or 1, default 1)')    

    
    parser.add_argument('--mem_size', metavar='mem_size', type=int, default=1000, help='The maximum number of rows in each kinship matrix to be computed. Larger values will improve run-time take up more memory')
    parser.add_argument('--jackknife', metavar='jackknife', type=int, default=1, help='Whether jackknife-based standard errors will be computed (0 or 1, default 1)')
    parser.add_argument('--num_perms', metavar='num_perms', type=int, default=0, help='number of permutation testing iterations')   
    
    parser.add_argument('--z1_nocov_out', metavar='z1_nocov_out', default=None, help='output file for Z-score statistics for study 1 without covariates')   
    parser.add_argument('--z2_nocov_out', metavar='z2_nocov_out', default=None, help='output file for Z-score statistics for study 2 without covariates')   
    parser.add_argument('--z1_cov_out', metavar='z1_cov_out', default=None, help='output file for Z-score statistics for study 1 with covariates')  
    parser.add_argument('--z2_cov_out', metavar='z2_cov_out', default=None, help='output file for Z-score statistics for study 2 with covariates')  
    parser.add_argument('--Gty1_nocov_out', metavar='Gty1_nocov_out', default=None, help='output file for covariate-less summary information for individuals in study 1')
    parser.add_argument('--Gty2_nocov_out', metavar='Gty2_nocov_out', default=None, help='output file for covariate-less summary information for individuals in study 2')
    parser.add_argument('--Gty1_cov_out', metavar='Gty1_cov_out', default=None, help='output file for covariates-summary information for individuals in study 1')
    parser.add_argument('--Gty2_cov_out', metavar='Gty2_cov_out', default=None, help='output file for covariates-summary information for individuals in study 2')
    
    parser.add_argument('--PC1', metavar='PC1', default=None, help='comma-separated indices of covariates that are PCs in covar1 (starting from 1)')
    parser.add_argument('--PC2', metavar='PC2', default=None, help='comma-separated indices of covariates that are PCs in covar2 (starting from 1)')

    parser.add_argument('--snp1', metavar='snp1', default=None, type=int, help='read only a subset of SNPs starting from snp1, starting from 1 (must be specified with snp2)')
    parser.add_argument('--snp2', metavar='snp1', default=None, type=int, help='read only a subset of SNPs ending with snp2, starting from 1 (must be specified with snp1)')
    

    args = parser.parse_args()
    print_preamble()
    
    
    #validate command line arguments
    #####################################################################################
    if (args.bfile2 is not None):
        assert args.pheno2 is not None, '--pheno2 must be specified with --bfile2'      
        assert args.prev2 is not None, '--prev2 must be specified with --bfile2'        
    if (args.bfile2 is None):
        assert args.keep2 is None, '--keep2 cannot be specified without --bfile2'
        assert args.covar2 is None, '--covar2 cannot be specified without --bfile2'
        assert args.prev2 is None, '--prev2 cannot be specified without --bfile2'
        assert args.pheno2 is None, '--pheno2 cannot be specified without --bfile2'
        assert args.z2_nocov_out is None, '--z2_nocov_out cannot be specified without --bfile2'
        assert args.z2_cov_out is None, '--z2_cov_out cannot be specified without --bfile2'
        assert args.numPCs2==0, '--numPCs2 cannot be specified without --bfile2'
        assert args.PC2 is None, '--PC2 cannot be specified without --bfile2'
        
    if (args.numPCs1>0): assert args.PC1 is None, 'PC1 cannot be specified with numPCs1'
    if (args.numPCs2>0): assert args.PC2 is None, 'PC2 cannot be specified with numPCs2'
    
    if (args.PC1 is not None):
        assert args.covar1 is not None, '--PC1 cannot be specified without --covar1'
        args.PC1 = np.array(args.PC1.split(','), dtype=np.int)
        assert np.all(args.PC1 >= 1), '--PC1 numbers must be >=1'
    if (args.PC2 is not None):
        assert args.covar2 is not None, '--PC2 cannot be specified without --covar2'
        args.PC2 = np.array(args.PC2.split(','), dtype=np.int)
        assert np.all(args.PC2 >= 1), '--PC2 numbers must be >=1'
        
    
    if (args.snp1 is not None):
        assert args.snp1>=1, '--snp1 must be >=1'
        assert args.snp2 is not None, '--snp1 must be specified with --snp2'
        assert args.bfile2 is None, '--snp1 cannot be specified when two bfiles are provided'
        assert args.Gty1_nocov_out is None, 'Gty1_nocov_out cannot be specified when --snp1 is set'
        assert args.Gty1_cov_out is None, 'Gty1_cov_out cannot be specified when --snp1 is set'
    if (args.snp2 is not None): assert args.snp1 is not None, '--snp2 must be specified with --snp1'
    
        
    if (args.maf is not None): assert args.norm=='maf', '--maf option can only be used when "--norm maf" option is invoked'
    if (args.norm == 'maf'): assert args.maf is not None, 'maf file must be provided to use "--norm maf"'
    if (args.covar1 is None):
        assert args.z1_cov_out is None, 'z1_cov_out cannor be specified without covar1'
        assert args.Gty1_cov_out is None, 'Gty1_out cannor be specified without covar1'
    if (args.covar2 is None):
        assert args.z2_cov_out is None, 'z2_cov_out cannor be specified without covar1'
        assert args.Gty2_cov_out is None, 'Gty2_out cannor be specified without covar1'
        
    if (args.sumstats_only > 0):
        assert args.z1_nocov_out is not None or args.z1_cov_out is not None, 'z1_nocov_out or z1_cov_out must be defined when sumstats_only=1'
        assert args.num_perms==0, 'permutation testing can not be used when sumstats_only=1'
    #####################################################################################
    
    
    #read and preprocess the data
    X1, bed1, phe1, cov1, X2, bed2, phe2, cov2 = pcgcs_utils.read_SNPs(bfile1=args.bfile1, pheno1=args.pheno1, prev1=args.prev1, covar1=args.covar1, keep1=args.keep1, bfile2=args.bfile2, pheno2=args.pheno2, prev2=args.prev2, covar2=args.covar2, keep2=args.keep2, extract=args.extract, missingPhenotype=args.missingPhenotype, chr=args.chr, norm=args.norm, maf=args.maf, center=args.center>0, snp1=args.snp1, snp2=args.snp2)
    assert np.all(~np.isnan(X1))
    if (cov1 is not None): assert np.all(~np.isnan(cov1))
    
    #regress out PCs
    s1, sum_s1, sum_s1_sqr = None, None, None
    if (args.PC1 is not None):
        print 'regressing given PCs out of bfile1'
        X1 = regress_given_PCs(X1, cov1, args.PC1)        
    elif (args.numPCs1>0):
        print 'Regressing top %d PCs out of bfile 1'%(args.numPCs1)
        X1, U1, s1, sum_s1, sum_s1_sqr = pcgcs_utils.regress_PCs(X1, args.numPCs1)
        print 'done'
        if (cov1 is None): cov1 = U1
        else: cov1 = np.concatenate((cov1, U1), axis=1)
        
    s2, sum_s2, sum_s2_sqr = None, None, None
    if (args.PC2 is not None):
        print 'regressing given PCs out of bfile2'
        X2 = regress_given_PCs(X2, cov2, args.PC2)
    elif (args.numPCs2>0):
        print 'Regressing top %d PCs out of bfile 2'%(args.numPCs2)
        X2, U2, s2, sum_s2, sum_s2_sqr = pcgcs_utils.regress_PCs(X2, args.numPCs2)
        print 'done'
        if (cov2 is None): cov2 = U2
        else: cov2 = np.concatenate((cov2, U2), axis=1)
    
    
    #print plink file sizes
    print_memory_usage(3.1)
    print 'bfile1: %d cases, %d controls, %d SNPs'%(np.sum(phe1>phe1.mean()), np.sum(phe1<=phe1.mean()), bed1.sid.shape[0])
    print_memory_usage(3.2)
    if (args.sumstats_only==0 or args.Gty1_nocov_out is not None or args.Gty1_cov_out is not None): G1_diag = np.einsum('ij,ij->i', X1,X1) / float(X1.shape[1])
    print_memory_usage(3.3)
    
    
    if (bed2 is not None):
        if (args.sumstats_only==0 or args.Gty2_nocov_out is not None or args.Gty2_cov_out is not None): G2_diag = np.einsum('ij,ij->i', X2,X2) / float(X2.shape[1])
        print 'bfile2: %d cases, %d controls, %d SNPs'%(np.sum(phe2>phe2.mean()), np.sum(phe2<=phe2.mean()), bed2.sid.shape[0])     

    print_memory_usage(4)
        
    #PCGC initial computations
    y1_norm, tau_i_1, pcgc_coeff1, ty1, u1_0, u1_1 = pcgcs_utils.prepare_PCGC(phe1, args.prev1, cov1)
    if (cov1 is not None): var_t1 = varLiab_covar(args.prev1, tau_i_1, phe1)
    else: var_t1=0
        
    if (bed2 is None): u2_0, u2_1, var_t2 = None, None, None
    else:
        y2_norm, tau_i_2, pcgc_coeff2, ty2, u2_0, u2_1 = pcgcs_utils.prepare_PCGC(phe2, args.prev2, cov2)
        if (cov2 is not None): var_t2 = varLiab_covar(args.prev2, tau_i_2, phe2)
        else: var_t2=0      
        pcgc_coeff12 = np.sqrt(pcgc_coeff1 * pcgc_coeff2)
        
        
    #compute z-scores 
    z1_nocov = y1_norm.dot(X1) / np.sqrt(len(phe1))
    z1_withcov = (ty1 * (u1_0+u1_1)).dot(X1)
    if (bed2 is not None):
        z2_nocov = y2_norm.dot(X2) / np.sqrt(len(phe2))
        z2_withcov = (ty2 * (u2_0+u2_1)).dot(X2)
    
    #write z-scores if required
    if (args.z1_nocov_out is not None): write_sumstats(z1_nocov, len(phe1), bed1.sid, args.z1_nocov_out)
    if (args.z1_cov_out is not None):   write_sumstats(z1_withcov, len(phe1), bed1.sid, args.z1_cov_out, compute_p=False)
    if (args.z2_nocov_out is not None): write_sumstats(z2_nocov, len(phe2), bed2.sid, args.z2_nocov_out)
    if (args.z2_cov_out is not None):   write_sumstats(z2_withcov, len(phe2), bed2.sid, args.z2_cov_out, compute_p=False)
    
    
    print_memory_usage(5)
    
    #write Gty files
    if (args.Gty1_nocov_out is not None):
        Gty1 = np.sqrt(G1_diag) * y1_norm
        df = pd.DataFrame(bed1.iid, columns=['fid', 'iid'])
        df['Gty1'] = Gty1
        df.to_csv(args.Gty1_nocov_out, sep='\t', index=False, float_format='%0.6e', header=None)
        
    if (args.Gty2_nocov_out is not None):
        Gty2 = np.sqrt(G2_diag) * y2_norm
        df = pd.DataFrame(bed2.iid, columns=['fid', 'iid'])
        df['Gty2'] = Gty2
        df.to_csv(args.Gty2_nocov_out, sep='\t', index=False, float_format='%0.6e', header=None)
    
    if (args.Gty1_cov_out is not None):
        Gty1 = np.sqrt(G1_diag) * ty1 * (u1_0 + u1_1)       
        df = pd.DataFrame(bed1.iid, columns=['fid', 'iid'])
        df['Gty1'] = Gty1
        df.to_csv(args.Gty1_cov_out, sep='\t', index=False, float_format='%0.6e', header=None)
        
    if (args.Gty2_cov_out is not None):
        Gty2 = np.sqrt(G2_diag) * ty2 * (u2_0 + u2_1)
        df = pd.DataFrame(bed2.iid, columns=['fid', 'iid'])
        df['Gty2'] = Gty2
        df.to_csv(args.Gty2_cov_out, sep='\t', index=False, float_format='%0.6e', header=None)

    if (args.sumstats_only > 0):
        print_sumstats(cov1, u1_0, u1_1, var_t1, cov2, u2_0, u2_1, var_t2, s1, sum_s1, sum_s1_sqr, s2, sum_s2, sum_s2_sqr)
        sys.exit(0)
        

    #find overlapping individuals
    if (bed2 is not None):
        print 'marking correlations between overlapping individuals...'
        is_same = np.zeros((X1.shape[0], X2.shape[0]), dtype=np.bool)
        is_same1 = np.zeros(X1.shape[0], dtype=np.bool)
        is_same2 = np.zeros(X2.shape[0], dtype=np.bool)
        num_overlap=0
        for i1, ind1 in enumerate(bed1.iid[:,1]):
            for i2, ind2 in enumerate(bed2.iid[:,1]):
                if (ind1 == ind2):
                    is_same[i1,i2] = True
                    is_same1[i1] = True
                    is_same2[i2] = True
                    num_overlap+=1
        
        print 'found %d overlapping individuals'%(num_overlap)      
        #G12_issame = np.mean(X1[is_same1] * X2[is_same2], axis=1)      
        G12_issame = np.einsum('ij,ij->i', X1[is_same1], X2[is_same2]) / float(X1.shape[1])
    

    #Compute PCGC estimates, ignore covariates
    #sig2g_1_nocov_old = np.sum(np.outer(y1_norm, y1_norm) * G1) / np.sum(G1**2) / pcgc_coeff1      
    sig2g1_numer = z1_nocov.dot(z1_nocov) * len(phe1) / float(X1.shape[1]) - G1_diag.dot(y1_norm**2)
    print 'computing PCGC denominator without covariates...'
    t0 = time.time()
    sig2g1_denom = pcgc_denom_lowmem_nocov(X1,X1, window_size=args.mem_size) 
    print 'done in %0.2f seconds'%(time.time() - t0)
    sig2g_1_nocov = sig2g1_numer / sig2g1_denom / pcgc_coeff1
    
    if (bed2 is not None):
        #sig2g_2_nocov_old = np.sum(np.outer(y2_norm, y2_norm) * G2) / np.sum(G2**2) / pcgc_coeff2
        sig2g2_numer = z2_nocov.dot(z2_nocov) * len(phe2) / float(X2.shape[1]) - G2_diag.dot(y2_norm**2)
        sig2g2_denom = pcgc_denom_lowmem_nocov(X2,X2, window_size=args.mem_size) 
        sig2g_2_nocov = sig2g2_numer / sig2g2_denom / pcgc_coeff2
        
        #rho_nocov_old = np.sum(np.outer(y1_norm, y2_norm) * G12) / np.sum(G12**2) / pcgc_coeff12
        rho_numer = z1_nocov.dot(z2_nocov) * np.sqrt(len(phe1) * len(phe2)) / float(X2.shape[1]) - np.sum(G12_issame * y1_norm[is_same1] * y2_norm[is_same2])
        rho_denom = pcgc_denom_lowmem_nocov(X1, X2, is_same=is_same, window_size=args.mem_size)
        rho_nocov = rho_numer / rho_denom / pcgc_coeff12
        
    
    #perform jackknife computations
    if (args.jackknife > 0):
        print 'Computing jackknife standard errors with omitted covariates...'
        t0 = time.time()
        if (bed2 is None):
            sig2g1_se_nocov = pcgc_jackknife_sig2g(X1, y1_norm, sig2g1_numer, sig2g1_denom, pcgc_coeff1, window_size=args.mem_size)
        else:
            sig2g1_se_nocov, sig2g2_se_nocov, rho_se_nocov, corr_se_nocov = pcgc_jackknife_corr(X1, X2, y1_norm, y2_norm,
                sig2g1_numer, sig2g1_denom, sig2g2_numer, sig2g2_denom, rho_numer, rho_denom,
                pcgc_coeff1, pcgc_coeff2, pcgc_coeff12,                 
                is_same=is_same, window_size=args.mem_size)
        print 'done in %0.2f seconds'%(time.time() - t0)
    
    
    print
    print 'Results when excluding covariates'
    print '---------------------------------'
    print 'study 1 h2: %0.4f'%(sig2g_1_nocov),
    if (args.jackknife>0): print '(%0.4f)'%(sig2g1_se_nocov),
    print
    if (bed2 is not None):
        print 'study 2 h2: %0.4f'%(sig2g_2_nocov),
        if (args.jackknife>0): print '(%0.4f)'%(sig2g2_se_nocov),
        print
        print 'genetic covariance: %0.4f'%(rho_nocov),
        if (args.jackknife>0): print '(%0.4f)'%(rho_se_nocov),
        print
        print 'genetic correlation: %0.4f'%(rho_nocov / np.sqrt(sig2g_1_nocov * sig2g_2_nocov)),
        if (args.jackknife>0): print '(%0.4f)'%(corr_se_nocov),
        print
        
    #permutation testing code
    if (args.num_perms > 0):
        print
        print 'Performing covariate-less permutation testing for heritability of study 1 with %d permutations...'%(args.num_perms)
        t0 = time.time()
        rho_pvalue_nocov = permutation_test_heritability(X1, y1_norm, G1_diag, num_perms=args.num_perms)
        print 'done in %0.2f seconds'%(time.time()-t0)
        print 'study 1 h2 p-value (excluding covariates): %0.5e'%(rho_pvalue_nocov)
        if (rho_pvalue_nocov < 100.0/args.num_perms):
            print 'WARNING: p-value is close to the possible limit due to the number of permutations. Please increase the number of permutations to obtain a more accurate result'
                    
    if (bed2 is not None and args.num_perms > 0):
        print
        print 'Performing covariate-less permutation testing for heritability of study 2 with %d permutations...'%(args.num_perms)
        t0 = time.time()
        rho_pvalue_nocov = permutation_test_heritability(X2, y2_norm, G2_diag, num_perms=args.num_perms)
        print 'done in %0.2f seconds'%(time.time()-t0)
        print 'study 2 h2 p-value (excluding covariates): %0.5e'%(rho_pvalue_nocov)
        if (rho_pvalue_nocov < 100.0/args.num_perms):
            print 'WARNING: p-value is close to the possible limit due to the number of permutations. Please increase the number of permutations to obtain a more accurate result'

    
        print
        print 'Performing covariate-less permutation testing for genetic correlation with %d permutations...'%(args.num_perms)
        t0 = time.time()
        rho_pvalue_nocov = permutation_test2(X1, y1_norm, X2, y2_norm, G12_issame, is_same1, is_same2, num_perms=args.num_perms)
        print 'done in %0.2f seconds'%(time.time()-t0)
        print 'genetic correlation p-value (excluding covariates): %0.5e'%(rho_pvalue_nocov)
        if (rho_pvalue_nocov < 100.0/args.num_perms):
            print 'WARNING: p-value is close to the possible limit due to the number of permutations. Please increase the number of permutations to obtain a more accurate result'
            
    print
    print


    if (cov1 is not None or cov2 is not None):
    
        qty1 = ty1 * (u1_0 + u1_1)
        if (bed2 is not None): qty2 = ty2 * (u2_0 + u2_1)       
    
        #Compute PCGC estimates, include covariates
        #sig2g_1_withcov_old = np.sum(np.outer(ty1, ty1)*G1*Q1) / np.sum((G1*Q1)**2)        
        numer_sig2g1 = z1_withcov.dot(z1_withcov) / X1.shape[1] - G1_diag.dot(qty1**2)
        denom_sig2g1 = pcgc_denom_lowmem(X1, X1, u1_0, u1_1, u1_0, u1_1, window_size=args.mem_size)
        sig2g_1_withcov = numer_sig2g1 / denom_sig2g1
        h2_1_withcov = sig2g_1_withcov / (1 + var_t1)

        if (bed2 is not None):
            #sig2g_2_withcov_old = np.sum(np.outer(ty2, ty2)*G2*Q2) / np.sum((G2*Q2)**2)
            numer_sig2g2 = z2_withcov.dot(z2_withcov) / X2.shape[1] - G2_diag.dot(qty2**2)
            denom_sig2g2 = pcgc_denom_lowmem(X2, X2, u2_0, u2_1, u2_0, u2_1, window_size=args.mem_size)         
            sig2g_2_withcov = numer_sig2g2 / denom_sig2g2
            h2_2_withcov = sig2g_2_withcov / (1 + var_t2)

            #rho_withcov_old = np.sum(np.outer(ty1, ty2)*G12*Q12) / np.sum((G12*Q12)**2)
            numer_rho = z1_withcov.dot(z2_withcov) / X2.shape[1] - np.sum(G12_issame * qty1[is_same1] * qty2[is_same2])
            denom_rho = pcgc_denom_lowmem(X1, X2, u1_0, u1_1, u2_0, u2_1, is_same, window_size=args.mem_size)
            rho_withcov = numer_rho / denom_rho
            
            
        if (args.jackknife > 0):
            print 'Computing jackknife standard errors with covariates...'
            t0 = time.time()
            if (bed2 is None):
                sig2g1_se_withcov = pcgc_jackknife_sig2g(X1, ty1, numer_sig2g1, denom_sig2g1, u0=u1_0, u1=u1_1, window_size=args.mem_size)
            else:
                sig2g1_se_withcov, sig2g2_se_withcov, rho_se_withcov, corr_se_withcov = pcgc_jackknife_corr(X1, X2, ty1, ty2,
                    numer_sig2g1, denom_sig2g1, numer_sig2g2, denom_sig2g2, numer_rho, denom_rho,
                    u1_0=u1_0, u1_1=u1_1, u2_0=u2_0, u2_1=u2_1,
                    is_same=is_same, window_size=args.mem_size)
            print 'done in %0.2f seconds'%(time.time()-t0)
            
            
            
        print
        print 'Results when including covariates'
        print '---------------------------------'
        if (args.jackknife==0):
            print 'study 1 h2: %0.4f  (genetic variance: %0.4f)'%(h2_1_withcov, sig2g_1_withcov)
        else:
            print 'study 1 h2: %0.4f (%0.4f) (genetic variance: %0.4f (%0.4f))'%(h2_1_withcov, sig2g1_se_withcov/(1+var_t1), sig2g_1_withcov, sig2g1_se_withcov)
        
        if (bed2 is not None):
            if (args.jackknife==0):
                print 'study 2 h2: %0.4f  (genetic variance: %0.4f)'%(h2_2_withcov, sig2g_2_withcov)
                print 'genetic covariance: %0.4f'%(rho_withcov)
                print 'genetic correlation: %0.4f'%(rho_withcov / np.sqrt(sig2g_1_withcov * sig2g_2_withcov))
            else:
                print 'study 2 h2: %0.4f (%0.4f)  (genetic variance: %0.4f (%0.4f))'%(h2_2_withcov, sig2g2_se_withcov/(1+var_t2),  sig2g_2_withcov, sig2g2_se_withcov)
                print 'genetic covariance: %0.4f (%0.4f)'%(rho_withcov, rho_se_withcov)
                print 'genetic correlation: %0.4f (%0.4f)'%(rho_withcov / np.sqrt(sig2g_1_withcov * sig2g_2_withcov), corr_se_withcov)
                
                
                
        #permutation testing code
        if (args.num_perms > 0):
            print
            print 'Performing covariate-aware permutation testing for heritability of study 1 with %d permutations...'%(args.num_perms)
            t0 = time.time()
            rho_pvalue_cov = permutation_test_heritability(X1, qty1, G1_diag, num_perms=args.num_perms)
            print 'done in %0.2f seconds'%(time.time()-t0)
            print 'study 1 h2 p-value (including covariates): %0.5e'%(rho_pvalue_cov)
            if (rho_pvalue_cov < 100.0/args.num_perms):
                print 'WARNING: p-value is close to the possible limit due to the number of permutations. Please increase the number of permutations to obtain a more accurate result'
                        
                
        if (args.covar2 is not None and args.num_perms > 0):
        
            print
            print 'Performing covariate-aware permutation testing for heritability of study 2 with %d permutations...'%(args.num_perms)
            t0 = time.time()
            rho_pvalue_cov = permutation_test_heritability(X2, qty2, G2_diag, num_perms=args.num_perms)
            print 'done in %0.2f seconds'%(time.time()-t0)
            print 'study 2 h2 p-value (including covariates): %0.5e'%(rho_pvalue_cov)
            if (rho_pvalue_cov < 100.0/args.num_perms):
                print 'WARNING: p-value is close to the possible limit due to the number of permutations. Please increase the number of permutations to obtain a more accurate result'
                                
        
            print
            print 'Performing covariate-aware permutation testing for genetic correlation with %d permutations...'%(args.num_perms)
            t0 = time.time()
            rho_pvalue_cov = permutation_test2(X1, qty1, X2, qty2, G12_issame, is_same1, is_same2, num_perms=args.num_perms)
            print 'done in %0.2f seconds'%(time.time()-t0)
            print 'genetic correlation p-value (including covariates): %0.5e'%(rho_pvalue_cov)
            if (rho_pvalue_cov < 100.0/args.num_perms):
                print 'WARNING: p-value is close to the possible limit due to the number of permutations. Please increase the number of permutations to obtain a more accurate result'
            
                
            
                
    
    print_sumstats(cov1, u1_0, u1_1, var_t1, cov2, u2_0, u2_1, var_t2, s1, sum_s1, sum_s1_sqr, s2, sum_s2, sum_s2_sqr)