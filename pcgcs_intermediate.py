import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import sys
import random
import time
import os
import os.path
import scipy.linalg.blas as blas
from pysnptools.snpreader.bed import Bed
import pandas as pd
import statsmodels.api as sm
import statsmodels.tools.tools as sm_tools
import itertools
np.set_printoptions(precision=4, linewidth=200)
from sklearn.linear_model import LinearRegression, LogisticRegression
import pcgcs_utils
from pcgcs_utils import print_memory_usage



def print_preamble():
    print '*********************************************************************'
    print '* PCGC-intermediate for heritability and genetic correlation estimates'
    print '* Version 1.0.0'
    print '* (C) 2017 Omer Weissbrod'
    print '* Technion - Israel Institute of Technology'
    print '*********************************************************************'
    print
    
    
def print_sumstats(cov, u_0, u_1, var_t, s=None, sum_s=None, sum_s_sqr=None):
    if (cov is None and sum_s is None): return
    print
    print
    print 'summary statistics for subsequent estimation:'
    print '-----------------------------------'
        
    if (cov is not None):
        print 'mean Q for study 1 (mean_Q): %0.4f'%(np.mean((u_0 + u_1)**2))
        print 'liability variance explained by covariates (var_t): %0.4f'%(var_t)
        
    if (sum_s is not None):
        print 'study 1 genotypes deflation factor (geno_factor): %0.6f'%((sum_s - s.sum()) / sum_s)
        print 'study 1 squared genotypes deflation factor  (sqr_geno_factor): %0.6f'%((sum_s_sqr - np.sum(s**2)) / sum_s_sqr)

    print
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

    #parameters
    parser.add_argument('--bfile', metavar='bfile', required=True, help='prefix of a binary plink file')    
    parser.add_argument('--pheno', metavar='pheno', required=True, help='phenotypes file')  
    parser.add_argument('--covar', metavar='covar', help='covariates file') 
    parser.add_argument('--prev', metavar='prev', type=float, required=True, help='trait prevalence')   
    parser.add_argument('--keep', metavar='keep', default=None, help='file with ids of individuals to keep (optional parameter)')   
    parser.add_argument('--extract', metavar='extract', default=None, help='file with list of SNPs to use')
    parser.add_argument('--norm', metavar='norm', default=None, help='SNPs normalization method (see help file)')   
    parser.add_argument('--maf', metavar='maf', default=None, help='MAFs file (to be used with "--norm maf" option)')   
    parser.add_argument('--missingPhenotype', metavar='missingPhenotype', default='-9', help='identifier for missing values (default: -9)')
    parser.add_argument('--center', metavar='center', type=int, default=1, help='whether to center SNPs prior to computing kinship (0 or 1, default 1)')    
    parser.add_argument('--mem_size', metavar='mem_size', type=int, default=50000, help='maximal number of SNPs that will be read in a single batch (smaller values will use less memory but will take more time)') 
    parser.add_argument('--PC', metavar='PC', default=None, help='comma-separated indices of covariates that are PCs (starting from 1)')
    
    parser.add_argument('--Gty_nocov_out', metavar='Gty_nocov_out', default=None, help='output file for covariate-less summary information')
    parser.add_argument('--Gty_cov_out', metavar='Gty_cov_out', default=None, help='output file for covariates-summary information')
    parser.add_argument('--eigenvalues_frac', metavar='eigenvalues_frac', default=None, help='comma-separated list of eigenvalues of PCs')
    parser.add_argument('--ref-ld', metavar='ref_ld', default=None, help='file with LD scores of SNPs, in ldsc format')
    parser.add_argument('--mean-ld', metavar='mean_ld', type=float, default=None, help='mean LD of SNPs')
    
    args = parser.parse_args()
    print_preamble()    

    
    if (args.Gty_cov_out is not None):
        assert args.covar is not None, '--Gty_cov_out must be specified with --covar'
    
    if (args.mean_ld is not None): assert args.ref_ld is None, 'ref-ld and mean-ld cannot both be specified'
    if (args.mean_ld is None):
        assert args.ref_ld is not None, 'either ref-ld or mean-ld must be specified'
        
    if (args.PC is not None):
        assert args.covar is not None, '--PC cannot be specified without --covar'
        args.PC = np.array(args.PC.split(','), dtype=np.int)
        assert np.all(args.PC >= 1), '--PC numbers must be >=1'
    
    
    #find number of SNPs
    bed, _ = pcgcs_utils.loadData(args.bfile, extractSim=args.extract, phenoFile=args.pheno, missingPhenotype=args.missingPhenotype, loadSNPs=False, keep=args.keep)
    num_snps = len(bed.sid)
    
    #compute G diag
    G_diag = None
    num_good_snps = 0
    for snp1 in xrange(0, num_snps, args.mem_size):
        snp2 = np.minimum(snp1+args.mem_size, num_snps-1)
        print 'reading SNPs %d to %d (out of %d total SNPs)'%(snp1+1, snp2+1, num_snps)
        X, _, phe, cov, _, _, _, _ = pcgcs_utils.read_SNPs(bfile1=args.bfile, pheno1=args.pheno, 
                prev1=args.prev, covar1=args.covar, keep1=args.keep, extract=args.extract, missingPhenotype=args.missingPhenotype,
                chr=None, norm=args.norm, maf=args.maf, center=args.center>0, snp1=snp1+1, snp2=snp2+1)
                
        if (args.PC is not None):
            print 'regressing given PCs out of bfile'
            X = regress_given_PCs(X, cov, args.PC)
            num_good_snps += X.shape[1]
            
        if (G_diag is None): G_diag = np.zeros(X.shape[0])        
        G_diag += np.einsum('ij,ij->i', X,X)
    
    G_diag /= float(num_good_snps)
    



    #PCGC initial computations
    y_norm, tau_i, pcgc_coeff, ty, u_0, u_1 = pcgcs_utils.prepare_PCGC(phe, args.prev, cov)
    if (cov is not None): var_t = varLiab_covar(args.prev, tau_i, phe)
    else: var_t=0   
    
    #write Gty files
    if (args.Gty_nocov_out is not None):
        Gty = np.sqrt(G_diag) * y_norm
        df = pd.DataFrame(bed.iid, columns=['fid', 'iid'])
        df['Gty'] = Gty
        df.to_csv(args.Gty_nocov_out, sep='\t', index=False, float_format='%0.6e', header=None)

    if (args.Gty_cov_out is not None):
        Gty = np.sqrt(G_diag) * ty * (u_0 + u_1)        
        df = pd.DataFrame(bed.iid, columns=['fid', 'iid'])
        df['Gty'] = Gty
        df.to_csv(args.Gty_cov_out, sep='\t', index=False, float_format='%0.6e', header=None)
        

    #compute mean LD score
    if (args.ref_ld is None): mean_ld = args.mean_ld
    else:
        df_ld = pd.read_csv(args.ref_ld, delimiter='\s+', index_col='SNP')
        assert (pd.Series(bed.sid).isin(df_ld.index)).all(), 'not all SNPs have LD information'
        df_ld = df_ld.loc[bed.sid]
        mean_ld = df_ld['L2'].mean()
        
        
    # ########### temp stuff ############   
    # X, _, _, _, _, _, _, _ = pcgcs_utils.read_SNPs(bfile1=args.bfile, pheno1=args.pheno, 
            # prev1=args.prev, covar1=args.covar, keep1=args.keep, extract=args.extract, missingPhenotype=args.missingPhenotype,
            # chr=None, norm=args.norm, maf=args.maf, center=args.center>0)
    # XXT = X.dot(X.T)
    # s,U = la.eigh(XXT)
    # s /= X.shape[1]
    # for s_i in np.sort(s)[::-1][:5]:
        # print s_i, s_i/s.sum()
    # sum_s = np.trace(XXT) / X.shape[1]
    # sum_s_sqr = np.sum(XXT**2) / X.shape[1]**2        
    # ###################################

    #compute sum of squares of eigenvalues using LD-score trick
    if (args.eigenvalues_frac is not None):
        sum_s_sqr = X.shape[0]**2 / float(num_good_snps) * mean_ld   +   np.sum(G_diag**2)
        sum_s = np.sum(G_diag)
        s = np.array(args.eigenvalues_frac.split(','), dtype=np.float)  
        assert np.all(s>0)
        assert (s.sum() < 1)
        s *= sum_s
    else:
        s, sum_s, sum_s_sqr = None, None, None
    print_sumstats(cov, u_0, u_1, var_t, s=s, sum_s=sum_s, sum_s_sqr=sum_s_sqr)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    