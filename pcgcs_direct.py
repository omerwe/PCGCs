from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import scipy.optimize as optimize
import scipy.integrate as integrate
import sys
import random
import time
import os
import os.path
import scipy.linalg.blas as blas
import pandas as pd
import statsmodels.api as sm
import statsmodels.tools.tools as sm_tools
import itertools
np.set_printoptions(precision=4, linewidth=200)
from sklearn.linear_model import LinearRegression, LogisticRegression
import pcgcs_utils



def jackknife_pcgc_corr(X1, G1, y1, z1, Q1=None, u1_0=None, u1_1=None, X2=None, G2=None, y2=None, z2=None, Q2=None, u2_0=None, u2_1=None, G12=None, Q12=None, zero_mask=None, num_blocks=200, pcgc_coeff1=None, pcgc_coeff2=None):

	if (pcgc_coeff1 is None): assert pcgc_coeff2 is None
	if (pcgc_coeff1 is not None and X2 is not None): assert pcgc_coeff2 is not None

	if (Q1 is None):
		G1Q=G1.copy()
		z1 = z1 * np.sqrt(y1.shape[0] / float(X1.shape[1]))
	else:
		G1Q = G1*Q1
		z1 = z1 / np.sqrt(X1.shape[1])
	
	if (X2 is not None):
		if (Q2 is None):
			G2Q=G2.copy()
			z2 = z2 * np.sqrt(y2.shape[0] / float(X1.shape[1]))
			G12Q = G12.copy()
		else:
			G2Q = G2*Q2
			z2 = z2 / np.sqrt(X1.shape[1])
			G12Q = G12*Q12
	
	#determine window size
	m = X1.shape[1]
	window_size = int(np.ceil(float(m) / float(num_blocks)))
	real_num_blocks = int(np.ceil(m / float(window_size)))
	m_jack = m - window_size
	
	numerator_sig2g1 = np.sum(G1Q * np.outer(y1, y1))	
	denominator_sig2g1 = np.sum(G1Q**2)


	#adjust statistics to use a reduced number of SNPs
	G1Q *= (float(m) / float(m_jack))
	numerator_sig2g1_jack = numerator_sig2g1 * (float(m) / float(m_jack))
	denominator_sig2g1_jack = denominator_sig2g1 * (float(m) / float(m_jack))**2
	X1n = X1 / np.sqrt(m_jack)
	X1ny = X1n * y1[:, np.newaxis]
	sig2g_1_estimators = np.empty(real_num_blocks)
	
	if (X2 is not None):
		numerator_sig2g2 = np.sum(G2Q * np.outer(y2, y2))
		numerator_rho    = np.sum(G12Q * np.outer(y1, y2))
		denominator_sig2g2 = np.sum(G2Q**2)
		denominator_rho    = np.sum(G12Q**2)	
		corr_estimator = numerator_rho/denominator_rho / np.sqrt(numerator_sig2g1/denominator_sig2g1  *  numerator_sig2g2/denominator_sig2g2)	
		G2Q *= (float(m) / float(m_jack))
		G12Q *= (float(m) / float(m_jack))
		numerator_sig2g2_jack = numerator_sig2g2 * (float(m) / float(m_jack))
		numerator_rho_jack    = numerator_rho    * (float(m) / float(m_jack))
		denominator_sig2g2_jack = denominator_sig2g2 * (float(m) / float(m_jack))**2
		denominator_rho_jack    = denominator_rho    * (float(m) / float(m_jack))**2
		X2n = (X2 / np.sqrt(m_jack))
		X2ny = X2n * y2[:, np.newaxis]
		sig2g_2_estimators = np.empty(real_num_blocks)
		rho_estimators = np.empty(real_num_blocks)
		corr_estimators = np.empty(real_num_blocks)
		
	
	if (Q1 is None): X1ny_sqr = (X1ny**2).sum(axis=0)		
	else: X1ny_sqr = ((X1n * (y1 * (u1_0 + u1_1))[:, np.newaxis])**2).sum(axis=0)				
	sum_diag1_jack = np.sum(X1ny_sqr)
	z1 *= np.sqrt(m/float(m_jack))
	z1_sqr = z1**2
	sum_z1_sqr_jack = np.sum(z1_sqr)
	
	
	if (X2 is not None):
		if (Q2 is None): X2ny_sqr = (X2ny**2).sum(axis=0)
		else: X2ny_sqr = ((X2n * (y2 * (u2_0 + u2_1))[:, np.newaxis])**2).sum(axis=0)		
		sum_diag2_jack = np.sum(X2ny_sqr)
		z2 *= np.sqrt(m/float(m_jack))
		z2_sqr = z2**2
		sum_z2_sqr_jack = np.sum(z2_sqr)
		z1z2 = z1*z2
		sum_z1z2_jack = np.sum(z1z2)
		
		
		if (is_same is not None and np.any(is_same)):
			has_overlap = True
			is_same_1 = np.where(is_same)[0]
			is_same_2 = np.where(is_same)[1]
			if (Q1 is None): X1_overlap_ny = X1[is_same_1] / np.sqrt(m_jack) * y1[is_same_1, np.newaxis]
			else: X1_overlap_ny = X1[is_same_1] / np.sqrt(m_jack) * (y1*(u1_0+u1_1))[is_same_1, np.newaxis]
			if (Q2 is None): X2_overlap_ny = X2[is_same_2] / np.sqrt(m_jack) * y2[is_same_2, np.newaxis]
			else: X2_overlap_ny = X2[is_same_2] / np.sqrt(m_jack) * (y2*(u2_0+u2_1))[is_same_2, np.newaxis]
			
			X12_overlap_ny = (X1_overlap_ny*X2_overlap_ny).sum(axis=0)
			sum_same12_jack = np.sum(X12_overlap_ny)
			
		else:
			has_overlap = False
	
	for iter, snp1 in enumerate(xrange(0, m, window_size)):
	
		#numerators
		# # # temp_sig2g1 = X1ny[:,snp1:snp1+window_size].dot(X1ny[:, snp1:snp1+window_size].T)
		# # # np.fill_diagonal(temp_sig2g1, 0)		
		# # # if (Q1 is not None): temp_sig2g1 *= Q1
		# # # numer_sig2g1_jack_old = numerator_sig2g1_jack - np.sum(temp_sig2g1)
		
		numer_sig2g1_jack = (sum_z1_sqr_jack - z1_sqr[snp1:snp1+window_size].sum()) - (sum_diag1_jack - X1ny_sqr[snp1:snp1+window_size].sum())
		#assert np.isclose(numer_sig2g1_jack, numer_sig2g1_jack_old)
		
		
		if (X2 is not None):
			# # # temp_sig2g2 = X2ny[:,snp1:snp1+window_size].dot(X2ny[:, snp1:snp1+window_size].T)
			# # # np.fill_diagonal(temp_sig2g2, 0)		
			# # # if (Q2 is not None): temp_sig2g2 *= Q2
			# # # numer_sig2g2_jack_old = numerator_sig2g2_jack - np.sum(temp_sig2g2)
			# # # temp_rho = X1ny[:,snp1:snp1+window_size].dot(X2ny[:, snp1:snp1+window_size].T)
			# # # temp_rho[zero_mask]=0
			# # # if (Q12 is not None): temp_rho *= Q12
			# # # numer_rho_jack_old = numerator_rho_jack - np.sum(temp_rho)		
						
			numer_sig2g2_jack = (sum_z2_sqr_jack - z2_sqr[snp1:snp1+window_size].sum()) - (sum_diag2_jack - X2ny_sqr[snp1:snp1+window_size].sum())
			#assert np.isclose(numer_sig2g2_jack, numer_sig2g2_jack_old)
			

			numer_rho_jack = (sum_z1z2_jack - z1z2[snp1:snp1+window_size].sum()) 
			if has_overlap: numer_rho_jack -= (sum_same12_jack - X12_overlap_ny[snp1:snp1+window_size].sum())
			#assert np.allclose(numer_rho_jack, numer_rho_jack_old)
			
			
		
		#denominators
		temp_sig2g1 = X1n[:,snp1:snp1+window_size].dot(X1n[:,snp1:snp1+window_size].T)
		np.fill_diagonal(temp_sig2g1, 0)			
		if (Q1 is not None): temp_sig2g1 *= Q1
		denom_sig2g1_jack = denominator_sig2g1_jack - 2*np.einsum('ij,ij->', temp_sig2g1,G1Q) + np.einsum('ij,ij->',temp_sig2g1,temp_sig2g1)
		
		if (X2 is not None):
			temp_sig2g2 = X2n[:,snp1:snp1+window_size].dot(X2n[:,snp1:snp1+window_size].T)
			np.fill_diagonal(temp_sig2g2, 0)			
			if (Q2 is not None): temp_sig2g2 *= Q2
			denom_sig2g2_jack = denominator_sig2g2_jack - 2*np.einsum('ij,ij->', temp_sig2g2,G2Q) + np.einsum('ij,ij->',temp_sig2g2,temp_sig2g2)
			
			temp_rho = X1n[:,snp1:snp1+window_size].dot(X2n[:,snp1:snp1+window_size].T)
			temp_rho[zero_mask] = 0
			if (Q12 is not None): temp_rho *= Q12	
			denom_rho_jack = denominator_rho_jack - 2*np.einsum('ij,ij->', temp_rho,G12Q) + np.einsum('ij,ij->',temp_rho,temp_rho)

		sig2g_1_estimators[iter] = numer_sig2g1_jack/denom_sig2g1_jack		
		if (X2 is not None):
			sig2g_2_estimators[iter] = numer_sig2g2_jack/denom_sig2g2_jack
			rho_estimators[iter] = numer_rho_jack / denom_rho_jack
		if (iter % 100 == 0 and snp1 > 0): print '\tJackknife finished %d/%d blocks'%(iter, real_num_blocks)	
	assert iter == len(sig2g_1_estimators)-1
	
	corr_estimators = rho_estimators / np.sqrt(sig2g_1_estimators * sig2g_2_estimators)	
	if (pcgc_coeff1 is not None):
		rho_estimators /= np.sqrt(pcgc_coeff1 * pcgc_coeff2)
		sig2g_1_estimators /= pcgc_coeff1
		sig2g_2_estimators /= pcgc_coeff2
		
	sig2g1_var = (real_num_blocks-1)/float(real_num_blocks) * np.sum((sig2g_1_estimators - sig2g_1_estimators.mean())**2)
	if (X2 is not None):
		sig2g2_var = (real_num_blocks-1)/float(real_num_blocks) * np.sum((sig2g_2_estimators - sig2g_2_estimators.mean())**2)
		rho_var  = (real_num_blocks-1)/float(real_num_blocks) * np.sum((rho_estimators - rho_estimators.mean())**2)
		corr_var = (real_num_blocks-1)/float(real_num_blocks) * np.sum((corr_estimators - corr_estimators.mean())**2)
		#print '\tjacknkife corr std estimator: %0.3f'%(np.sqrt(corr_var))	
	
	if (X2 is None): return np.sqrt(sig2g1_var)
	return np.sqrt(sig2g1_var), np.sqrt(sig2g2_var), np.sqrt(rho_var), np.sqrt(corr_var)
	
	
	
	



def write_sumstats(z, n, snpNames, out_file):

	#Compute p-values
	t = z / np.sqrt(n)
	t[t>1.0] = 1.0
	t[t<-1.0] = -1.0
	degrees_f = n-2
	TINY = 1.0e-20
	stat = t * np.sqrt(degrees_f / ((1.0-t+TINY) * (1.0+t+TINY)))
	pvals = stats.t.sf(np.abs(stat), degrees_f)*2

	df = pd.DataFrame(snpNames, columns=['snpid'])
	df['a1'] = ['1']*len(pvals)
	df['a2'] = ['2']*len(pvals)
	df['N'] = [n]*len(pvals)
	df['P'] = pvals
	df['Z'] = z
	
	if (len(out_file) < 4 or out_file[-5:] != '.gzip' or out_file[-3:] != '.gz'): out_file += '.gz'	
	df.to_csv(out_file, sep='\t', index=False, float_format='%0.6e', compression='gzip')



def print_preamble():
	print '*********************************************************************'
	print '* PCGC-direct for heritability and genetic correlation estimates'
	print '* Version 1.0.0'
	print '* (C) 2016 Omer Weissbrod'
	print '* Technion - Israel Institute of Technology'
	print '*********************************************************************'
	print


#compute liability variance due to covariates
def varLiab_covar(prev, tau_i, phe):
	var_E_t_given_y = prev * (1-prev) * (tau_i[phe>phe.mean()].mean() - tau_i[phe<phe.mean()].mean())**2
	E_var_t_given_y = prev * np.var(tau_i[phe>phe.mean()]) + (1-prev) * np.var(tau_i[phe<phe.mean()])
	var_t = var_E_t_given_y + E_var_t_given_y
	return var_t




#initial computations required for PCGC
def prepare_PCGC(phe, prev, cov, return_intermediate=False):
	y = (phe > phe.mean()).astype(np.int)
	P = y.mean()
	tau = stats.norm(0,1).isf(prev)
	phi_tau = stats.norm(0,1).pdf(tau)
	y_norm = (y-P) / np.sqrt(P*(1-P))
	pcgc_coeff = P*(1-P) / (prev**2 * (1-prev)**2) * phi_tau**2
	
	
	if (cov is None):
		Pi = np.ones(len(phe)) * P
	else:
		logreg = LogisticRegression(penalty='l2', C=500000, fit_intercept=True)			
		logreg.fit(cov, y)
		Pi = logreg.predict_proba(cov)[:,1]
		
	K = prev
	Ki = K*(1-P) / (P*(1-K)) * Pi / (1 + K*(1-P) / (P*(1-K))*Pi - Pi)
	tau_i = stats.norm(0,1).isf(Ki)
	tau_i[Ki>=1.] = -999999999
	tau_i[Ki<=0.] = 999999999	
	phi_tau_i = stats.norm(0,1).pdf(tau_i)
	
	u_prefix = phi_tau_i / np.sqrt(Pi*(1-Pi)) / (Ki + (1-Ki)*(K*(1-P))/(P*(1-K)))
	u0 = u_prefix * K*(1-P) / (P*(1-K)) * Pi
	u1 = u_prefix * (1-Pi)
	Q = np.outer(u0,u0) + np.outer(u0,u1) + np.outer(u1,u0) + np.outer(u1,u1)
	
	ty = (y-Pi) / np.sqrt(Pi * (1-Pi))
	
	if return_intermediate:
		return K, P, Ki, Pi, phi_tau_i
	
	return y_norm, tau_i, pcgc_coeff, ty, Q, u0, u1, logreg.intercept_[0], logreg.coef_[0]
	
	
def compute_Q12(phe1, prev1, cov1, phe2, prev2, cov2):

	K1, P1, Ki1, Pi1, phi_tau_i_1 = prepare_PCGC(phe1, prev1, cov1, return_intermediate=True)
	K2, P2, Ki2, Pi2, phi_tau_i_2 = prepare_PCGC(phe2, prev2, cov2, return_intermediate=True)
	
	B0 = np.outer(Ki1 + (1-Ki1)*(K1*(1-P1))/(P1*(1-K1)), Ki2 + (1-Ki2)*(K2*(1-P2))/(P2*(1-K2)))
	u_prefix = np.outer(phi_tau_i_1 / np.sqrt(Pi1*(1-Pi1)), phi_tau_i_2 / np.sqrt(Pi2*(1-Pi2))) / B0
	u00 = u_prefix * K1*(1-P1) / (P1*(1-K1)) * K2*(1-P2) / (P2*(1-K2)) * np.outer(Pi1, Pi2)
	u01 = u_prefix * K1*(1-P1) / (P1*(1-K1)) * np.outer(Pi1, 1-Pi2)
	u10 = u_prefix * K2*(1-P2) / (P2*(1-K2)) * np.outer(1-Pi1, Pi2)
	u11 = u_prefix * np.outer(1-Pi1, 1-Pi2)
	Q12 = u00 + u01 + u10 + u11
	
	return Q12


	
	

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()

	#parameters for exact computations
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
	
	parser.add_argument('--jackknife', metavar='jackknife', type=int, default=0, help='Whether to estimate standard errors via jackknife (0 or 1, default 0)')	
	parser.add_argument('--n-blocks', metavar='n_blocks', type=int, default=200, help='Number of block jackknife blocks')	
	parser.add_argument('--center', metavar='center', type=int, default=1, help='whether to center SNPs prior to computing kinship (0 or 1, default 1)')	
	
	parser.add_argument('--z1_nocov_out', metavar='z1_nocov_out', default=None, help='output file for Z-score statistics for study 1 without covariates')	
	parser.add_argument('--z2_nocov_out', metavar='z2_nocov_out', default=None, help='output file for Z-score statistics for study 2 without covariates')	
	parser.add_argument('--z1_cov_out', metavar='z1_cov_out', default=None, help='output file for Z-score statistics for study 1 with covariates')	
	parser.add_argument('--z2_cov_out', metavar='z2_cov_out', default=None, help='output file for Z-score statistics for study 2 with covariates')	
	parser.add_argument('--Gty1_nocov_out', metavar='Gty1_nocov_out', default=None, help='output file for covariate-less summary information for individuals in study 1')
	parser.add_argument('--Gty2_nocov_out', metavar='Gty2_nocov_out', default=None, help='output file for covariate-less summary information for individuals in study 2')
	parser.add_argument('--Gty1_cov_out', metavar='Gty1_cov_out', default=None, help='output file for covariates-summary information for individuals in study 1')
	parser.add_argument('--Gty2_cov_out', metavar='Gty2_cov_out', default=None, help='output file for covariates-summary information for individuals in study 2')
	

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
		
	if (args.maf is not None): assert args.norm=='maf', '--maf option can only be used when "--norm maf" option is invoked'
	if (args.norm == 'maf'): assert args.maf is not None, 'maf file must be provided to use "--norm maf"'
	if (args.covar1 == 'maf'): assert args.maf is not None, 'maf file must be provided to use "--norm maf"'
	if (args.covar1 is None):
		assert args.z1_cov_out is None, 'z1_cov_out cannor be specified without covar1'
		assert args.Gty1_cov_out is None, 'Gty1_out cannor be specified without covar1'
	if (args.covar2 is None):
		assert args.z2_cov_out is None, 'z2_cov_out cannor be specified without covar1'
		assert args.Gty2_cov_out is None, 'Gty2_out cannor be specified without covar1'
	if (args.jackknife == 0):
		print 'jackknife standard errors will not be computed. To compute standard errors, please use the option "--jackknife 1"'
	#####################################################################################
	
	
	#read and preprocess the data
	bed1, phe1, cov1, bed2, phe2, cov2 = pcgcs_utils.read_SNPs(bfile1=args.bfile1, pheno1=args.pheno1, prev1=args.prev1, covar1=args.covar1, keep1=args.keep1, bfile2=args.bfile2, pheno2=args.pheno2, prev2=args.prev2, covar2=args.covar2, keep2=args.keep2, extract=args.extract, missingPhenotype=args.missingPhenotype, chr=args.chr, norm=args.norm, maf=args.maf, center=args.center>0)
	
	#regress out PCs
	if (args.numPCs1 > 0):
		print 'Regressing top %d PCs out of bfile 1'%(args.numPCs1)
		bed1.val, U1, s1, sum_s1, sum_s1_sqr = pcgcs_utils.regress_PCs(bed1.val, args.numPCs1)
		print 'done'
		if (cov1 is None): cov1 = U1
		else: cov1 = np.concatenate((cov1, U1), axis=1)
		
	if (args.numPCs2 > 0):
		print 'Regressing top %d PCs out of bfile 2'%(args.numPCs2)
		bed2.val, U2, s2, sum_s2, sum_s2_sqr = pcgcs_utils.regress_PCs(bed2.val, args.numPCs2)
		print 'done'
		if (cov2 is None): cov2 = U2
		else: cov2 = np.concatenate((cov2, U2), axis=1)
	
	
	X1 = bed1.val
	print 'bfile1: %d cases, %d controls, %d SNPs'%(np.sum(phe1>phe1.mean()), np.sum(phe1<=phe1.mean()), bed1.sid.shape[0])
	
	
	if (bed2 is not None):
		X2 = bed2.val
		print 'bfile2: %d cases, %d controls, %d SNPs'%(np.sum(phe2>phe2.mean()), np.sum(phe2<=phe2.mean()), bed2.sid.shape[0])
		

	
	#compute kinship matrices
	print 'computing kinship matrix for study 1...'
	t0 = time.time()
	G1 = pcgcs_utils.symmetrize(blas.dsyrk(1.0/X1.shape[1], X1, lower=0))
	print 'done in %0.2f seconds'%(time.time() - t0)
	G1_diag = np.diag(G1).copy()
	
	if (bed2 is not None):
		print 'computing kinship matrix for study 2...'
		t0 = time.time()
		G2 = pcgcs_utils.symmetrize(blas.dsyrk(1.0/X2.shape[1], X2, lower=0))	
		print 'done in %0.2f seconds'%(time.time() - t0)
		G2_diag = np.diag(G2).copy()

		print 'computing kinship matrix between studies 1 and 2...'
		G12 = X1.dot(X2.T) / X1.shape[1]				
		print 'done in %0.2f seconds'%(time.time() - t0)

		print 'marking correlations between overlapping individuals...'
		is_same = np.zeros(G12.shape, dtype=np.bool)
		num_overlap=0
		for i1, ind1 in enumerate(bed1.iid[:,1]):
			for i2, ind2 in enumerate(bed2.iid[:,1]):
				if (ind1 == ind2 or G12[i1,i2]>0.7):
					is_same[i1,i2]=True
					num_overlap+=1
					
		# print 'marking random overlapping individuals!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
		# for i1 in np.random.permutation(len(phe1))[:20]:
			# for i2 in np.random.permutation(len(phe2))[:20]:
				# is_same[i1,i2] = True; num_overlap+=1
		
		print 'found %d overlapping individuals'%(num_overlap)		

		
	#PCGC initial computations
	y1_norm, tau_i_1, pcgc_coeff1, ty1, Q1, u1_0, u1_1, intercept1, fixed1 = prepare_PCGC(phe1, args.prev1, cov1)
	y1y1_nocov = np.outer(y1_norm, y1_norm)
	y1y1_withcov = np.outer(ty1, ty1)
	if (cov1 is not None): var_t1 = varLiab_covar(args.prev1, tau_i_1, phe1)
	else: var_t1=0
		
	if (bed2 is not None):
		y2_norm, tau_i_2, pcgc_coeff2, ty2, Q2, u2_0, u2_1, intercept2, fixed2 = prepare_PCGC(phe2, args.prev2, cov2)
		y2y2_nocov = np.outer(y2_norm, y2_norm)
		y2y2_withcov = np.outer(ty2, ty2)
		if (cov2 is not None): var_t2 = varLiab_covar(args.prev2, tau_i_2, phe2)
		else: var_t2=0
		
		#computations for cross-study estimation
		y1y2_nocov = np.outer(y1_norm, y2_norm)
		y1y2_withcov = np.outer(ty1, ty2)
		pcgc_coeff12 = np.sqrt(pcgc_coeff1 * pcgc_coeff2)
		Q12 = compute_Q12(phe1, args.prev1, cov1, phe2, args.prev2, cov2)
		
		
		
	#remove diagonal entries of G1, G2 and G12
	np.fill_diagonal(G1, 0)
	if (bed2 is not None):
		np.fill_diagonal(G2, 0)
		G12[is_same]=0		
		
		
	#compute z-scores 
	z1_nocov = y1_norm.dot(X1) / np.sqrt(len(phe1))		
	z1_withcov = (ty1 * (u1_0+u1_1)).dot(X1)	
	if (bed2 is not None):
		z2_nocov = y2_norm.dot(X2) / np.sqrt(len(phe2))
		z2_withcov = (ty2 * (u2_0+u2_1)).dot(X2)
	
	#write z-scores if required
	if (args.z1_nocov_out is not None): write_sumstats(z1_nocov, len(phe1), bed1.sid, args.z1_nocov_out)
	if (args.z1_cov_out is not None):   write_sumstats(z1_withcov, len(phe1), bed1.sid, args.z1_cov_out)
	if (args.z2_nocov_out is not None): write_sumstats(z2_nocov, len(phe2), bed2.sid, args.z2_nocov_out)
	if (args.z2_cov_out is not None):   write_sumstats(z2_withcov, len(phe2), bed2.sid, args.z2_cov_out)
	
	#write Gty files
	
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

	
	#Compute PCGC estimates, ignore covariates
	sig2g_1_nocov = np.sum(y1y1_nocov*G1) / np.sum(G1**2) / pcgc_coeff1	
	if (bed2 is not None):
		sig2g_2_nocov = np.sum(y2y2_nocov*G2) / np.sum(G2**2) / pcgc_coeff2
		rho_nocov = np.sum(y1y2_nocov*G12) / np.sum(G12**2) / pcgc_coeff12
		
	if (args.jackknife > 0):
		print 'Computing jackknife standard errors with omitted covariates...'
		if (bed2 is None):
			sig2g1_se_nocov = jackknife_pcgc_corr(X1, G1, y1_norm, z1_nocov, num_blocks=args.n_blocks, pcgc_coeff1=pcgc_coeff1)
		else:
			sig2g1_se_nocov, sig2g2_se_nocov, rho_se_nocov, corr_se_nocov = jackknife_pcgc_corr(X1, G1, y1_norm, z1_nocov, X2=X2, G2=G2, y2=y2_norm, z2=z2_nocov, G12=G12, zero_mask=is_same, num_blocks=args.n_blocks, pcgc_coeff1=pcgc_coeff1, pcgc_coeff2=pcgc_coeff2)
	
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
	print
	print


	if (cov1 is not None or cov2 is not None):
	
		#Compute PCGC estimates, include covariates
		sig2g_1_withcov = np.sum(y1y1_withcov*G1*Q1) / np.sum((G1*Q1)**2)
		#print 'numer/denom for study 1:', np.sum(y1y1_withcov*G1*Q1), np.sum((G1*Q1)**2)		
		h2_1_withcov = sig2g_1_withcov / (1 + var_t1)

		if (bed2 is not None):
			sig2g_2_withcov = np.sum(y2y2_withcov*G2*Q2) / np.sum((G2*Q2)**2)
			#print 'numer/denom for study 2:', np.sum(y2y2_withcov*G2*Q2), np.sum((G2*Q2)**2)
			h2_2_withcov = sig2g_2_withcov / (1 + var_t2)
			rho_withcov = np.sum(y1y2_withcov*G12*Q12) / np.sum((G12*Q12)**2)
			#print 'numer/denom for rho:', np.sum(y1y2_withcov*G12*Q12), np.sum((G12*Q12)**2)
			
		if (args.jackknife > 0):
			print 'Computing jackknife standard errors with covariates...'
			if (bed2 is None):
				sig2g1_se_withcov = jackknife_pcgc_corr(X1, G1, ty1, z1_withcov, Q1, u1_0, u1_1, num_blocks=args.n_blocks)
			else:
				sig2g1_se_withcov, sig2g2_se_withcov, rho_se_withcov, corr_se_withcov = jackknife_pcgc_corr(X1, G1, ty1, z1_withcov, Q1, u1_0, u1_1, X2, G2, ty2, z2_withcov, Q2, u2_0, u2_1, G12, Q12, zero_mask=is_same, num_blocks=args.n_blocks)
			
			

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
				print 'study 2 h2: %0.4f (%0.4f)  (genetic variance: %0.4f (%0.4f))'%(h2_2_withcov, sig2g2_se_withcov/(1+var_t2),  sig2g_2_withcov, sig2g1_se_withcov)
				print 'genetic covariance: %0.4f (%0.4f)'%(rho_withcov, rho_se_withcov)
				print 'genetic correlation: %0.4f (%0.4f)'%(rho_withcov / np.sqrt(sig2g_1_withcov * sig2g_2_withcov), corr_se_withcov)
	
	print
	print
	print 'summary statistics for subsequent estimation:'
	print '-----------------------------------'
		
	if (args.covar1 is not None):
		print 'mean Q for study 1 (mean_Q1): %0.4f'%(np.mean(np.diag(Q1)))
		print 'liability variance explained by covariates (var_t1): %0.4f'%(var_t1)

	if (args.covar2 is not None):
		print 'mean Q for study 2 (mean_Q2): %0.4f'%(np.mean(np.diag(Q2)))
		print 'liability variance explained by covariates (var_t2): %0.4f'%(var_t2)
		
		
	if (args.numPCs1 > 0):
		print 'study 1 genotypes deflation factor (geno1_factor): %0.6f'%((sum_s1 - s1.sum()) / sum_s1)
		print 'study 1 squared genotypes deflation factor  (sqr_geno1_factor): %0.6f'%((sum_s1_sqr - np.sum(s1**2)) / sum_s1_sqr)
		
	if (args.numPCs2 > 0):
		print 'study 2 genotypes deflation factor (geno2_factor): %0.6f'%((sum_s2 - s2.sum()) / sum_s2)
		print 'study 2 squared genotypes deflation factor  (sqr_geno2_factor): %0.6f'%((sum_s2_sqr - np.sum(s2**2)) / sum_s2_sqr)

	
		
	print
	print
		