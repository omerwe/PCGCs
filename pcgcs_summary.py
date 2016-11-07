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
pd.set_option('display.width', 200)


def permutation_test(z1, z2, num_perms=10000, chunk_size=1000):
	real_stat = z1.dot(z2)
	z1_perms = np.empty((chunk_size, z1.shape[0]))
	z2_perms = np.empty((chunk_size, z2.shape[0]))
	null_stats = np.empty(num_perms)
	
	for perm1 in xrange(0, num_perms, chunk_size):
		last_perm = perm1+chunk_size
		if (last_perm >= num_perms): last_perm = num_perms-1
		perm_size = last_perm - perm1
		
		for i in xrange(perm_size):
			z1_perms[i,:] = np.random.permutation(z1)
			z2_perms[i,:] = np.random.permutation(z2)
		null_stats[perm1 : last_perm] = np.sum(z1_perms[:perm_size] * z2_perms[:perm_size], axis=1)
	
	pvalue = np.mean(np.abs(null_stats) > np.abs(real_stat))
	return pvalue
	
		
	
	
	
	


def jackknife_summary(z1, n1, intercept1, denom_base1, z2=None, n2=None, intercept2=None, denom_base2=None, intercept12=None, denom_base12=None, ld_scores=None, mean_ld=None, num_blocks=200, is_cov=False, no_square=False):

	#determine window size
	m = z1.shape[0]
	window_size = int(np.ceil(float(m) / float(num_blocks)))
	real_num_blocks = int(np.ceil(m / float(window_size)))
	m_jack = m - window_size
	
	#preprocessing
	if no_square: z1z1 = z1 / m_jack
	else: z1z1 = z1**2 / m_jack
	
	if not is_cov: z1z1 *= n1
	sum_z1z1 = np.sum(z1z1)	
	
	if (z2 is not None):
		if no_square:  z2z2 = z2 / m_jack
		else: z2z2 = z2**2 / m_jack			
		z1z2 = z1*z2 / m_jack
		if not is_cov:			
			z2z2 *= n2
			z1z2 *= np.sqrt(args.n1 * args.n2)		
		sum_z2z2 = np.sum(z2z2)
		sum_z1z2 = np.sum(z1z2)
	
	if (ld_scores is not None):
		ld_scores_jack = ld_scores / m_jack
		mean_ld_jack = ld_scores_jack.sum()
	else:
		mean_ld_jack = mean_ld
		
	# print (sum_z1z1* (m_jack/float(m)) - intercept1) / denom_base1 * m / mean_ld
	# print (sum_z2z2* (m_jack/float(m)) - intercept2) / denom_base2 * m / mean_ld
	# print (sum_z1z2* (m_jack/float(m)) - intercept12) / denom_base12 * m / mean_ld

	estimators_z1z1 = np.empty(real_num_blocks)
	estimators_z2z2 = np.empty(real_num_blocks)
	estimators_z1z2 = np.empty(real_num_blocks)
	for iter, snp1 in enumerate(xrange(0, m, window_size)):
	
		if (ld_scores is None): mean_ld_jack_iter = mean_ld_jack
		else: mean_ld_jack_iter = mean_ld_jack - ld_scores_jack[snp1:snp1+window_size].sum()	
	
		estimators_z1z1[iter] = (sum_z1z1 - z1z1[snp1:snp1+window_size].sum() - intercept1) / denom_base1 * m_jack / mean_ld_jack_iter
		
		if (z2 is not None):
			estimators_z2z2[iter] = (sum_z2z2 - z2z2[snp1:snp1+window_size].sum() - intercept2) / denom_base2 * m_jack / mean_ld_jack_iter
			estimators_z1z2[iter] = (sum_z1z2 - z1z2[snp1:snp1+window_size].sum() - intercept12) / denom_base12 * m_jack / mean_ld_jack_iter
	
	
	var1 = (real_num_blocks-1)/float(real_num_blocks) * np.sum((estimators_z1z1 - estimators_z1z1.mean())**2)
	if (z2 is None): return np.sqrt(var1)
	
	estimators_corr = estimators_z1z2 / np.sqrt(estimators_z1z1 * estimators_z2z2)	
	var2 = (real_num_blocks-1)/float(real_num_blocks) * np.sum((estimators_z2z2 - estimators_z2z2.mean())**2)
	var12 = (real_num_blocks-1)/float(real_num_blocks) * np.sum((estimators_z1z2 - estimators_z1z2.mean())**2)
	var_corr = (real_num_blocks-1)/float(real_num_blocks) * np.sum((estimators_corr - estimators_corr.mean())**2)	
	return np.sqrt(var1), np.sqrt(var2), np.sqrt(var12), np.sqrt(var_corr)
	
	


def validate_args(args):

	if (args.z1_nocov is None):
		assert args.z1_cov is not None, 'either z1_nocov or z1_cov must be specified'
		assert args.P1 is None, 'P1 cannot be specified without z1_nocov'
		assert args.pheno1 is None, 'pheno1 cannot be specified without z1_nocov'
		assert args.pheno2 is None, 'pheno1 cannot be specified without z1_nocov'
		assert args.prev1 is None, 'prev1 cannot be specified without z1_nocov'
		assert args.prev2 is None, 'prev2 cannot be specified without z1_nocov'
		assert args.bfile is None, 'bfile cannot be specified without z1_nocov'
		assert args.ncon_con == 0, 'ncon-con cannot be specified without z1_nocov'
		assert args.ncas_con == 0, 'ncas-con cannot be specified without z1_nocov'
		assert args.ncon_cas == 0, 'ncon-cas cannot be specified without z1_nocov'
		assert args.ncas_cas == 0, 'ncas-cas cannot be specified without z1_nocov'
		assert args.Gty1_nocov is None, 'Gty1_nocov cannot be specified without z1_nocov'
		assert args.Gty2_nocov is None, 'Gty2_nocov cannot be specified without z1_nocov'
	
	if (args.mean_ld is not None): assert args.ref_ld is None, 'ref-ld and mean-ld cannot both be specified'
	if (args.mean_ld is None):
		assert args.ref_ld is not None, 'either ref-ld or mean-ld must be specified'
		
	if (args.z1_nocov is not None):
		assert args.P1 is not None, 'z1_nocov must be used with "--P1"'
		assert args.prev1 is not None, 'z1_nocov must be used with "--prev1"'		
		
	if (args.bfile is None):
		assert args.chr is None, '--chr can only be used with --bfile option'
		assert args.norm is None, '--norm can only be used with --bfile option'
		assert args.maf is None, '--maf can only be used with --bfile option'
		
		
	if (args.z2_nocov is not None):
		assert args.z1_nocov is not None, 'z2_nocov cannot be used without z1_nocov'
		assert args.n2 is not None, 'z2_nocov must be used with "--n2"'
		assert args.P2 is not None, 'z2_nocov must be used with "--P2"'
		assert args.prev2 is not None, 'z2_nocov must be used with "--prev2"'
		
	# if (args.pheno1 is not None):
		# assert args.z1_cov is not None, 'pheno1 cannot be specified without z1_cov'
	# if (args.pheno2 is not None):
		# assert args.z2_cov is not None, 'pheno2 cannot be specified without z2_cov'
		
	if (args.Gty1_nocov is not None):
		assert args.pheno1 is None, 'no need to specify pheno1 when providing Gty1_nocov'
		assert args.bfile is None, 'no need to specify bfile when providing Gty1_nocov'
		assert args.ncon_con==0, 'no need to specify ncon-con when providing Gty1_nocov'
		assert args.ncon_cas==0, 'no need to specify ncon-cas when providing Gty1_nocov'
		assert args.ncas_con==0, 'no need to specify ncas-con when providing Gty1_nocov'
		assert args.ncas_cas==0, 'no need to specify ncas-cas when providing Gty1_nocov'
		assert args.z1_nocov is not None, 'Gty1_nocov cannot be specified without z1_nocov'
		if (args.z2_nocov is not None):
			assert args.Gty2_nocov is not None, 'Gty2_nocov must be provided'
		
	if (args.Gty2_nocov is not None):
		assert args.pheno2 is None, 'no need to specify pheno2 when providing Gty2_nocov'
		assert args.bfile is None, 'no need to specify bfile when providing Gty2_nocov'
		assert args.Gty1_nocov is not None, 'Gty1_nocov must be provided with Gty2_nocov'
		assert args.z2_nocov is not None, 'Gty2_nocov cannot be specified without z2_nocov'		
		
	if (args.Gty1_cov is not None):
		assert args.z1_cov is not None, 'Gty1_cov cannot be specified without z1_cov'
		
		
	if (args.Gty2_cov is not None):
		assert args.z2_cov is not None, 'Gty2_cov cannot be specified without z2_cov'
		assert args.Gty1_cov is not None, 'Gty2_cov must be specified with Gty1_cov'
		
	if (args.z1_cov is None):
		assert (args.mean_Q1 is None), 'mean_Q1 cannot be specified without z1_cov'
		assert (args.var_t1 is None), 'var_t1 cannot be specified without z1_cov'
	
	if (args.z2_cov is None):
		assert (args.mean_Q2 is None), 'mean_Q2 cannot be specified without z2_cov'
		assert (args.var_t2 is None), 'var_t2 cannot be specified without z2_cov'

	if (args.z1_cov is not None):		
		assert args.n1 is not None, 'z1_cov must be used with "--n1"'
		#assert args.prev1 is not None, 'z1_cov must be used with "--prev1"'
		assert (args.mean_Q1 is not None), 'z1_cov must be used with mean_Q1'
		#assert (args.Gty1_cov is not None), 'z1_cov must be used with Gty1_cov'
		
		if (args.z2_cov is None):
			assert args.Gty1_cov is not None, 'Gty1_cov must be provided for heritability estimation'
		
		if (args.bfile is not None):# or args.ncas_cas>0 or args.ncas_con>0 or args.ncon_cas>0 or args.ncon_con>0):
			assert args.pheno1 is not None, 'phenotypes file 1 for overlapping individuals must be provided'
			assert args.pheno2 is not None, 'phenotypes file 2 for overlapping individuals must be provided'
		
		
	if (args.z2_cov is not None):
		assert args.n2 is not None, 'z2_cov must be used with "--n2"'
		#assert args.prev2 is not None, 'z2_cov must be used with "--prev2"'
		assert (args.mean_Q1 is not None), 'z2_cov must be used with mean_Q1'
		assert (args.mean_Q2 is not None), 'z2_cov must be used with mean_Q2'
		#assert (args.Gty2_cov is not None), 'z2_cov must be used with Gty2_cov'
		
		if (args.Gty1_cov is not None): assert args.Gty2_cov is not None, 'Gty1_cov and Gty2_cov must both be provided'
		
		
		
	if (args.bfile is not None):
		assert args.pheno1 is not None, 'pheno1 must be provided with bfile'
		assert args.pheno2 is not None, 'pheno2 must be provided with bfile'
		
	if (args.pheno1 is not None): assert args.pheno2 is not None, 'pheno1 and pheno2 must be provided together'
	if (args.pheno1 is None): assert args.pheno2 is None, 'pheno1 and pheno2 must be provided together'
	
	if (args.maf is not None): assert args.norm=='maf', '--maf option can only be used when "--norm maf" option is invoked'
	if (args.norm == 'maf'): assert args.maf is not None, 'maf file must be provided to use "--norm maf"'	



def print_preamble():
	print '*********************************************************************'
	print '* PCGC-summary for heritability and genetic correlation estimates'
	print '* Version 1.0.0'
	print '* (C) 2016 Omer Weissbrod'
	print '* Technion - Israel Institute of Technology'
	print '*********************************************************************'
	print
	
	
	
def read_zscores(z1_file, z2_file=None, extractSnpsSet=None, ref_ld=None):

	#read Z-scores of study 1
	df1 = pd.read_csv(z1_file, delimiter='\s+', index_col='snpid')
	if (extractSnpsSet is not None):
		df1 = df1[df1['snpid'].isin(extractSnpsSet)]
		print 'retaining %d Z-scores in study 1 from SNPs in extract file'%(len(df1))
	z1 = df1['Z'].values	
	used_snps = set(df1.index)
	
	#read Z-scores of study 2	
	if (z2_file is None):
		df_joined = df1
		z2 = None
	else:
		df2 = pd.read_csv(z2_file, delimiter='\s+', index_col='snpid')
			
		#join the dataframes
		df_joined = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=('_1','_2'), how='inner')
		assert (len(df_joined) > 0), 'no SNPs exist in both studies'
		print '%d z-scores exist in both studies'%(len(df_joined))
		
		#flip z-scores with opposite strands
		df_conflict = df_joined[df_joined['a1_1'] != df_joined['a1_2']]
		assert np.all(df_conflict['a2_1'] == df_conflict['a1_2']), '%d SNPs have different reported alleles'%(np.sum(df_conflict['a2_1'] != df_conflict['a2_2']))
		if (len(df_conflict) > 0):
			print '%d SNPs have opposite base alleles'%(len(df_conflict))
			df_joined.loc[df_joined['a1_1'] != df_joined['a1_2'], 'Z_2'] *= (-1)
		
		z1 = df_joined['Z_1']
		z2 = df_joined['Z_2']
		used_snps = set(df_joined.index)			
	
	if (ref_ld is None):
		return z1, z2, df_joined.index, None
		
	#read LD-scores
	df_ld = pd.read_csv(args.ref_ld, delimiter='\s+', index_col='SNP')
	df_joined2 = pd.merge(df_joined, df_ld, left_index=True, right_index=True)
	assert (len(df_joined2) > 0), 'no SNPs have LD information'
	if (len(df_joined2) < len(df_joined)):
		print '%d SNPs had no LD score information and were discarded'%(len(df_joined)-len(df_joined2))
	if (z2_file is None):
		z1 = df_joined2['Z']
		z2 = None
	else:
		z1 = df_joined2['Z_1']
		z2 = df_joined2['Z_2']
	ld_scores = df_joined2['L2']
	used_snps = set(df_joined2.index)

	return z1, z2, df_joined2.index, ld_scores




if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()

	#parameters for exact computations
	parser.add_argument('--bfile', metavar='bfile1', default=None, help='plink file for overlapping individuals')
	parser.add_argument('--extract', metavar='extract', default=None, help='list of SNPs to use')
	parser.add_argument('--keep', metavar='keep', default=None, help='file with list of overlapping individuals to keep')
	parser.add_argument('--pheno1', metavar='pheno1', default=None, help='phenotypes file for overlapping individuals in study 1')
	parser.add_argument('--pheno2', metavar='pheno2', default=None, help='phenotypes file for overlapping individuals in study 2')
	parser.add_argument('--norm', metavar='norm', default=None, help='SNPs normalization method (see help file on github page)')	
	parser.add_argument('--maf', metavar='maf', default=None, help='MAFs file (to be used with "--norm maf" option)')
	parser.add_argument('--chr', metavar='chr', type=int, default=None, help='use only SNPs from a specific chromosome')	
	parser.add_argument('--missingPhenotype', metavar='missingPhenotype', default='-9', help='identifier for missing values (default: -9)')
	
	parser.add_argument('--prev1', metavar='prev1', type=float, help='population prevalence of study 1')
	parser.add_argument('--prev2', metavar='prev2', type=float, default=None, help='population prevalence of study 2')
	parser.add_argument('--P1', metavar='P1', type=float, default=None, help='case-control ratio in study 1')	
	parser.add_argument('--P2', metavar='P1', type=float, default=None, help='case-control ratio in study 2')	
	
	
	parser.add_argument('--z1_nocov', metavar='z1_nocov', default=None, help='z-scores file for study 1 when omitting covariates')
	parser.add_argument('--z2_nocov', metavar='z2_nocov', default=None, help='z-scores file for study 2 when omitting covariates')	
	parser.add_argument('--z1_cov', metavar='z1_cov', default=None, help='z-scores file for study 1 when including covariates')
	parser.add_argument('--z2_cov', metavar='z2_cov', default=None, help='z-scores file for study 2 when including covariates')
	
	
	parser.add_argument('--n1', metavar='n1', required=True, type=int, help='sample size of study 1')
	parser.add_argument('--n2', metavar='n2', default=None, type=int, help='sample size of study 2')
	
	parser.add_argument('--n-blocks', metavar='n_blocks', type=int, default=200, help='Number of block jackknife blocks')	
	
	parser.add_argument('--ref-ld', metavar='ref_ld', default=None, help='file with LD scores of SNPs, in ldsc format')
	parser.add_argument('--mean-ld', metavar='mean_ld', type=float, default=None, help='mean LD of SNPs (if this is specified instead of ref-ld, jackknife estimates will be slightly less accurate)')
	
	parser.add_argument('--ncas-cas', metavar='ncas_cas', type=int, default=0, help='#ovrelapping cases-cases')
	parser.add_argument('--ncas-con', metavar='ncas_cas', type=int, default=0, help='#ovrelapping cases-controls')
	parser.add_argument('--ncon-cas', metavar='ncon_cas', type=int, default=0, help='#ovrelapping controls-cases')
	parser.add_argument('--ncon-con', metavar='ncon_con', type=int, default=0, help='#ovrelapping controls-controls')
	
	parser.add_argument('--mean_Q1', metavar='mean_Q1', type=float, default=None, help='mean diagonal value of Q in study 1')
	parser.add_argument('--mean_Q2', metavar='mean_Q2', type=float, default=None, help='mean diagonal value of Q in study 2')
	
	parser.add_argument('--Gty1_cov', metavar='Gty1_cov', default=None, help='covariate-aware Gty values for study 1')
	parser.add_argument('--Gty2_cov', metavar='Gty2_cov', default=None, help='covariate-aware Gty values for study 2')
	parser.add_argument('--Gty1_nocov', metavar='Gty1_nocov', default=None, help='covariate-less Gty values for study 1')
	parser.add_argument('--Gty2_nocov', metavar='Gty2_nocov', default=None, help='covariate-less Gty values for study 2')
	
	parser.add_argument('--var_t1', metavar='var_t1', type=float, default=None, help='study 1 liability variance due to covariates')
	parser.add_argument('--var_t2', metavar='var_t2', type=float, default=None, help='study 2 liability variance due to covariates')
	
	
	
	parser.add_argument('--geno1_factor', metavar='geno1_factor', type=float, default=1.0, help='deflation factor for genotypes of study 1')
	parser.add_argument('--geno2_factor', metavar='geno2_factor', type=float, default=1.0, help='deflation factor for genotypes of study 2')
	parser.add_argument('--sqr_geno1_factor', metavar='sqr_geno1_factor', type=float, default=1.0, help='deflation factor for squared genotypes of study 1')
	parser.add_argument('--sqr_geno2_factor', metavar='sqr_geno2_factor', type=float, default=1.0, help='deflation factor for squared genotypes of study 2')
	
	

	args = parser.parse_args()
	print_preamble()
	validate_args(args)
	
	
	#read extract SNPS file
	if (args.extract is None): extractSnpsSet = None
	else:
		df = pd.read_csv(args.extract, header=None)
		extractSnpsSet = set(df.values[:,0])
		
	#read keep file
	if (args.keep is None): keep_df = None
	else:
		keep_df = pd.read_csv(args.keep, header=None)
		assert len(keep_df.columns)==2, 'keep file must have exactly two columns'
		keep_df.columns = ['fid', 'iid']
		
		
	#read Z scores and LD scores, and compute mean LD score
	if (args.z1_nocov is not None):
		z1_nocov, z2_nocov, snp_names, ld_scores = read_zscores(args.z1_nocov, args.z2_nocov, extractSnpsSet, args.ref_ld)
		if (args.ref_ld is None): mean_ld = args.mean_ld
		else: mean_ld = ld_scores.mean()	
		print 'Total number of SNPs with omitted-covariates Z-scores: %d'%(len(snp_names))		
	
	if (args.z1_cov is not None):
		#read Z scores and LD scores, and compute mean LD score
		z1_cov, z2_cov, snp_names, ld_scores = read_zscores(args.z1_cov, args.z2_cov, extractSnpsSet, args.ref_ld)
		if (args.ref_ld is None): mean_ld = args.mean_ld
		else: mean_ld = ld_scores.mean()
		print 'Total number of SNPs with included-covariates Z-scores: %d'%(len(snp_names))
						
		
	
	if (args.bfile is None):
		#read phenotypes without bfile
		if (args.pheno1 is not None):
			assert args.pheno2 is not None, 'pheno1 and pheno2 must both be specified'
			df1 = pd.read_csv(args.pheno1, header=None, sep='\s+')
			df1.columns = ('fid', 'iid', 'pheno')
			df2 = pd.read_csv(args.pheno2, header=None, sep='\s+')
			df2.columns = ('fid', 'iid', 'pheno')
			#if (keep_df is None): assert len(df1) == len(df2), 'phenotypes files have a different number of individuals'
			df_joined = pd.merge(df1, df2, on=['fid', 'iid'], suffixes=('_1', '_2'))		
			#if (keep_df is None): assert len(df_joined) == len(df1), 'phenotypes files individuals do not match exactly'
			if (keep_df is not None): df_joined = df_joined.isin(keep_df)
			phe1 = df_joined['pheno_1'].values
			phe2 = df_joined['pheno_2'].values
			print 'found %d overlapping individuals in phenotype files'%(len(phe1))
			
			uniq_phe1 = np.unique(phe1)
			if (len(uniq_phe1) == 2):			
				y1_norm_overlap = ((phe1>phe1.mean()) - args.P1) / np.sqrt(args.P1 * (1-args.P1))
			elif (len(uniq_phe1) == 1):
				assert uniq_phe1[0] in [0,1,2], 'non-binary phenotype found in pheno1 file'
				if (uniq_phe1[0]==2): uniq_phe1[0]=1
				print 'only one phenotype code found in pheon1 file'
				if (uniq_phe1[0] == 1): print 'treating all individuals in pheno1 file as cases'
				elif (uniq_phe1[0]==0): print 'treating all individuals in pheno1 file as controls'				
				y1_norm_overlap = np.ones(len(phe1)) * (uniq_phe1[0] - args.P1) / np.sqrt(args.P1 * (1-args.P1))
			else: raise Exception('more than two phenotype codes found in pheno1 file')
			
			uniq_phe2 = np.unique(phe2)
			if (len(uniq_phe2) == 2):			
				y2_norm_overlap = ((phe2>phe2.mean()) - args.P2) / np.sqrt(args.P2 * (1-args.P2))
			elif (len(uniq_phe2) == 1):
				assert uniq_phe2[0] in [0,1,2], 'non-binary phenotype found in pheno2 file'
				if (uniq_phe2[0]==2): uniq_phe2[0]=1
				if (uniq_phe2[0] == 1): print 'treating all individuals in pheno2 file as cases'
				elif (uniq_phe2[0]==0): print 'treating all individuals in pheno2 file as controls'				
				y2_norm_overlap = np.ones(len(phe2)) * (uniq_phe2[0] - args.P2) / np.sqrt(args.P2 * (1-args.P2))
			else: raise Exception('more than two phenotype codes found in pheno2 file')
			

	#read bfile if provided
	else:
		bed, phe1, _, _, phe2, _ = pcgcs_utils.read_SNPs(bfile1=args.bfile, pheno1=args.pheno1, prev1=args.prev1, keep1=args.keep, bfile2=args.bfile, pheno2=args.pheno2, prev2=args.prev2, keep2=args.keep, extract=set(snp_names), missingPhenotype=args.missingPhenotype, chr=args.chr, norm=args.norm, maf=args.maf, center=False)
		assert len(bed.sid) == len(snp_names), 'bfile has less SNPs than there are z-scores'
		assert len(phe1)==len(phe2), 'phenotype files have a different number of individuals'
		print 'found %d overlapping individuals in plink and phenotype files'%(len(phe1))
		G_overlap = np.sum(bed.val**2, axis=1) / bed.val.shape[1]
		y1_norm_overlap = ((phe1>phe1.mean()) - args.P1) / np.sqrt(args.P1 * (1-args.P1))
		y2_norm_overlap = ((phe2>phe2.mean()) - args.P2) / np.sqrt(args.P2 * (1-args.P2))
		
		
	#read Gty1_nocov and Gty2_nocov
	if (args.Gty1_nocov is not None):
		df1 = pd.read_csv(args.Gty1_nocov, header=None, sep='\s+')
		df1.columns = ('fid', 'iid', 'Gty')
		Gty1_nocov = df1['Gty'].values
		assert len(Gty1_nocov)==args.n1, 'number of individuals in Gty1_nocov file is different from n1 argument'
		if (args.z2_nocov is not None):
			df2 = pd.read_csv(args.Gty2_nocov, header=None, sep='\s+')
			df2.columns = ('fid', 'iid', 'Gty')
			Gty2_nocov = df2['Gty'].values
			assert len(Gty2_nocov)==args.n2, 'number of individuals in Gty2_nocov file is different from n2 argument'
			df_joined = pd.merge(df1, df2, on=['fid', 'iid'], suffixes=('_1', '_2'))		
			if (keep_df is not None): df_joined = df_joined.isin(keep_df)
			Gty1_nocov_overlap = df_joined['Gty_1'].values
			Gty2_nocov_overlap = df_joined['Gty_2'].values
			print 'found %d overlapping individuals in files Gty1_nocov and Gty2_nocov'%(len(Gty1_nocov_overlap))
		
		
	
	#compute PCGC coefficients and other quantities
	if (args.z1_nocov is not None):		
		tau1 = stats.norm(0,1).isf(args.prev1)
		phi_tau1 = stats.norm(0,1).pdf(tau1)
		pcgc_coeff1 = args.P1*(1-args.P1) / (args.prev1**2 * (1-args.prev1)**2) * phi_tau1**2
		y1_norm_con = (-args.P1)  / np.sqrt(args.P1 * (1-args.P1))
		y1_norm_cas = (1-args.P1) / np.sqrt(args.P1 * (1-args.P1))	
		if (args.z2_nocov is not None):
			tau2 = stats.norm(0,1).isf(args.prev2)
			phi_tau2 = stats.norm(0,1).pdf(tau2)
			pcgc_coeff2 = args.P2*(1-args.P2) / (args.prev2**2 * (1-args.prev2)**2) * phi_tau2**2
			pcgc_coeff12 = np.sqrt(pcgc_coeff1 * pcgc_coeff2)
			y2_norm_con = (-args.P2)  / np.sqrt(args.P2 * (1-args.P2))
			y2_norm_cas = (1-args.P2) / np.sqrt(args.P2 * (1-args.P2))

		
		############### excluded covariates code ###############
		
		#estimate sig2g_1
		if (args.Gty1_nocov is None):
			sig2g1_intercept = args.n1
			sig2g1_intercept *= args.geno1_factor
		else:
			sig2g1_intercept = Gty1_nocov.dot(Gty1_nocov)
		sig2g1_nocov_numer = np.sum(z1_nocov*z1_nocov)*args.n1 / float(z1_nocov.shape[0]) - sig2g1_intercept		
		sig2g1_nocov_denom = args.n1**2 / float(z1_nocov.shape[0]) * mean_ld		
		sig2g1_nocov_denom *= args.sqr_geno1_factor
		sig2g_1_nocov = sig2g1_nocov_numer / sig2g1_nocov_denom / pcgc_coeff1
		
		if (args.z2_nocov is None): se_sig2g_1 = jackknife_summary(z1_nocov, args.n1, sig2g1_intercept, args.n1**2 * pcgc_coeff1, ld_scores=ld_scores, mean_ld=mean_ld, is_cov=False, num_blocks=args.n_blocks)
			
		
		if (args.z2_nocov is not None):
			#estimate sig2g_2
			if (args.Gty2_nocov is None):
				sig2g2_intercept = args.n2
				sig2g2_intercept *= args.geno2_factor
			else:
				sig2g2_intercept = Gty2_nocov.dot(Gty2_nocov)
			sig2g2_nocov_numer = np.sum(z2_nocov*z2_nocov)*args.n2 / float(z2_nocov.shape[0]) - sig2g2_intercept
			sig2g2_nocov_denom = args.n2**2 / (z2_nocov.shape[0])  * mean_ld
			sig2g2_nocov_denom *= args.sqr_geno2_factor			
			sig2g_2_nocov = sig2g2_nocov_numer / sig2g2_nocov_denom / pcgc_coeff2
			
			#estimate rho
			if (args.Gty2_nocov is not None):
				rho_intercept = Gty1_nocov_overlap.dot(Gty2_nocov_overlap)		
			elif (args.bfile is not None):
				rho_intercept = np.sum(G_overlap * y1_norm_overlap * y2_norm_overlap)
			elif (args.pheno1 is not None):
				rho_intercept = np.sum(y1_norm_overlap * y2_norm_overlap)				
			else:			
				rho_intercept = (args.ncon_con*y1_norm_con*y2_norm_con +  
								 args.ncon_cas*y1_norm_con*y2_norm_cas +  
								 args.ncas_con*y1_norm_cas*y2_norm_con +  
								 args.ncas_cas*y1_norm_cas*y2_norm_cas)
				if (rho_intercept==0): print 'Assuming no overlapping individuals...'
				
			rho_intercept *= np.minimum(args.geno1_factor, args.geno2_factor)
			rho_nocov_numer = np.sum(z1_nocov*z2_nocov)*np.sqrt(args.n1*args.n2) / float(z2_nocov.shape[0]) - rho_intercept
			rho_nocov_denom = args.n1*args.n2 / float(z1_nocov.shape[0]) * mean_ld
			rho_nocov_denom *= np.minimum(args.sqr_geno1_factor, args.sqr_geno2_factor)			
			rho_nocov = rho_nocov_numer / rho_nocov_denom / pcgc_coeff12
			if (np.abs(rho_nocov) > np.sqrt(sig2g_1_nocov * sig2g_2_nocov)):
				print 'WARNING: extreme genetic correlation estimate. Please double check if you correctly specified the number of overlapping individuals'
			
			se_sig2g_1, se_sig2g_2, se_rho, se_corr = jackknife_summary(z1_nocov, args.n1, sig2g1_intercept, args.n1**2 * pcgc_coeff1, z2_nocov, args.n2, sig2g2_intercept, args.n2**2 * pcgc_coeff2, rho_intercept, args.n1*args.n2 * pcgc_coeff12, ld_scores=ld_scores, mean_ld=mean_ld, is_cov=False, num_blocks=args.n_blocks)
			
		print
		print 'Results when excluding covariates'
		print '---------------------------------'
		print 'study 1 h2: %0.4f (%0.4f)'%(sig2g_1_nocov, se_sig2g_1)
		if (args.z2_nocov is not None):
			print 'study 2 h2: %0.4f (%0.4f)'%(sig2g_2_nocov, se_sig2g_2)
			print 'genetic covariance: %0.4f (%0.4f)'%(rho_nocov, se_rho)
			print 'genetic correlation: %0.4f (%0.4f)'%(rho_nocov / np.sqrt(sig2g_1_nocov * sig2g_2_nocov), se_corr)
			
			
		if (args.z1_cov is None): sys.exit(0)



	
	############### included covariates code ###############
	

	#read Gty1_cov and Gty2_cov
	if (args.Gty1_cov is not None):
		df1 = pd.read_csv(args.Gty1_cov, header=None, sep='\s+')
		df1.columns = ('fid', 'iid', 'Gty')
		Gty1_cov = df1['Gty'].values
		assert len(Gty1_cov)==args.n1, 'number of individuals in Gty1_cov file is different from n1 argument'
		if (args.z2_cov is not None):
			df2 = pd.read_csv(args.Gty2_cov, header=None, sep='\s+')
			df2.columns = ('fid', 'iid', 'Gty')
			Gty2_cov = df2['Gty'].values
			assert len(Gty2_cov)==args.n2, 'number of individuals in Gty2_cov file is different from n2 argument'
			df_joined = pd.merge(df1, df2, on=['fid', 'iid'], suffixes=('_1', '_2'))		
			if (keep_df is not None): df_joined = df_joined.isin(keep_df)
			Gty1_cov_overlap = df_joined['Gty_1'].values
			Gty2_cov_overlap = df_joined['Gty_2'].values
			print 'found %d overlapping individuals in files Gty1_cov and Gty2_cov'%(len(Gty1_cov_overlap))
	

	#estimate sig2g_1
	if (args.Gty1_cov is not None):
		sig2g1_intercept = Gty1_cov.dot(Gty1_cov)
		sig2g1_cov_numer = np.sum(z1_cov*z1_cov) / float(z1_cov.shape[0]) - sig2g1_intercept
		sig2g1_cov_denom = args.n1**2 / float(z1_cov.shape[0]) * mean_ld * args.mean_Q1**2
		sig2g1_cov_denom *= args.sqr_geno1_factor
		sig2g_1_cov = sig2g1_cov_numer / sig2g1_cov_denom
		if (args.var_t1 is not None): h2_1_cov = sig2g_1_cov / (1+args.var_t1)
		#print 'numer/denom for study 1:', sig2g1_cov_numer, sig2g1_cov_denom
		#print 'sum G1**2:', args.n1**2 / float(z1_cov.shape[0]) * mean_ld
		if (args.z2_cov is None): se_sig2g_1 = jackknife_summary(z1_cov, args.n1, sig2g1_intercept, args.n1**2 * args.mean_Q1**2, ld_scores=ld_scores, mean_ld=mean_ld, is_cov=True, num_blocks=args.n_blocks)

	
	if (args.z2_cov is not None):
		if (args.Gty2_cov is not None):
			#estimate sig2g_2
			sig2g2_intercept = Gty2_cov.dot(Gty2_cov)
			sig2g2_cov_numer = np.sum(z2_cov*z2_cov) / float(z2_cov.shape[0]) - sig2g2_intercept
			sig2g2_cov_denom = args.n2**2 / float(z2_cov.shape[0]) * mean_ld * args.mean_Q2**2
			sig2g2_cov_denom *= args.sqr_geno2_factor
			sig2g_2_cov = sig2g2_cov_numer / sig2g2_cov_denom
			if (args.var_t2 is not None): h2_2_cov = sig2g_2_cov / (1+args.var_t2)
			#print 'numer/denom for study 2:', sig2g2_cov_numer, sig2g2_cov_denom
		
		
		#estimate rho
		if (args.Gty1_cov is None and args.Gty2_cov is None):
			assert (args.ncon_con==0 and args.ncon_cas==0 and args.ncas_con==0 and args.ncas_cas==0), 'studies have overlapping individuals but Gty files were not provided'
			print 'Gty1_cov and Gty2_cov not provided. Heritabilty and correlation cannot be computed.'
			print 'Genetic covariance can be computed, assuming that there are no overlapping individuals between the two studies...'
			rho_intercept=0
		else:
			rho_intercept = Gty1_cov_overlap.dot(Gty2_cov_overlap)
			
		rho_cov_numer = np.sum(z1_cov*z2_cov) / float(z2_cov.shape[0]) - rho_intercept
		rho_cov_denom = args.n1*args.n2 / float(z1_cov.shape[0]) * mean_ld * args.mean_Q1 * args.mean_Q2
		rho_cov_denom *= np.minimum(args.sqr_geno1_factor, args.sqr_geno2_factor)
		rho_cov = rho_cov_numer / rho_cov_denom
		#print 'numer/denom for rho:', rho_cov_numer, rho_cov_denom
		
		if (args.Gty1_cov is None and args.Gty2_cov is None):
			se_rho = jackknife_summary(z1_cov*z2_cov, np.sqrt(args.n1*args.n2), 0, args.n1*args.n2 * args.mean_Q1 * args.mean_Q2, ld_scores=ld_scores, mean_ld=mean_ld, is_cov=True, num_blocks=args.n_blocks, no_square=True)
		else:		
			se_sig2g_1, se_sig2g_2, se_rho, se_corr = jackknife_summary(z1_cov, args.n1, sig2g1_intercept, args.n1**2 * args.mean_Q1**2, z2_cov, args.n2, sig2g2_intercept, args.n2**2 * args.mean_Q2**2, rho_intercept, args.n1*args.n2 * args.mean_Q1 * args.mean_Q2, ld_scores=ld_scores, mean_ld=mean_ld, is_cov=True, num_blocks=args.n_blocks)
			
		
	print
	print
	print 'Results when including covariates'
	print '---------------------------------'
	if (args.Gty1_cov is not None):
		if (args.var_t1 is None):
			print 'study 1 sig2g: %0.4f (%0.4f) (h2 cannot be determined without var_t1)'%(sig2g_1_cov, se_sig2g_1)
		else:
			print 'study 1 h2: %0.4f (%0.4f) (genetic variance: %0.4f (%0.4f)'%(h2_1_cov, se_sig2g_1/(1+args.var_t1), sig2g_1_cov, se_sig2g_1)
	if (args.z2_cov is not None):
		if (args.Gty2_cov is not None):
			if (args.var_t2 is None):
				print 'study 2 sig2g: %0.4f (%0.4f)  (h2 cannot be determined without var_t2)'%(sig2g_2_cov, se_sig2g_2)
			else:
				print 'study 2 h2: %0.4f (%0.4f) (genetic variance: %0.4f (%0.4f)'%(h2_2_cov, se_sig2g_2/(1+args.var_t2), sig2g_2_cov, se_sig2g_2)
		print 'genetic covariance: %0.4f (%0.4f)'%(rho_cov, se_rho)
		if (args.Gty1_cov is not None and args.Gty2_cov is not None):
			print 'genetic correlation: %0.4f (%0.4f)'%(rho_cov / np.sqrt(sig2g_1_cov * sig2g_2_cov), se_corr)
			
		rho_pvalue_cov = permutation_test(z1_cov, z2_cov, num_perms=10000, chunk_size=1000)
		print
		print 'correlation p-value: %0.5e'%(rho_pvalue_cov)
			
				
	
	
	print
	print
	
	
	
	