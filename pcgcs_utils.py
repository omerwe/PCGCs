import numpy as np
from optparse import OptionParser
import scipy.linalg as la
import scipy.stats as stats
import sys
import scipy.linalg.blas as blas
import pandas as pd
import time
from pysnptools.snpreader.bed import Bed
import pysnptools.util as pstutil
import pysnptools.util.pheno as phenoUtils
np.set_printoptions(precision=3, linewidth=200)
import resource
import copy
from sklearn.linear_model import LinearRegression, LogisticRegression


def memory_usage_resource():
	rusage_denom = 1024.
	if sys.platform == 'darwin':
		# ... it seems that in OSX the output is different units ...
		rusage_denom = rusage_denom * rusage_denom
	mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
	return mem
	
def print_memory_usage(message=None):
	return
	if (message is not None): print message,
	print 'memory: %0.5e'%(memory_usage_resource())
	
	
	
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
	#Q = np.outer(u0,u0) + np.outer(u0,u1) + np.outer(u1,u0) + np.outer(u1,u1)
	
	ty = (y-Pi) / np.sqrt(Pi * (1-Pi))
	
	if return_intermediate:
		return K, P, Ki, Pi, phi_tau_i
	
	return y_norm, tau_i, pcgc_coeff, ty, u0, u1
	
	



def loadData(bfile, extractSim, phenoFile, missingPhenotype='-9', loadSNPs=False, keep=None, standardize=False, fileNum=None, snp1=None, snp2=None):
	try: bed = Bed(bfile, count_A1=True)
	except: bed = Bed(bfile)
	
	if (snp1 is not None):
		assert snp1>=1, 'snp1 must be >=1'
		assert snp2 is not None
		assert (snp2 <= len(bed.sid)), 'snp2 cannot be greater than %d'%(len(bed.sid))
		bed = bed[:, (snp1-1):snp2]
	
	if (extractSim is not None):
		try:
			extractSim.union(extractSim)
			extractSnpsSet = extractSim
		except:
			df = pd.read_csv(extractSim, header=None)
			assert len(df.columns)==1, 'extract file must have exactly one column'
			extractSnpsSet = set([s.strip() for s in df.values[:,0]])
		keepSnpsInds = [i for i in xrange(bed.sid.shape[0]) if bed.sid[i] in extractSnpsSet]		
		if (fileNum is None): print 'keeping %d SNPs in bfile'%(len(keepSnpsInds))
		else: print 'keeping %d SNPs in bfile %d'%(len(keepSnpsInds), fileNum)		
		bed = bed[:, keepSnpsInds]
		
	#Remove individuals not in keep file
	if (keep is not None):
		df = pd.read_csv(keep, delimiter='\s+', header=None)
		assert len(df.columns)==2, 'keep file should have exactly two columns'
		keepDict = set([(r[0], r[1]) for r in df.values])
		keepInds = [i for i in xrange(bed.iid.shape[0]) if (bed.iid[i,0], bed.iid[i,1]) in keepDict]
		print 'keeping %d individuals in bfile'%(len(keepInds))
		bed = bed[keepInds, :]
		
		
		
	phe = None
	if (phenoFile is not None):	bed, phe = loadPheno(bed, phenoFile, missingPhenotype)
	
	if loadSNPs:
		bed = bed.read()
		if (standardize): bed = bed.standardize()	
	
	return bed, phe
	
	
def loadPheno(bed, phenoFile, missingPhenotype='-9', keepDict=False):
	pheno = phenoUtils.loadOnePhen(phenoFile, missing=missingPhenotype, vectorize=True)
	checkIntersection(bed, pheno, 'phenotypes')
	bed, pheno = pstutil.intersect_apply([bed, pheno])
	if (not keepDict): pheno = pheno['vals']
	return bed, pheno
	
	
def checkIntersection(bed, fileDict, fileStr, checkSuperSet=False):
	bedSet = set((b[0], b[1]) for b in bed.iid)
	fileSet = set((b[0], b[1]) for b in fileDict['iid'])
	
	if checkSuperSet:
		if (not fileSet.issuperset(bedSet)): raise Exception(fileStr + " file does not include all individuals in the bfile")
	
	intersectSet = bedSet.intersection(fileSet)
	if (len(intersectSet) != len (bedSet)):
		print len(intersectSet), 'individuals appear in both the plink file and the', fileStr, 'file'

	
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())
	
	

def loadCovars(bed, covarFile):
	covarsDict = phenoUtils.loadPhen(covarFile)
	checkIntersection(bed, covarsDict, 'covariates', checkSuperSet=True)
	_, covarsDict = pstutil.intersect_apply([bed, covarsDict])
	covar = covarsDict['vals']
	return covar	
	

def _fixupBedAndPheno(bed, pheno, missingPhenotype='-9'):
	bed = _fixupBed(bed)
	bed, pheno = _fixup_pheno(pheno, bed, missingPhenotype)
	return bed, pheno
	
def _fixupBed(bed):
	if isinstance(bed, str):
		return Bed(bed).read()
	else: return bed

def _fixup_pheno(pheno, bed=None, missingPhenotype='-9'):
	if (isinstance(pheno, str)):
		if (bed is not None):
			bed, pheno = loadPheno(bed, pheno, missingPhenotype, keepDict=True)
			return bed, pheno
		else:
			phenoDict = phenoUtils.loadOnePhen(pheno, missing=missingPhenotype, vectorize=True)
			return phenoDict
	else:
		if (bed is not None): return bed, pheno			
		else: return pheno

		
#Mean-impute missing SNPs	
def imputeSNPs(X):

	#fast code, but not memory efficient
	snpsMean = np.nanmean(X, axis=0)
	# isNan = np.isnan(X)
	# X[isNan]=0
	# X += isNan*snpsMean
	
	#slower code but more memory efficient
	for i,m in enumerate(snpsMean):		
		X[np.isnan(X[:,i]), i] = m
		
	return X
	

def read_bed_lowmem(bed):
	X = np.empty((bed.iid.shape[0], bed.sid.shape[0]), dtype=np.float32)
	batch_size = 1000
	
	for i in xrange(0, bed.iid.shape[0], batch_size):
		bed_copy = copy.deepcopy(bed)
		bed_copy = bed_copy[i:i+batch_size, :]
		bed_copy=bed_copy.read()
		X[i:i+batch_size, :] = bed_copy.val
	return X
	
	
	
#Regress top PCs out of the genotypes matrix
def regress_PCs(snps, numPCs):	
	XXT = symmetrize(blas.dsyrk(1.0, snps, lower=0))
	s,U = la.eigh(XXT, eigvals=(snps.shape[0]-numPCs, snps.shape[0]-1))
	if (np.min(s) < -1e-4): raise Exception('Negative eigenvalues found')
	U_top = U
	snps -= (U_top.dot(U_top.T)).dot(snps)
	
	s /= snps.shape[1]
	sum_s = np.trace(XXT) / snps.shape[1]
	sum_s_sqr = np.sum(XXT**2) / snps.shape[1]**2
	
	return snps, U_top, s, sum_s, sum_s_sqr
	

#Read and preprocess the data
def read_SNPs(bfile1, pheno1, prev1, covar1=None, keep1=None, bfile2=None, pheno2=None, prev2=None, covar2=None, keep2=None, extract=None, missingPhenotype='-9', chr=None, norm=None, maf=None, center=False, lowmem=True, snp1=None, snp2=None):

	print	
	print
	print 'Reading and preprocessing data from plink files'
	print '----------------------------------------'

	#read metadata
	bed1, phe1 = loadData(bfile1, extract, pheno1, missingPhenotype, loadSNPs=False, standardize=False, keep=keep1, fileNum=1, snp1=snp1, snp2=snp2)
	assert len(np.unique(phe1)==2), 'phenotypes file 1 is not case-control data'
	if (bfile2 is None):
		bed2, phe2, X2 = None, None, None
	else:
		assert snp1 is None
		bed2, phe2 = loadData(bfile2, extract, pheno2, missingPhenotype, loadSNPs=False, standardize=False, keep=keep2, fileNum=2)	
		assert len(np.unique(phe2)==2), 'phenotypes file 2 is not case-control data'
		
		#remove non-shared SNPs
		snps1 = set(list(bed1.sid))
		snps2 = set(list(bed2.sid))
		snps_remove = snps1 ^ snps2
		if (len(snps_remove) > 0):
			print 'removing %d non-overlapping SNPs'%(len(snps_remove))
			inds_to_remove1 = [i for i in xrange(len(bed1.sid)) if bed1.sid[i] in snps_remove]
			inds_to_remove2 = [i for i in xrange(len(bed2.sid)) if bed2.sid[i] in snps_remove]
			should_keep_snps1 = np.ones(len(bed1.sid), dtype=np.bool); should_keep_snps1[inds_to_remove1]=False
			should_keep_snps2 = np.ones(len(bed2.sid), dtype=np.bool); should_keep_snps2[inds_to_remove2]=False
			bed1 = bed1[:, should_keep_snps1]
			bed2 = bed2[:, should_keep_snps2]
		assert np.all(bed1.sid == bed2.sid)
		assert np.all(bed1.pos[:,0] == bed2.pos[:,0])
		assert np.allclose(bed1.pos, bed2.pos)
	
	#exclude SNPs not on requested chromosome
	if (chr is not None):
		print 'keeping only chromosome %d SNPs'%(chr)
		bed1 = bed1[:, bed1.pos[:,0]==chr]
		print 'Remaining number of SNPs in bfile 1:', bed1.sid.shape[0]
		if (bfile2 is not None):
			bed2 = bed2[:, bed2.pos[:,0]==chr]
			print 'Remaining number of SNPs in bfile 2:', bed2.sid.shape[0]

	if lowmem: X1 = read_bed_lowmem(bed1)
	else:
		bed1=bed1.read()
		X1 = bed1.val
	if (bfile2 is not None):
		if lowmem: X2 = read_bed_lowmem(bed2)
		else:
			bed2=bed2.read()
			X2 = bed2.val
		
		#align different strands...		
		t0 = time.time()
		num_diff = np.zeros(bed1.sid.shape[0], dtype=np.int)
		for i1, ind1 in enumerate(bed1.iid[:,1]):
			for i2, ind2 in enumerate(bed2.iid[:,1]):
				if (ind1 != ind2): continue					
				snps1 = X1[i1]
				snps2 = X2[i2]
				diff_spots = ((~np.isnan(snps1)) & (~np.isnan(snps2)) & (snps1 != snps2))
				num_diff[diff_spots] += 1
		
		is_diff_strand = (num_diff>0)			
		if (is_diff_strand.sum() > 0):
			
			print 'using overlapping individuals to align potentially opposite strands in the two files...'
			print 'found %d SNPs with different strands'%(is_diff_strand.sum())

			#sanity check
			for i1, ind1 in enumerate(bed1.iid[:,1]):
				for i2, ind2 in enumerate(bed2.iid[:,1]):
					if (ind1 != ind2): continue
					snps1 = X1[i1, is_diff_strand]
					snps2 = X2[i2, is_diff_strand]
					
					keep = (~np.isnan(snps1) & (~np.isnan(snps2)) & (snps1!=1) & (snps2!=1))				
					if (np.sum(keep) == 0): continue
					snps1 = snps1[keep]
					snps2 = snps2[keep]
					assert np.all(snps1 != snps2), 'the two files have non-consistent strand information. Please double check your input for errors'
					
			X2[:, is_diff_strand] *= (-1)
			X2[:, is_diff_strand] += 2
			
			print 'strand alignment done in %0.2f seconds'%(time.time() - t0)

	print_memory_usage(1)
		
	#load covariates
	if (covar1 is None): cov1 = None
	else:
		cov1 = loadCovars(bed1, covar1)
		print 'loading covariates file', covar1
		for i in xrange(cov1.shape[1]):
			cov_i = cov1[:,i]
			cov1[np.isnan(cov_i), i] = np.nanmean(cov_i)				
			#cov1 -= cov1.mean(axis=0)
			#cov1 /= cov1.std(axis=0)
	
	if (covar2 is None): cov2 = None
	else:
		cov2 = loadCovars(bed2, covar2)
		print 'loading covariates file', covar2
		for i in xrange(cov2.shape[1]):
			cov_i = cov2[:,i]
			cov2[np.isnan(cov_i), i] = np.nanmean(cov_i)				
		#cov2 -= cov2.mean(axis=0)
		#cov2 /= cov2.std(axis=0)
			
	
	#standardize SNPs
	if (norm=='bed' or norm is None):
	
		#impute SNPs (separately for cases and controls)
		print 'imputing SNPs...'
		X1[phe1>phe1.mean(), :] = imputeSNPs(X1[phe1>phe1.mean(), :])
		X1[phe1<=phe1.mean(), :] = imputeSNPs(X1[phe1<=phe1.mean(), :])
		assert np.all(~np.isnan(X1))
		if (bfile2 is not None):
			X2[phe2>phe2.mean(), :]  = imputeSNPs(X2[phe2>phe2.mean(), :])
			X2[phe2<=phe2.mean(), :] = imputeSNPs(X2[phe2<=phe2.mean(), :])
			assert np.all(~np.isnan(X2))
	
		print 'WARNING: normalizing SNPs using in-sample MAFs (highly unrecommended for case-control studies)'
		#bed1=bed1.standardize()
		X1_nanmean = np.nanmean(X1, axis=0)
		num_nan = np.sum(np.isnan(X1_nanmean))
		if (num_nan > 0):		
			print '%d SNPs have 0 MAF'%(num_nan)
			###print bed1.sid[np.where(np.isnan(X1_nanmean))[0][:10]]
		X1 -= X1_nanmean		
		X1_std = np.nanstd(X1, axis=0)
		X1_std[X1_std==0] = 1		
		X1 /= X1_std
		assert np.all(~np.isnan(X1))
		if (bfile2 is not None):
			#bed2=bed2.standardize()
			X2 -= np.nanmean(X2, axis=0)
			X2_std = np.nanstd(X2, axis=0)
			X2_std[X2_std==0] = 1
			X2 /= X2_std
			assert np.all(~np.isnan(X2))
		
		
		print_memory_usage(2)
		
	elif (norm == 'maf'):
		assert maf is not None, 'maf file must be specified for "--norm maf"'
	
		#impute SNPs (separately for cases and controls)
		print 'imputing SNPs...'
		X1[phe1>phe1.mean(), :] = imputeSNPs(X1[phe1>phe1.mean(), :])
		X1[phe1<=phe1.mean(), :] = imputeSNPs(X1[phe1<=phe1.mean(), :])
		assert np.all(~np.isnan(X1))
		if (bfile2 is not None):
			X2[phe2>phe2.mean(), :]  = imputeSNPs(X2[phe2>phe2.mean(), :])
			X2[phe2<=phe2.mean(), :] = imputeSNPs(X2[phe2<=phe2.mean(), :])
			assert np.all(~np.isnan(X2))
	
		#read MAFs
		df = pd.read_csv(maf, delimiter='\s+')		
		is_snp_col = np.array([('snp' in c.lower() or 'rs' in c.lower()) for c in df.columns])
		if (is_snp_col.sum() == 0):
			raise Exception('no column in MAF file match looks like SNP names')
		if (is_snp_col.sum() > 1):
			print 'multiple colmns have potential SNP information:'
			for b_i, b in enumerate(is_snp_col):
				if b: print df.columns[b_i]
			raise Exception('ambiguous SNP column')		
			
		is_maf_col = np.array([('maf' in c.lower() or 'frq' in c.lower() or 'freq' in c.lower()) for c in df.columns])		
		if (is_maf_col.sum() == 0):
			raise Exception('no column in MAF file match looks like it has MAF data')
		if (is_maf_col.sum() > 1):
			print 'multiple colmns have potential MAF information:'
			for b_i, b in enumerate(is_maf_col):
				if b: print df.columns[b_i]
			raise Exception('ambiguous MAF column')
			
		df_maf_snps = df.values[:, np.where(is_snp_col)[0][0]]
		df_mafs = df.values[:, np.where(is_maf_col)[0][0]]
		snp_to_maf = dict([(df_maf_snps[i], df_mafs[i]) for i in xrange(len(df_mafs))])
		
		mafs = np.empty(bed1.sid.shape[0]); mafs[:] = np.nan
		for sid_i,sid in enumerate(bed1.sid):
			if (sid in snp_to_maf): mafs[sid_i] = snp_to_maf[sid]
		if (np.any(np.isnan(mafs))):
			print 'removing %d SNPs with no MAF information'%(np.sum(np.isnan(mafs)))
		bed1 = bed1[:, ~np.isnan(mafs)]
		X1 = X1[:, ~np.isnan(mafs)]
		if (bfile2 is not None):
			bed2 = bed2[:, ~np.isnan(mafs)]
			X2 = X2[:, ~np.isnan(mafs)]
		mafs = mafs[~np.isnan(mafs)]
		assert np.all(mafs>0.005)
		assert np.all(mafs<0.51)
		X1 -= 2*mafs
		X1 /= np.sqrt(2*mafs*(1-mafs))
		if (bfile2 is not None):			
			X2 -= 2*mafs
			X2 /= np.sqrt(2*mafs*(1-mafs))
		assert np.all(~np.isnan(X1))
		if (bfile2 is not None):
			assert np.all(~np.isnan(X2))
		
		
	elif (norm=='both'):
		assert bfile2 is not None, '"both" normalization cannot be used without two plink files'
		t0 = time.time()
		#estimate MAFs (before imputation)
		print 'estimating in-sample MAFs...'
		maf1_controls = np.nanmean(X1[phe1<=phe1.mean()], axis=0)
		maf1_cases = np.nanmean(X1[phe1>phe1.mean()], axis=0)
		maf1 = prev1*maf1_cases + (1-prev1)*maf1_controls
		
		maf2_controls = np.nanmean(X2[phe2<=phe2.mean()], axis=0)
		maf2_cases = np.nanmean(X2[phe2>phe2.mean()], axis=0)
		maf2 = prev2*maf2_cases + (1-prev2)*maf2_controls
		mafs = (bed1.iid.shape[0]*maf1 + bed2.iid.shape[0]*maf2) / (bed1.iid.shape[0] + bed2.iid.shape[0])
		mafs /= 2.0
		assert np.all(mafs>0)
		assert np.all(mafs<1)
		
		#impute SNPs (separately for cases and controls)
		print 'imputing SNPs...'
		X1[phe1>phe1.mean(), :]  = imputeSNPs(X1[phe1>phe1.mean(), :])
		X1[phe1<=phe1.mean(), :] = imputeSNPs(X1[phe1<=phe1.mean(), :])
		X2[phe2>phe2.mean(), :]  = imputeSNPs(X2[phe2>phe2.mean(), :])
		X2[phe2<=phe2.mean(), :] = imputeSNPs(X2[phe2<=phe2.mean(), :])
		
		#normalize SNPs
		print 'normalizing SNPs with "both" method...'
		X1 -= 2*mafs
		X1 /= np.sqrt(2*mafs*(1-mafs))
		X2 -= 2*mafs
		X2 /= np.sqrt(2*mafs*(1-mafs))
		
		print 'total normalization time: %0.2f'%(time.time() - t0)
		
	elif (norm == 'none'):
		pass

	else:
		raise Exception('unknown SNP normalization specified')
	
	
	#center SNPs
	if center:
		print 'centering SNPs...'
		X1 -= X1.mean(axis=0)
		if (bfile2 is not None): X2 -= X2.mean(axis=0)	
		
	
	print 'done'
	print '----------------------------------------'
	print
	
	print_memory_usage(3)
	
	return X1, bed1, phe1, cov1, X2, bed2, phe2, cov2
