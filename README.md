# PCGC-s
Heritability and Genetic Correlation Estimation using Summary Statistics

PCGC-s is a Python package for estimation of genetic heritability, covariance and correlation in case-control studies. PCGC-s can either work directly on genotype and phenotypes files, or it can be invoked using only privacy-preserving summary statistics.
The method can be seen as an adaptation of [LD score regression](http://www.nature.com/ng/journal/v47/n11/full/ng.3406.html) for case-control studies.

<br><br>
# Installation
PCGC-s is designed to work in Python 2.7, and depends on the following freely available Python packages:
* [numpy](http://www.numpy.org/) and [scipy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [PySnpTools](https://github.com/MicrosoftGenomics/PySnpTools)
* [Pandas](https://pandas.pydata.org/getpandas.html)

Typically, the packages can be installed with the command "pip install --user \<package_name\>".
PCGC-s is particularly easy to use with the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda).

Once all the prerequisite packages are installed, PCGC-s can be installed on a git-enabled machine by typing:
```
git clone https://github.com/omerwe/PCGCs
```

The project can also be downloaded as a zip file from the Github website.


<br><br>
# Usage Overview
There are two ways to run PCGC-s. The first is the script `pcgcs_direct.py`, which works directly with
[binary Plink files](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed) and with [phenotype and covariate files written in Plink format](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml). 
The second is the script `pcgcs_summary.py`, which works with summary statistics.

The list of available options for both files can be seen by typing
```
python pcgcs_direct.py --help
python pcgcs_summary.py --help
```

In addition, the file `pcgcs_intermediate.py` can help with the analysis of large data sets that cannot all fit into the computer memory at once, as explained in the **Working with Huge Datasets** section below.



## TL;DR
For an example, please go to the "example" directory and run the following two commands (using the anaconda version of python if available):
```
python ../pcgcs_direct.py \
--bfile1 study1 --pheno1 study1.phe --covar1 study1.cov --prev1 0.01 \
--bfile2 study2 --pheno2 study2.phe --covar2 study2.cov --prev2 0.05 \
--norm maf --maf example.mafs.gz \
--z1_cov_out z1_cov.csv --z2_cov_out z2_cov.csv  \
--Gty1_cov_out study1_cov.Gty --Gty2_cov_out study2_cov.Gty

python ../pcgcs_summary.py \
--ref-ld example.l2.ldscore.gz \
--n1 2000 --n2 2000 \
--z1_cov z1_cov.csv.gz --z2_cov z2_cov.csv.gz \
--Gty1_cov study1_cov.Gty --Gty2_cov study2_cov.Gty \
--mean_Q1 1.1540  --mean_Q2 0.9452   \
--var_t1 0.3302 --var_t2 0.3044
```
The first command will invoke PCGC to directly estimate the heritability of the two studies and the between-study genetic correlation.

The second command will use summary statistics produced by the first command to approximate the same quantities. Although of course there is no need to approximate these quantities if the individual-level data is provided, this demonstration shows that PCGC-s can produce accurate estimates without having access to raw genetic data.



<br><br>
# Detailed Instructions

<br><br>
## Computing Heritability and Genetic Correlation Directly
The file `pcgcs_direct.py` directly estimates heritability and genetic correlation.
This file can accept either one or two plink files representing distinct genetic studies. If only one file is provided, `pcgcs_direct.py` will estimate heritability. If two files are provided, it will also estimate heritability for the second study and the genetic correlation between the studies. Standard errors can be computed via a jackknife over individuals, by specifying `--jackknife 1` (but this will slow down the analysis considerably).
The command-line arguments can be broken down into several categories, which we now describe:

#### Raw Data:
```
--bfile1 <plink file> --pheno1 <phenotypes file> --covar1 <covariates file> --prev1 <trait 1 prevalence>
```
These fields will specify a plink file with genetic data, a phenotypes file, a covariates file and the prevalence of trait 1, respectively. The phenotypes file must contain only two phenotype values (preferably 0/1 for controls/cases).
The same fields may also be specified for a second study (using `--bfile2`, `--pheno2`, etc).

#### SNP normalization options:
```
--norm <norm_option> --maf <maf file>
```
These will specify how PCGC-s will normalize SNPs prior to computing kinship matrices. Although it is very common to normalize SNPs using their in-sample frequencies, this is very problematic when estimating genetic correlation for case-control studies, because different normalizations in the two studies can lead to very biased estimates.
**It it strongly recommended to normalize SNPs according to their minor allele frequencies (MAFs) using estimates from a reference population, such as the 1000 genomes project**.

The `--norm` field currently accepts three options: "bed" will use in-sample normalization, "both" will use  in-sample estimates estimated from the two files jointly (while also accounting for the trait prevalence) and "maf" will use an external file with MAF estimates. **The recommended option is to use the maf option.** This requires also using  the `--maf` option to provide a csv file with two columns: A column called "snpid" with SNP names and a column called "maf" with MAF estimates. Please see the use example for one such file.

#### Summary statistics options:
```
--sumstats_only 0/1
```
If turned on, `PCGCs_direct.py` will only compute summary statistics without computing the actual heritability and genetic correlation estimates and their standard errors. This will be much faster than the full computation.

#### SNP weight options:
```
--snp_weights <snp_weights file>
```
This option allows assigning different weights to different SNPs. This can be be useful, for example, if one wants to use the [LDAK model](https://www.nature.com/articles/ng.3865), wherein [each SNP is weighted according to its MAF and its LD patterns](http://dougspeed.com/get-weightings/). This file should include two tab-delimited columns: SNP name and SNP weight (no header is required). This option is currently supported only for `pcgcs_direct.py`. Support for `pcgcs_summary.py` will come in the future.

#### Memory utilization options:
```
--mem_size <# of individuals>
```
`pcgcs_direct.py` avoids computing large kinship matrices to save memory. Instead, it computes intermediate kinship matrices of size `mem_size x n` (rather than full kinship matrices of size `n x n`). This is useful for very large studies, where the kinship matrix is very large. By default `mem size = 1000`. Using smaller values will require less memory but may increase the run-time. Note that `pcgcs_direct.py` stores the entire contents of the plink files in memory, which may themselves be very large.

```
--snp1 <snp number>
--snp2 <snp number>
```
These options tell `pcgcs_direct.py` to compute summary statistics only for the range of SNPs snp1-snp2 (the first SNP has the number 1). This is useful for large data sets that cannot fit in the computer memory. The summary statistics can then be joined together and then analyzed with `pcgcs_direct.py`, as explained in the **Working with Huge Datasets** section below.



#### Permutation testing:
```
--num_perms <number of permutations>
```
This flag controls permutation testing to test the hypothesis that the true genetic correlation is 0. Permutation testing is turned off by default. A larger number of permutations enable obtaining more significant p-values but take up more computation time.


<br><br>
## Creating Summary Statistics
The file `pcgcs_direct.py` can not only estimate heritability and genetic correlation, but can also create summary statistics that can be used to estimate these quantities without having access to genetic data. This is especially useful if two research groups wish to collaborate to estimate genetic correlation between traits, but cannot share genetic data with each other.
The following categories of options are used to create summary statistic files:

#### Z-score output options:
```
--z1_nocov_out <file_name> --z1_cov_out <file_name>
```
These options will tell PCGC-s to output z-score summary statistics, similarly to LD score regression. The first option ignores measured covariates, while the second option outputs a covariates-aware z-scores file. Both options can be be used simultaneously.
Similar options exist also for study 2 (`--z2_nocov_out` and `--z2_cov_out`).

#### intercept output options:
```
--Gty1_nocov_out <file name> --Gty1_cov_out <file name>
```
These two output files contain individual-specific information required to compute the second term of the numerator of the PCGC estimator (the so-called intercept of LD score regression). `--Gty1_nocov_out` creates a file that ignores covaraites, whereas `-Gty1_cov_out` creates a covariates-aware file.

The information in the file `Gty1_nocov_out` is not strictly required for subsequent analysis, but including it may result in increased accuracy. However, the information in  `Gty1_cov_out` must be provided if one wishes to estimate genetic correlation between studies with overlapping individuals. Note that these files expose a (noisy version of) the phenotypes of individuals, as well as some information about their covariates. In practice only information about overlapping individuals between two studies is strictly required: One may delete information about other individuals from the output files if privacy is a concern.
Similar options can be provided for study 2.








<br><br>
## Inclusion of Covariates
Genetic studies often include covariates that represent major risk factors, such as age and sex.
Accounting for such covariates can improve estimation accuracy, while omitting such covariates can sometimes lead to biased estimates.
PCGC-s makes a major distinction between analyses that omit covariates and those that include them.
Omitting measured covariates can often decrease estimation accuracy, especially if they have large effects.
On the other hand, accounting for covariates requires specialized summary statistics and requires exposing some information about overlapping individuals.

If the file `pcgcs_direct.py` is invoked with the options `--covar1` and `--covar2`, it will estimate both covariate-less and covariate-aware estimates. It will additionally print some quantities required for subsequent analysis with the summary statistic files it produces.

Estimating heritability and genetic correlation with `pcgcs_summary.py` requires specialized z-score files created with `pcgcs_direct.py`. Additionally, it requires the summary statistics `mean_Q1` and `mean_Q2` reported in the output of `pcgcs_direct.py`. Finally, it requires individual-specific phenotype information in the files `Gty1_cov` and `Gty2_cov` that can be created with `pcgcs_direct.py`. The next section explains the use of these files.





<br><br>
## Computing Heritability and Genetic Correlation with Summary Statistics

As explained above, there is a major distinction between analyses that omit covariates and those that include them. `pcgcs_direct.py` can carry out both types of analyses. We first describe covariate-less analyses, and then proceed to describe covariate-aware analyses. We note that the PCGC-s paper describes a method to compute exact estimates (without approximation) by computing the covariance between every pair of variants in the study. However, this is a very computationally expensive operation and is currently not implemented in the software.


#### covariate-less analysis with summary statistics
Heritability and genetic correlation estimation without covariates requires several pieces of information:
 1. z-score files. These are provided with the arguments `--z1_nocov` and `--z2_nocov`. Ideally these files should be created by `pcgcs_direct.py` using the flags `--z1_nocov_out`, `--z2_nocov_out`. Other software can also be used to create these files. In this case, it is highly recommended to first standardize the z-score files using the script `munge_sumstats.py` from the [ldsc package](https://github.com/bulik/ldsc). 
 If only `--z1_nocov` is provided, PCGC-s will estimate heritability. If `--z2_nocov` is also provided, PCGC-s will also estimate heritability for the second study and genetic correlation between the two studies.
 2. LD information: Heritability and genetic correlation estimation with summary statistics requires LD information. These can be specified in two ways: The first is to specify only the mean LD score of the genetic variants in the analysis, using the `--mean-ld` flag. The second is to provide a file with LD scores of every variant via the options `--ref-ld`. See the example directory for one such file. This must be a csv file with a column called 'snpid' and a column called 'L2'. If possible, it is highly recommended to estimate LD scores in-sample using the [ldsc tool](https://github.com/bulik/ldsc). Using `--ref-ld` will provide slightly more accurate standard error estimates.
 3. Trait prevalence: These can be specified with the arguments `--prev1` and `--prev2`.
 4. Fraction of cases in the studies: These can be specified with the flags `--P1` and `--P2`.
 5. Study sizes: These can be specified with the flags `--n1` and `--n2`.
 6. Information regarding overlapping individuals. This can be specified in several ways:
	1. The best way to provide information regarding overlapping individuals is via Gty files. These can be produced by `pcgcs_direct.py` via the arguments `--Gty1_nocov_out`, `--Gty2_nocov_out`, and can be  provided to `pcgcs_summary.py` via the arguments `--Gty1_nocov`, `--Gty2_nocov`.
	2. If Gty files are not available, it is possible to specify phenotype files with the flags `--pheno`, `--pheno2`. The overlapping individuals between the two files will be considered the overlapping individuals for the analysis. If only these two files are provided, PCGC-s will proceed by assuming that the kinship coefficient of an individual with herself is 1.0. It is possible to use a more accurate estimate by also providing a `--bfile` flag with genotypes of overlapping individuals if available. However, note that it is very important to normalize the SNPs in the plink file in exactly the same way as the normalization in the analysis that created the z-score files. The script `pcgcs_summary.py` therefore supports the same SNP normalization options as `pcgcs_direct.py`. A subset of the individuals in the plink/phenotypes can be used via the `--keep` flag.
	3. Finally, another way to inform PCGC-s about overlapping individuals is using the flags `--ncon-con`, `--ncon-cas`, `--ncas-con` and `--ncas-cas`, which specify the number of control-control, control-case, case-control and case-case overlapping individuals, respectively.
	4. If none of these options is specified, PCGC-s will assume that there are no overlapping individuals.
	
<br><br>
#### covariate-aware analysis with summary statistics
Heritability and genetic correlation estimation with covariates requires several pieces of information:
 1. z-score files: These must be covariate-aware z-score files computed via `pcgcs_direct.py` with the flags `--z1_cov_out`, `--z2_cov_out`. **Other software cannot be used to create these files**. The files are provided to `pcgcs_summary.py` with the arguments `--z1_cov` and `--z2_cov`. If only `--z1_cov` is provided, PCGC-s will estimate heritability. If `--z2_nocov` is also provided, PCGC-s will also estimate heritability for the second study and genetic correlation between the two studies. 
 2. LD information: This can be provided exactly like explained above, via either the `--ref-ld` flag or via the `--mean-ld` flag.
 3. Study sizes: These can be specified with the flags `--n1` and `--n2`.
 4. Gty files: These are required for heritability estimation, and also for genetic correlation estimation in the presence of overlapping individuals. These can be created by `pcgcs_direct.py` via the flags `--Gty1_cov_out`, `--Gty2_cov_out`, and can be provided to `pcgcs_summary.py` with the flags `--Gty1_cov`, `--Gty2_cov`. If these are not provided, covariate-aware heritability cannot be estimated, and genetic covariance can only be estimated under the assumption that there are no overlapping individuals. The set of overlapping individuals is inferred from these two files.
 5. mean Q values: These are the mean diagonal values of Q for the studies (see the paper for a definition), and can be provided using the flags `--mean_Q1` and `--mean_Q2`. These values are printed out by `pcgcs_direct.py` when invoked.
 6. Liability variance due to covariates: PCGC estimates genetic variance, which is different from heritability in the presence of covariates. Converting this estimate to heritability in the presence of covariates requires dividing the genetic variance estimate by `1+var(t)`, where `var(t)` is the estimate of the liability variance due to measured covariates. This information cannot be computed via summary statistics, but can be provided to `pcgcs_summary.py` via the flags `--var_t1`, `--var_t2`. These values are printed out by `pcgcs_direct.py`. If these values are not provided, `pcgcs_summary.py` will only report the genetic variance estimates and not the heritability estimates.
	
	

<br><br>
## Regression of Principal Components
It is sometimes desirable to include principal components (PCs) as covariates in the analysis to prevent possible confounding due to population structure. We recommend computing principal components via external software (e.g. [FlashPCA2](https://github.com/gabraham/flashpca)) and including them as additional covariates in the covariates file. 

A particular complexity of case-control studies is that the PCs are reflected in the kinship matrix entries, which can bias the estimation. It is therefore recommended to regress genotype vectors to the subspace that is orthogonal to the leading PCs. This can be done in `pcgcs_direct.py` via the flags `--PC1 <i1,i2,...,im>`, `--PC2 <i1,i2,...,im>`, where the arguments are a comma-separated list of covariate indices that are principal components (starting from 1). For example, if a covariates file includes 5 PCs and a sex covariate, the flag should be specified as `--PC1 1,2,3,4,5`.

Unfortunately, regression of PCs out of genotype vectors leads to biased estimates if not properly accounted for, because the regression shrinks all kiship coefficients towards zero. To see this, consider the fact that the sum of the squared kinship coefficients is equal to the sum of the squares of the eigenvalues of the kinship matrix; hence setting the leading eigenvalues to zero will shrink the sum of the squared kinship coefficients. To account for this deflation, `pcgcs_direct.py` will report the deflation factors in its output when one of the flags `--PC1`, `--PC2`, is invoked. Afterwards, the reported deflation factors should be passed to `pcgcs_summary.py`. The flag names are `--geno1_factor`, `--sqr_geno1_factor`, which report the deflation in the sum of the eigenvalues and in the sum of the squares of the eigenvalues (which corresponds to the sum of the diagonal of the kinship matrix and the sum of the squares of all kinship coefficients, respectively).
Similar quantities will also be reported for study 2, and should be passed as well with the flags `--geno2_factor`, `--sqr_geno2_factor`.


<br><br>
## Working with Huge Datasets
`pcgcs_direct.py` may have difficulty loading the entire matrix of genotypes into memory at once. To alleviate this concern, it is possible to compute summary statitics for different subsets of SNPs, and then concatenate them together and analyze the concatenated files with `pcgcs_summary.py`. This can be done with the flags `--snp1 <snp number>`, `--snp2 <snp number>`, as explained above. For example, one can run the following commands:

`pcgcs_direct.py --bfile <bfile> --pheno <pheno> --covar <covar> --snp1 1 --snp2 50000 --z1_cov_out z1_1_50000.csv`

`pcgcs_direct.py --bfile <bfile> --pheno <pheno> --covar <covar> --snp1 50001 --snp2 100000 --z1_cov_out z1_50001_100000.csv`

`zcat z1_1_50000.csv.gz > z1_combined.csv`

`zcat z1_50001_100000.csv | tail -n +2 >> z1_combined.csv`

These commands will invoke `pcgcs_direct.py`, once for SNPs 1-50000 and once for SNPs 500001-100000. The next two commands will concatenate the summary statistics together into `z1_combined.csv` (while making sure to only include a single header from the first file).

When this is done, we still need to compute several required quantities, such as Gty files. This can be done using the script `pcgcs_intermediate.py`. This script accepts many of the same arguments as `pcgcs_direct.py` (as can be examined with the command `python pcgcs_intermediate.py --help`). The most important argument is `--mem_size <#SNPs>`, which will limit the number of SNPs loaded into memory at once. The second new argument that should be provided is `--eigenvalues_frac <f1,f2,...,fm>`. This argument provides a comma-separated list of fraction of variance explained by the PCs that are specified using the flag `--PC`. These are provided by [FlashPCA2](https://github.com/gabraham/flashpca) in the file pve.txt.

A typical use example is:

`python pcgcs_intermediate.py --bfile <plink file> --pheno <pheno file> --covar <covariates file> --prev <trait prevalence> --norm maf --maf <MAFs file> --PC 1,2,3,4,5,6,7,8,9,10 --Gty_nocov_out Gty1_nocov.txt --Gty_cov_out Gty1_cov.txt --ref-ld <ref-ld file> --eigenvalues_frac 0.0013,0.00066,0.00048,0.00038,0.00037,0.00035,0.00034,0.00034,0.00034,0.00033`


After finising running `pcgcs_intermediate.py`, we can continue to run `pcgcs_summary.py` as usual.


<br><br>
# Important notes
1. Overlapping individuals (shared between the two studies) will not be automatically detected. Please make sure that overlapping individuals are clearly marked in the plink files by having exactly the same family id and individual id.

2. `pcgcs_direct.py` attemps to avoid storing large matrices in memory, and in particular avoids computing kinship matrices. Instead, it computes intermediate matrices of size `w x n`, where `w` is the `mem_size` parameter and `n` is the study size. However, it keeps the full contents of the plink files in memory, which may itself take up large amounts of memory. If this is a problem, please refer to the **Working with Huge Datasets** section above.


<br><br>
-----------------
Contact
---------
For questions and comments, please contact Omer Weissbrod at omerw[at]cs.technion.ac.il



