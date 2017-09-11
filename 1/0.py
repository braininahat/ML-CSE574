# Author: Varun Shijo
# UB CSE 574
# Person Number 50244968
# Referred extensively to the official documentation of libs
# https://pandas.pydata.org/pandas-docs/
# https://docs.scipy.org/doc/numpy-1.13.0/reference/

import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_excel('university data.xlsx')

# selecting the four columns of interest
columns = df.columns[2:6]

# Computing the means for named columns
means = df[columns].mean()
# mu1 = means[0] # and so on but keeping the more verbose assignment
# for readability
mu1 = df['CS Score (USNews)'].mean()
mu2 = df['Research Overhead %'].mean()
mu3 = df['Admin Base Pay$'].mean()
mu4 = df['Tuition(out-state)$'].mean()

# Computing variance for named columns
variances = df[columns].var()

var1 = df['CS Score (USNews)'].var()
var2 = df['Research Overhead %'].var()
var3 = df['Admin Base Pay$'].var()
var4 = df['Tuition(out-state)$'].var()

# Computing standard deviation for named columns
sigmas = df[columns].std()
sigma1 = df['CS Score (USNews)'].std()
sigma2 = df['Research Overhead %'].std()
sigma3 = df['Admin Base Pay$'].std()
sigma4 = df['Tuition(out-state)$'].std()

# Calculating Covariance matrix
subset = df[columns]
covarianceMat = subset.cov()
# correlationMat = subset.cov().as_matrix if needed as a numpy matrix

# Calculating Correlation matrix
correlationMat = subset.corr()
# correlationMat = subset.corr().as_matrix if needed as a numpy matrix

# TODO figure out seaborn to plot this stuff

cleaned = subset.dropna()  # getting rid of nasty NaN at the end (avg)
density_function = norm.pdf(cleaned[columns], means, sigmas)
# print(density_function)

logLikelihood = sum(np.log(density_function))  # TODO confirm calculation
print(logLikelihood)
# print("Means ", means)
# print("Variances ", variances)
# print("Standard Deviations ", sigmas)
# print("Covariance ", covarianceMat)
# print("Correlation ", correlationMat)
# print("Log-likelihood ", logLikelihood)
