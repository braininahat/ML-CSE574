# Author: Varun Shijo
# UB CSE 574
# Person Number 50244968
# Referred extensively to the official documentation of libs
# https://pandas.pydata.org/pandas-docs/
# https://docs.scipy.org/doc/numpy-1.13.0/reference/

import pandas as pd
import numpy as np
import seaborn as sb
from scipy.stats import norm
from math import exp, pi, sqrt
from numpy.linalg import det, pinv
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt


sb.set(color_codes=True)

def calc_pdf(row, mean, cov):
    coeff = (1 / (pow((2 * pi), 2) * sqrt(det(cov))))
    power = (-0.5) * np.matmul(np.transpose((row - mean)), pinv(cov), (row - mean))
    pdf = coeff * np.exp(power)
    return pdf


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
print(variances)
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

# print(covarianceMat)
# covarianceMat = subset.cov().as_matrix if needed as a numpy matrix

# Calculating Correlation matrix
correlationMat = subset.corr()
# correlationMat = subset.corr().as_matrix if needed as a numpy matrix

# TODO figure out seaborn to plot this stuff

cleaned = subset.dropna()  # getting rid of nasty NaN at the end (avg)

rows = np.asarray(list(cleaned.itertuples()))
# print(rows)

# pdf = [rows[i][1:5] for i in range(len(rows))]
pdf = [calc_pdf(rows[i][1:5], np.asarray(means),
                np.asarray(covarianceMat)) for i in range(len(rows))]

# pdf2 = [mn.pdf(rows[i][1:5], np.asarray(means),
#                 np.asarray(covarianceMat)) for i in range(len(rows))]

logs = np.log(pdf)
logLikelihood = [sum(logs[i]) for i in range(len(logs))]
print(sum(logLikelihood))
plt.plot(logLikelihood)
plt.show()
# print(np.array(means))
# print("Means ", means)
# print("Variances ", variances)
# print("Standard Deviations ", sigmas)
# print("Covariance ", covarianceMat)
# print("Correlation ", correlationMat)
# print("Log-likelihood ", logLikelihood)
