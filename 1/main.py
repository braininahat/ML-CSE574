# Authors: Pranav Sankhe, Varun Shijo
# UB CSE 574 Project 1
# Referred extensively to the official documentation of libs
# https://pandas.pydata.org/pandas-docs/
# https://docs.scipy.org/doc/numpy-1.13.0/reference/

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import seaborn as sns

print("UBitName = pranavgi")
print("personNumber = 50244956")

print("UBitName = varunshi")
print("personNumber = 50244968")


# Switching to seaborn's pretty colors
sns.set(color_codes=True)

# Reading excel file into pandas dataset
ip_file = pd.read_excel('DataSet/university data.xlsx')

size = ip_file.shape


# Using pandas mean method for dataframes
def mean(data, col):
    m = data[col].mean()
    return m


# Using pandas covariance method for dataframes after cleaning the NaNs
def variance(data, col):
    v = np.cov(data[col].dropna(), ddof=0)
    return v


# Using pandas mean method for dataframes
def std(data, col):
    st = data[col].std()
    return st


mu1 = mean(ip_file, "CS Score (USNews)")
mu2 = mean(ip_file, "Research Overhead %")
mu3 = mean(ip_file, "Admin Base Pay$")
mu4 = mean(ip_file, "Tuition(out-state)$")

mu = [mu1, mu2, mu3, mu4]
print("mu1 = %.3f" % mu1)
print("mu2 = %.3f" % mu2)
print("mu3 = %.3f" % mu3)
print("mu4 = %.3f" % mu4)

var1 = variance(ip_file, "CS Score (USNews)")
var2 = variance(ip_file, "Research Overhead %")
var3 = variance(ip_file, "Admin Base Pay$")
var4 = variance(ip_file, "Tuition(out-state)$")

print("var1 = %.3f" % var1)
print("var2 = %.3f" % var2)
print("var3 = %.3f" % var3)
print("var4 = %.3f" % var4)

sigma1 = std(ip_file, "CS Score (USNews)")
sigma2 = std(ip_file, "Research Overhead %")
sigma3 = std(ip_file, "Admin Base Pay$")
sigma4 = std(ip_file, "Tuition(out-state)$")

print("sigma1 = %.3f" % sigma1)
print("sigma2 = %.3f" % sigma2)
print("sigma3 = %.3f" % sigma3)
print("sigma4 = %.3f" % sigma4)

# Selecting the 4 variables we want and their values
df = ip_file.iloc[0:49, 2:6]

# Storing for use in the multivariate pdf calculation step
cov_mat = df.cov()

print("covarianceMat = ", df.cov().round(3).as_matrix().tolist())
print("correlationMat = ", df.corr().round(3).as_matrix().tolist())

print("Using multivariate equation")
X = 0
for i in range(0, 49):
    X += (multivariate_normal.logpdf(
        df.iloc[i, :], mu, cov_mat, allow_singular='True'))
    X = X.round(3)
print("logLikelihood = ", X)


print("Using independent variables")
pdf_row = []
for i in range(0, 49):
    row = df.iloc[i, :]
    pdf_col1 = norm.logpdf(row[0], mu1, sigma1)
    pdf_col2 = norm.logpdf(row[1], mu2, sigma2)
    pdf_col3 = norm.logpdf(row[2], mu3, sigma3)
    pdf_col4 = norm.logpdf(row[3], mu4, sigma4)
    pdf_row.append(pdf_col1 + pdf_col2 + pdf_col3 + pdf_col4)

pdf = sum(pdf_row)
print("logLikelihood = %.3f" % pdf)

# seaborn pairplots (easier to interpret than heatmaps)
sns.pairplot(cov_mat)
sns.pairplot(df.corr())
plt.show()
