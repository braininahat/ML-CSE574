import pandas as pd

data = pd.read_excel('university data.xlsx')
data = data.to_csv('data.csv', encoding='utf-8')

df = pd.read_csv('data.csv')
# df.columns.tolist()


# Computing the means for named columns
mu1 = df['CS Score (USNews)'].mean()
mu2 =  df['Research Overhead %'].mean()
mu3 = df['Admin Base Pay$'].mean()
mu4 = df['Tuition(out-state)$'].mean()

# Computing variance for named columns
var1 = df['CS Score (USNews)'].var()
var2 =  df['Research Overhead %'].var()
var3 = df['Admin Base Pay$'].var()
var4 = df['Tuition(out-state)$'].var()

# Computing standard deviation for named columns
sigma1 = df['CS Score (USNews)'].std()
sigma2 =  df['Research Overhead %'].std()
sigma3 = df['Admin Base Pay$'].std()
sigma4 = df['Tuition(out-state)$'].std()

