from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

header = ['M', ['cf_letor', 'sgd_letor', 'cf_synth', 'sgd_synth']]

out = [[float(i) for i in check_output(['python3', 'letor.py',
        '30', '.01', '.3', '1000', str(10 * (j+1))]).split()] for j in range(10)]

count = 0
y =[]
for i in out:
    count += 1
    for j in i:
        y.append(j) 

cf_letor = y[0::4]
sgd_letor = y[1::4]
cf_synth = y[2::4]
sgd_synth = y[3::4]

x=[]

for foo in range(len(cf_letor)):
    x.append((foo+1)*10)

plt.subplot(4,1,1)
plt.plot(x,cf_letor,'r.-')
plt.title('cf_letor')
plt.xlabel('minibatch factor')
plt.ylabel('Error')

plt.subplot(4,1,2)
plt.plot(x,sgd_letor,'r.-')
plt.title('sgd_letor')
plt.xlabel('minibatch factor')
plt.ylabel('Error')

plt.subplot(4,1,3)
plt.plot(x,cf_synth,'r.-')
plt.title('cf_synth')
plt.xlabel('minibatch factor')
plt.ylabel('Error')

plt.subplot(4,1,4)
plt.plot(x,sgd_synth,'r.-')
plt.title('sgd_synth')
plt.xlabel('minibatch factor')
plt.ylabel('Error')

plt.show()
