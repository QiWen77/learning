#coding:utf-8
import xlrd
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from pylab import *
import pandas as pd 
from sklearn import datasets
from sklearn import linear_model
from sklearn import svm 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


fname = "energy_difference-descending_+1+2-1.xlsx"
bk = xlrd.open_workbook(fname)
shxrange = range(bk.nsheets)
try:
    sh = bk.sheet_by_name("ebh")
except:
    print("no sheet in %s named ebh" % fname)
nrows = sh.nrows#get rows
ncols = sh.ncols#get columns
print("nrows %d,ncols %d"%(nrows,ncols))
dopants_list = [ ]
ebh_list = [ ]
pauling_electronegativity = [ ]
ionic_radii = [ ]
atomic_radii = [ ]
percent_volume_change = [ ]
for i in range(1,nrows):
    row_data = sh.row_values(i)
    dopant = row_data[0]
    ebh = row_data[2]
    pauling = row_data[3]
    ionic = row_data[5]
    atomic = row_data[7]
    volume = row_data[8]
    dopants_list.append(dopant)
    ebh_list.append(ebh)
    pauling_electronegativity.append(pauling)
    ionic_radii.append(ionic)
    atomic_radii.append(atomic)
    percent_volume_change.append(volume*100)
print(('dopants:',dopants_list),('ebh:',ebh_list),('pauling_electronegativity:',pauling_electronegativity),('ionic_radii:',ionic_radii),('atomic_radii:',atomic_radii),('percent volume change:',percent_volume_change))
A_222 = range(0,10)
B_222 = range(11,38)
X_222 = range(39,42)
A_233 = range(44,54)
B_233 = range(55,82)   
X_233 = range(83,86)
I = range(88,89) 
regr = linear_model.LinearRegression()
### A site ebh and Ionic radii ###
plt.figure(figsize=(8,6))
# s1 = 30000*pauling_electronegativity[0:10:1]
# s2 = 30000*pauling_electronegativity[44:54:1]
plt.scatter(ionic_radii[0:10:1],ebh_list[0:10:1],s=50,c='red',marker='o',alpha=0.5,label='A site 2*2*2')
for x,y,z in zip(ionic_radii[0:10:1],ebh_list[0:10:1],dopants_list[0:10:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x1 = np.array(ionic_radii[0:10]).reshape(-1,1)
y1 = np.array(ebh_list[0:10]).reshape(-1,1)
regr.fit(x1,y1)
plt.plot(x1,regr.predict(x1),color = 'red',linewidth = 3)

plt.scatter(ionic_radii[44:54:1],ebh_list[44:54:1],s=50,c='blue',marker='*',alpha=0.5,label='A site 2*3*3')
for x,y,z in zip(ionic_radii[44:54:1],ebh_list[44:54:1],dopants_list[44:54:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x2 = np.array(ionic_radii[44:54]).reshape(-1,1)
y2 = np.array(ebh_list[44:54]).reshape(-1,1)
regr.fit(x2,y2)
plt.plot(x2,regr.predict(x2),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Ionic radii')
plt.xlabel('Ionic radii [Angstrom]')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper right')
plt.show()

### A site ebh and Pauling electronegativity ###
plt.figure(figsize=(8,6))
# s1 = 30000*pauling_electronegativity[0:10:1]
# s2 = 30000*pauling_electronegativity[44:54:1]
plt.scatter(pauling_electronegativity[0:10:1],ebh_list[0:10:1],s=50,c='red',marker='o',alpha=0.5,label='A site 2*2*2')
for x,y,z in zip(pauling_electronegativity[0:10:1],ebh_list[0:10:1],dopants_list[0:10:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x3 = np.array(pauling_electronegativity[0:10]).reshape(-1,1)
y3 = np.array(ebh_list[0:10]).reshape(-1,1)
regr.fit(x3,y3)
plt.plot(x3,regr.predict(x3),color = 'red',linewidth = 3)

plt.scatter(pauling_electronegativity[44:54:1],ebh_list[44:54:1],s=50,c='blue',marker='*',alpha=0.5,label='A site 2*3*3')
for x,y,z in zip(pauling_electronegativity[44:54:1],ebh_list[44:54:1],dopants_list[44:54:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x4 = np.array(pauling_electronegativity[44:54]).reshape(-1,1)
y4 = np.array(ebh_list[44:54]).reshape(-1,1)
regr.fit(x4,y4)
plt.plot(x4,regr.predict(x4),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Pauling electronegativity')
plt.xlabel('Pauling electronegativity')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper right')
plt.show()

###A site ebh and Percent volume change###
plt.figure(figsize=(8,6))
plt.scatter(percent_volume_change[0:10:1],ebh_list[0:10:1],s=50,c='red',marker='o',alpha=0.5,label='A site 2*2*2')
for x,y,z in zip(percent_volume_change[0:10:1],ebh_list[0:10:1],dopants_list[0:10:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x5 = np.array(percent_volume_change[0:10]).reshape(-1,1)
y5 = np.array(ebh_list[0:10]).reshape(-1,1)
regr.fit(x5,y5)
plt.plot(x5,regr.predict(x5),color = 'red',linewidth = 3)

plt.scatter(percent_volume_change[44:54:1],ebh_list[44:54:1],s=50,c='blue',marker='*',alpha=0.5,label='A site 2*3*3')
for x,y,z in zip(percent_volume_change[44:54:1],ebh_list[44:54:1],dopants_list[0:10:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x6 = np.array(percent_volume_change[44:54]).reshape(-1,1)
y6 = np.array(ebh_list[44:54]).reshape(-1,1)
regr.fit(x6,y6)
plt.plot(x6,regr.predict(x6),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Percent volume change')
plt.xlabel('Percent volume change [%]')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper right')
plt.show()

### B site ebh and Ionic radii###
plt.figure(figsize=(8,6))
plt.scatter(ionic_radii[11:38:1],ebh_list[11:38:1],s=50,c='red',marker='o',alpha=0.5,label='B site 2*2*2')
for x,y,z in zip(ionic_radii[11:38:1],ebh_list[11:38:1],dopants_list[11:38:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x9 = np.array(ionic_radii[11:38]).reshape(-1,1)
y9 = np.array(ebh_list[11:38]).reshape(-1,1)
regr.fit(x9,y9)
plt.plot(x9,regr.predict(x9),color = 'red',linewidth = 3)

plt.scatter(ionic_radii[55:82:1],ebh_list[55:82:1],s=50,c='blue',marker='*',alpha=0.5,label='B site 2*3*3')
for x,y,z in zip(ionic_radii[55:82:1],ebh_list[55:82:1],dopants_list[55:82:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x10 = np.array(ionic_radii[55:82]).reshape(-1,1)
y10 = np.array(ebh_list[55:82]).reshape(-1,1)
regr.fit(x10,y10)
plt.plot(x10,regr.predict(x10),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Ionic radii')
plt.xlabel('Ionic radii [Angstrom]')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper right')
plt.show()

### B site ebh and Pauling electronegativity ###
plt.figure(figsize=(8,6))
# s1 = 30000*pauling_electronegativity[0:10:1]
# s2 = 30000*pauling_electronegativity[44:54:1]
plt.scatter(pauling_electronegativity[11:38:1],ebh_list[11:38:1],s=50,c='red',marker='o',alpha=0.5,label='B site 2*2*2')
for x,y,z in zip(pauling_electronegativity[11:38:1],ebh_list[11:38:1],dopants_list[11:38:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x7 = np.array(pauling_electronegativity[11:38]).reshape(-1,1)
y7 = np.array(ebh_list[11:38]).reshape(-1,1)
regr.fit(x7,y7)
plt.plot(x7,regr.predict(x7),color = 'red',linewidth = 3)

plt.scatter(pauling_electronegativity[55:82:1],ebh_list[55:82:1],s=70,c='blue',marker='*',alpha=0.5,label='B site 2*3*3')
for x,y,z in zip(pauling_electronegativity[55:82:1],ebh_list[55:82:1],dopants_list[55:82:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='right',va='bottom')
x8 = np.array(pauling_electronegativity[55:82]).reshape(-1,1)
y8 = np.array(ebh_list[55:82]).reshape(-1,1)
regr.fit(x8,y8)
plt.plot(x8,regr.predict(x8),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Pauling electronegativity')
plt.xlabel('Pauling electronegativity')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper left')
plt.show()


### B site ebh and Percent volume change ###
plt.figure(figsize=(8,6))
plt.scatter(percent_volume_change[11:38:1],ebh_list[11:38:1],s=50,c='red',marker='o',alpha=0.5,label='B site 2*2*2')
for x,y,z in zip(percent_volume_change[11:38:1],ebh_list[11:38:1],dopants_list[11:38:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x11 = np.array(percent_volume_change[11:38]).reshape(-1,1)
y11 = np.array(ebh_list[11:38]).reshape(-1,1)
regr.fit(x11,y11)
plt.plot(x11,regr.predict(x11),color = 'red',linewidth = 3)

plt.scatter(percent_volume_change[55:82:1],ebh_list[55:82:1],s=50,c='blue',marker='*',alpha=0.5,label='B site 2*3*3')
for x,y,z in zip(percent_volume_change[55:82:1],ebh_list[55:82:1],dopants_list[55:82:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x12 = np.array(percent_volume_change[55:82]).reshape(-1,1)
y12 = np.array(ebh_list[55:82]).reshape(-1,1)
regr.fit(x12,y12)
plt.plot(x12,regr.predict(x12),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Percent volume change')
plt.xlabel('Percent volume change [%]')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper right')
plt.show()

### X site ebh and Pauling electronegativity ###
plt.figure(figsize=(8,6))
# s1 = 30000*pauling_electronegativity[0:10:1]
# s2 = 30000*pauling_electronegativity[44:54:1]
plt.scatter(pauling_electronegativity[39:42:1],ebh_list[39:42:1],s=50,c='red',marker='o',alpha=0.5,label='X site 2*2*2')
for x,y,z in zip(pauling_electronegativity[39:42:1],ebh_list[39:42:1],dopants_list[39:42:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x13 = np.array(pauling_electronegativity[39:42]).reshape(-1,1)
y13 = np.array(ebh_list[39:42]).reshape(-1,1)
regr.fit(x13,y13)
plt.plot(x13,regr.predict(x13),color = 'red',linewidth = 3)

plt.scatter(pauling_electronegativity[83:86:1],ebh_list[83:86:1],s=70,c='blue',marker='<',alpha=0.5,label='X site 2*3*3')
for x,y,z in zip(pauling_electronegativity[83:86:1],ebh_list[83:86:1],dopants_list[83:86:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='right',va='bottom')
x14 = np.array(pauling_electronegativity[83:86]).reshape(-1,1)
y14 = np.array(ebh_list[83:86]).reshape(-1,1)
regr.fit(x14,y14)
plt.plot(x14,regr.predict(x14),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Pauling electronegativity')
plt.xlabel('Pauling electronegativity')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper left')
plt.show()

### X site ebh and Ionic radii ###
plt.figure(figsize=(8,6))
# s1 = 30000*pauling_electronegativity[0:10:1]
# s2 = 30000*pauling_electronegativity[44:54:1]
plt.scatter(ionic_radii[39:42:1],ebh_list[39:42:1],s=50,c='red',marker='o',alpha=0.5,label='X site 2*2*2')
for x,y,z in zip(ionic_radii[39:42:1],ebh_list[39:42:1],dopants_list[39:42:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
x15 = np.array(ionic_radii[39:42]).reshape(-1,1)
y15 = np.array(ebh_list[39:42]).reshape(-1,1)
regr.fit(x15,y15)
plt.plot(x15,regr.predict(x15),color = 'red',linewidth = 3)

plt.scatter(ionic_radii[83:86:1],ebh_list[83:86:1],s=70,c='blue',marker='<',alpha=0.5,label='X site 2*3*3')
for x,y,z in zip(ionic_radii[83:86:1],ebh_list[83:86:1],dopants_list[83:86:1]):
    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='right',va='bottom')
x16 = np.array(ionic_radii[83:86]).reshape(-1,1)
y16 = np.array(ebh_list[83:86]).reshape(-1,1)
regr.fit(x16,y16)
plt.plot(x16,regr.predict(x16),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Ionic radii')
plt.xlabel('Ionic radii [Angstrom]')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper left')
plt.show()

### X site ebh and Percent volume change ###
plt.figure(figsize=(8,6))
plt.scatter(percent_volume_change[39:42:1],ebh_list[39:42:1],s=50,c='red',marker='o',alpha=0.5,label='X site 2*2*2')
for x,y,z in zip(percent_volume_change[39:42:1],ebh_list[39:42:1],dopants_list[39:42:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x17 = np.array(percent_volume_change[39:42]).reshape(-1,1)
y17 = np.array(ebh_list[39:42]).reshape(-1,1)
regr.fit(x17,y17)
plt.plot(x17,regr.predict(x17),color = 'red',linewidth = 3)

plt.scatter(percent_volume_change[83:86:1],ebh_list[83:86:1],s=50,c='blue',marker='*',alpha=0.5,label='X site 2*3*3')
for x,y,z in zip(percent_volume_change[83:86:1],ebh_list[83:86:1],dopants_list[83:86:1]):
    plt.annotate('%s'%z,xy=(x,y),textcoords='offset points',ha='left',va='bottom')
x18 = np.array(percent_volume_change[83:86]).reshape(-1,1)
y18 = np.array(ebh_list[83:86]).reshape(-1,1)
regr.fit(x18,y18)
plt.plot(x18,regr.predict(x18),color = 'blue',linewidth = 3)
plt.title('Energy above hull and Percent volume change')
plt.xlabel('Percent volume change [%]')
plt.ylabel('Energy above hull [eV/atom]')
# plt.xlim(0,3)
# plt.ylim(0,0.15)
plt.legend(loc = 'upper right')
plt.show()



