#-*- coding: utf8 -*-
import numpy as np
from matplotlib import pyplot as plt
import xlrd
bk = xlrd.open_workbook('/Users/qiwen/Desktop/Graduation_Project/Tolerance_factor/Tolerance_actor.xlsx')
sh = bk.sheet_by_index(0)
element = []
Tol = []
Oct = []
area = []
lowlimit_oct = 0.41
lowlimit_tol = 0.875
cell = sh.cell_value(1,3)
for i in range(1,35):
    ele = sh.cell_value(i,0)
    element.append(ele)
    tol = sh.cell_value(i,2)#r(MA)=2.7埃，IF sh.cell_value(i,2),r(MA)=2.16埃
    Tol.append(tol)
    octa = sh.cell_value(i,4)
    Oct.append(octa)
    ar = 50*np.pi*(tol)*(octa)
    area.append(ar)
plt.figure(figsize=(8,5),dpi=80)
axes = plt.subplot(111)
colors = np.random.rand(35,3)
bsite = axes.scatter(Oct,Tol,s=area,c=colors,alpha=1,marker=(9,2,30))
plt.xlabel('Octahedral factor')
plt.ylabel('Tolerance factor')
lowoct = np.linspace(0.41,0.41,100)
y = np.linspace(0.75,1.4,100)
x = np.linspace(0.3,1.2,100)
lowtor = np.linspace(0.875,0.875,100)
plt.plot(lowoct,y,label='lower cotahedral factor limit')
plt.plot(x,lowtor,label='lower tolerance factor limit')
plt.show()
