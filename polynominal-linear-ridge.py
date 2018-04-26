import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline #在linux下处理文本的时候经常使用管道”|”，这里也可以用管道把前面的几个步骤串联起来，相当于管道的意思
from sklearn.linear_model import LinearRegression #普通的线性回顾
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import Ridge #岭回归 


''''生成数据'''
x=np.arange(0,1,0.002)
y=norm.rvs(0,size=500,scale=0.1)#生成均值为0 方差为0.1的500个随机数
y=y+x**2
''''' 均方误差根 '''  
def rmse(y_test,y):
    return sp.sqrt(sp.mean((y_test-y)**2))
'''计算R square'''
def R2(y_test,y_true):
    return 1-((y_test-y_true)**2).sum()/((y_true-y_true.mean())**2).sum()

'''计算rmse'''
def R22(y_test, y_true):  
    y_mean = np.array(y_true)  
    y_mean[:] = y_mean.mean()  
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true) 

plt.scatter(x, y, s=5)
degree=[1,2,5]
y_test=[]
y_test=np.array(y_test)
for  d in degree:
    clf=Pipeline([('poly',PolynomialFeatures(degree=d)),\
                  ('linear',LinearRegression(fit_intercept=False))])
#                 ('linear',Ridge(fit_intercept=False))])  岭回归
    clf.fit(x[:,np.newaxis],y)
    print(u'多项式为第{0}项时的线性回归的系数'.format(d))
    
    print(clf.named_steps['linear'].coef_)
    y_test = clf.predict(x[:, np.newaxis]) 
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f' %(rmse(y_test, y),R2(y_test, y),R22(y_test, y),clf.score(x[:, np.newaxis], y)))
    plt.plot(x, y_test, linewidth=2)
plt.grid()  
plt.legend(['1','2','1000'], loc='upper left')  
plt.show()
