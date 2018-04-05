import time 
import pandas as pd  
from pandas import DataFrame
import matplotlib.pyplot as plt  
from matplotlib import cm,colors 
import numpy as np  
from matplotlib import rc  
from sklearn import linear_model,tree,svm,neighbors,ensemble
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ridge_regression,RidgeCV
from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA
from sklearn.tree import ExtraTreeRegressor 
from sklearn import metrics,svm,preprocessing
from sklearn.cross_validation import train_test_split

base_df = pd.read_excel('energy_difference-descending_+1+2-1.xlsx',sheet_name=3)
# print("base_df-info:",base_df.info)
A_222 = base_df['index_number'] == 1
B_222 = base_df['index_number'] == 2
X_222 = base_df['index_number'] == 3
A_233 = base_df['index_number'] == 4
B_233 = base_df['index_number'] == 5
X_233 = base_df['index_number'] == 6
A_all = base_df['index_number'].isin([1,4])
B_all = base_df['index_number'].isin([2,5])
X_all = base_df['index_number'].isin([3,6])
All = base_df['index_number'].isin([1,2,3,4,5,6])
# print("A_all:",base_df[A_all])

# print(base_df.sort_values(by=['ebh','Pauling electronegativity'],ascending=[True,False]))
# print(base_df[base_df.index==A_222])
# print(base_df['Pauling electronegativity'][A_222])


linreg = linear_model.LinearRegression()

colour = ['red','blue','green','red','blue','green']
mark = ['o','+','*','o','+','*']
site_list=[X_222,X_233]
site_list_ = ['B_222','B_233']
property_list=['ebh','Pauling electronegativity','percent_volume_change','crystal ionic radii','ionic radii','atom radii','c_2.16_tolerance factor','i_2.16_tolerance factor','a_2.16_tolerance factor','c_2.7_tolerance factor','i_2.7_tolerance factor','a_2.7_tolerance factor','Pettifor chemical scale','average bond distance difference Å','bader dopant-Pb']

# For plotting the relevance between any two features
def plt_relevance(site_list,property_list):#,title=None,unit=None):
    # title = self.title or "relevance of {i+j}".format(i for i in self.label_list,j for j in self.property_list)
    for i in range(len(site_list)):
        for j in range(len(property_list)):
            for k in range(j+1,len(property_list)):
                # print('j and k:',j,k)
                x_i_j = base_df[site_list[i]][property_list[j]].values
            # print('j and k:',j,k)
                y_i_k = base_df[site_list[i]][property_list[k]].values
                z_i_j = base_df[site_list[i]]['dopant'].values
                # print(x_i_j,z_i_j)
                plt.figure(figsize=(10,8))
                plt.scatter(x_i_j,y_i_k,s=50,c=colour[i],marker=mark[i],alpha=0.5,label=' ')#str(site_list[i])+property_list[j]+'_'+property_list[k]
                for x,y,z in zip(x_i_j,y_i_k,z_i_j):
                    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
                plt.title(site_list_[i]+'_'+property_list[j]+' vs '+property_list[k])
                plt.xlabel(property_list[j])
                plt.ylabel(property_list[k])
                h_i_j = np.array(x_i_j).reshape(-1,1)
                p_i_j = np.array(y_i_k).reshape(-1,1)
                linreg.fit(h_i_j,p_i_j)
                print("interception:",linreg.intercept_)
                print("coefficient:",linreg.coef_)
                plt.plot(h_i_j,linreg.predict(h_i_j),color = colour[i],linewidth = 3)
                # print('x_{m}_{n}:'.format(m=i,n=j),x_i_j,'y_{h}_{p}:'.format(h=i,p=k),y_i_k)     
        if j < len(property_list)-1:
            j+=1
    # return plt.show()
# relevance = plt_relevance(site_list,property_list)

# For calculating the correlation matrix
a = base_df[B_all][property_list]
a = a.convert_objects(convert_numeric=True,copy=True)
# help(a.convert_objects(convert_numeric=True))
# print(a.info())
b = a['ebh'].corr(a['Pauling electronegativity'])
# print('b:',b)
c_s = a.corr(method='spearman')['ebh']
# c_s2 = np.corrcoef(x=a,y=a)
# help(np.corrcoef)
# print("c_s2:",c_s2[0,0])
c_p = a.corr(method='pearson')['ebh']
c_k = a.corr(method='kendall')['ebh']
# print(a)
# print('spearman:',c_s[1])

x = []
y_s = []
y_p = []
y_k = []
x_ticks = []
for i in range(len(c_s)):
    x.append(i)
    y_s.append(c_s[i])
    y_p.append(c_p[i])
    y_k.append(c_k[i])
    x_ticks.append(c_s.index[i])
    # print(i,j)
# print('spearman:',y_s)
# print('pearson:',y_p)
# print('kendall:',y_k)
plt.figure(figsize=(10,12))
plt.plot(x,y_s,'ro--',label='spearman coefficient')
plt.ylabel('ebh')
for xy in zip(x,y_s):
    plt.annotate("%s"%np.round(xy[-1],decimals=3),xy=xy,xytext=(0,0),textcoords='offset points')
plt.plot(x,y_p,'b<--',label='pearson coefficient')
for xy in zip(x,y_p):
    plt.annotate("%s"%np.round(xy[-1],decimals=3),xy=xy,xytext=(0,0),textcoords='offset points')
plt.plot(x,y_k,'g*--',label='kendall coefficient')
for xy in zip(x,y_k):
    plt.annotate("%s"%np.round(xy[-1],decimals=3),xy=xy,xytext=(0,0),textcoords='offset points')

plt.xticks(x,x_ticks,rotation=90)
# plt.margins(0.8)
plt.legend(loc=0)
plt.title('B site correlation coefficient of various attributers')
# plt.show()                                              
# print('spearman:',c_s.sort_values())
# print('pearson:',c_p.sort_values())
# print('kendall:',c_k.sort_values())

# For multiple linear regression
features = ["atom radii","i_2.7_tolerance factor","r+/r-","Pauling electronegativity","Pettifor chemical scale","average bond distance difference Å","bader dopant-Pb","percent_volume_change","CFSE(Dq)"]#,"CFSE(Dq)"
X = base_df[B_all][features]
# X_dopant = base_df[X_all]['dopant']
X = X.convert_objects(convert_numeric=True,copy=True)
# X = preprocessing.minmax_scale(X)
X = X.apply(lambda x:(x-np.average(x))/np.std(x))
# print("X:",X)
# X = preprocessing.normalize(X)
# print('2:',X.info())
Y = base_df[B_all]['ebh']
# print("Y1:",Y)
Y = Y.convert_objects(convert_numeric=True,copy=True)
# Y = Y.apply(lambda x:(x-np.average(x))/np.std(x))
# print("Y2:",Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.9,random_state=1)
start_time_linreg = time.time()
model = linreg.fit(X_train,Y_train)

print("linreg train time %fs"%(time.time()-start_time_linreg))
print("interception:",linreg.intercept_)
print("coefficient:",linreg.coef_)
coeff = linreg.coef_.tolist()
pai = zip(features,coeff)
pair = []
for i in pai:
    pair.append(i)
print("features and coefficient:",pair)
print("score of the model:",model.score(X_test,Y_test))
print("parameters of the model:",model.get_params())
Y_pred = linreg.predict(X_test)
X_test_dopant = base_df.iloc[X_test.index]['dopant'].values
# precision = metrics.precision_score(Y_test,Y_pred)
# recall = metrics.recall_score(Y_test,Y_pred)
# print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
# accuracy = metrics.accuracy_score(Y_test,Y_pred)
# print('accuracy: %.2f%%' % (100 * accuracy))
# print(type(X_test_dopant))
# print("X_test:",X_test.values,"\n","Y_test:",Y_test.values,"\n","X_train:",X_train.values,"\n","Y_pred:",Y_pred,"\n","Y_train:",Y_train.values,"\n")
# Calculating Root Mean Squared Error(RMSE)
sum_mean=0
for i in range(len(Y_pred)):
    sum_mean+=(Y_pred[i]-Y_test.values[i])**2
sum_erro=np.sqrt(sum_mean/len(Y_pred))
# calculate RMSE by hand
print("RMSE by hand:",sum_erro)
# Plotting the tested and predicted vaules

plt.figure(figsize=(10,8))
plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
plt.plot(range(len(Y_pred)),Y_test.values,'r',label="test")
plt.legend(loc="upper right") 
plt.xticks(range(len(Y_pred)),X_test_dopant)
plt.xlabel("dopant")
plt.ylabel('value of ebh')
plt.title("B site")
plt.figure(figsize=(10,8))
plt.scatter(Y_test,Y_pred,s=50,c='red',marker='<')
plt.plot(Y_test,Y_test,'b',label='y=x')
plt.xlabel("test data")
plt.ylabel("predicted data")
plt.title("comparison of test data and predicted data")
# plt.show()

def try_different_model(model):
    model.fit(X_train,Y_train)
    print("params:%s"%model.get_params())
    # print("intercep:%s"%model.intercept_)
    # print("coefficient:%s"%model.coef_)
    score = model.score(X_test,Y_test)
    # print(len(X_test),len(Y_test))
    result = model.predict(X_test)
    if hasattr(model,'feature_importances_'):
        print(str(model).split('(')[0]+'\'s'+' '+"feature importances:%s"%model.feature_importances_)
    plt.figure(figsize=(10,8))
    plt.plot(range(len(result)),result,'b',label="predict")
    plt.plot(range(len(result)),Y_test.values,'r',label="test")
    plt.legend(loc="upper right") 
    plt.xticks(range(len(result)),X_test_dopant)
    plt.xlabel("dopant")
    plt.title(str(model).split('(')[0]+'\'s'+' '+'score:%f'%score)
    plt.figure(figsize=(10,8))
    plt.scatter(Y_test,result,s=50, c='b',marker='<')
    # print((X_train))#, len(Y_train))
    # plt.scatter(X_train, Y_train, s=80, c='orange', marker='*')
    plt.plot(Y_test,Y_test,'r',label='y=x')
    plt.xlabel("test data")
    plt.ylabel("predicted data")
    plt.title(str(model).split('(')[0]+'\'s'+' '+'score:%f'%score)
    plt.show()
if __name__ == '__main__':
    for i in [tree.DecisionTreeRegressor(),linear_model.LinearRegression(),svm.SVR(kernel='rbf',C=1e1,gamma=0.1),neighbors.KNeighborsRegressor(),ensemble.RandomForestRegressor(n_estimators=10,oob_score=True),ensemble.AdaBoostRegressor(n_estimators=10),ensemble.GradientBoostingRegressor(n_estimators=10),BaggingRegressor(),ExtraTreeRegressor(),KernelRidge(alpha=0.05,kernel='rbf')]:#,PCA(n_components=5),linear_model.ridge_regression(alpha=1.0,X=X_train,y=Y_train),ensemble.GradientBoostingRegressor(n_estimators=10,learning_rate=0.1,max_depth=7,random_state=0,loss='ls')
        try_different_model(i)
    





    
