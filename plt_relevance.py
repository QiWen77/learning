#coding=utf-8

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
from sklearn.model_selection import ShuffleSplit
from sklearn.learning_curve import learning_curve
from IPython.display import Image
import pydotplus


base_df = pd.read_excel('/Users/qiwen/Desktop/MAPbI3_paper/energy_difference-descending_+1+2-1.xlsx',header=0,sheet_name=3)
# print("base_df-info:",base_df.head)
A_222 = base_df["index_number"] == 1
B_222 = base_df["index_number"] == 2
X_222 = base_df["index_number"] == 3
A_233 = base_df["index_number"] == 4
B_233 = base_df["index_number"] == 5
X_233 = base_df["index_number"] == 6
A_all = base_df["index_number"].isin([1,4])
B_all = base_df["index_number"].isin([2,5])
X_all = base_df["index_number"].isin([3,6])
All = base_df["index_number"].isin([1,2,3,4,5,6])
# print("A_all:",base_df[A_all])

# print(base_df.sort_values(by=['ebh','Pauling electronegativityr'],ascending=[True,False]))
# print(base_df[base_df.index==A_222])
# print(base_df['Pauling electronegativity'][A_222])


linreg = linear_model.LinearRegression()

colour = ['red','blue','green','red','blue','green']
mark = ['o','+','*','o','+','*']
site_list=[B_222,B_233]
site_list_ = ['B_222','B_233']
property_list=['ebh','Pauling electronegativity','percent_volume_change','crystal ionic radii','ionic radii','atom radii','c_2.16_tolerance factor','i_2.16_tolerance factor','a_2.16_tolerance factor','c_2.7_tolerance factor','i_2.7_tolerance factor','a_2.7_tolerance factor','Pettifor chemical scale','average bond distance difference Å','bader dopant-']

# For plotting the relevance between any two features
def plt_relevance(site_list,property_list):#,title=None,unit=None):
    # title = self.title or "relevance of {i+j}".format(i for i in self.label_list,j for j in self.property_list)
    for i in range(len(site_list)):
        for j in range(len(property_list)):
            for k in range(j+1,len(property_list)):
                # print('j and k:',j,k)
                x_i_j = base_df[site_list[i]][property_list[j]].convert_objects(convert_numeric=True,copy=True).values
            # print('j and k:',j,k)
                y_i_k = base_df[site_list[i]][property_list[k]].convert_objects(convert_numeric=True,copy=True).values
                z_i_j = base_df[site_list[i]]['dopant'].convert_objects(convert_numeric=True,copy=True).values
                # print(x_i_j,z_i_j)
                plt.figure(figsize=(10,8))
                plt.scatter(x_i_j,y_i_k,s=50,c=colour[i],marker=mark[i],alpha=0.5,label=' ')#str(site_list[i])+property_list[j]+'_'+property_list[k]
                for x,y,z in zip(x_i_j,y_i_k,z_i_j):
                    plt.annotate('%s'%z,xy =(x,y),textcoords='offset points',ha='left',va='bottom')
                plt.title(site_list_[i]+'_'+property_list[j]+' vs '+property_list[k])
                plt.xlabel(property_list[j])
                plt.ylabel(property_list[k])
                h_i_j = np.array(x_i_j).reshape(-1,1)
                print("h_i_j:",h_i_j)
                p_i_j = np.array(y_i_k).reshape(-1,1)
                print("p_i_j:",p_i_j)
                linreg.fit(h_i_j,p_i_j)
                print("interception:",linreg.intercept_)
                print("coefficient:",linreg.coef_)
                plt.plot(h_i_j,linreg.predict(h_i_j),color = colour[i],linewidth = 3)
                # print('x_{m}_{n}:'.format(m=i,n=j),x_i_j,'y_{h}_{p}:'.format(h=i,p=k),y_i_k)     
        if j < len(property_list)-1:
            j+=1
#     return plt.show()
# relevance = plt_relevance(site_list,property_list)

# For calculating the correlation matrix
a = base_df[B_all][property_list]
a = a.convert_objects(convert_numeric=True,copy=True)
# help(a.convert_objects(convert_numeric=True))
# print(a.info())
# b = a['ebh'].corr(a['Pauling electronegativity'])
# # print('b:',b)
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
features = ["atom radii","i_2.7_tolerance factor","Pauling electronegativity","average bond distance difference Å","bader dopant-","percent_volume_change","Pettifor chemical scale"]#,,"CFSE(Dq)","r+/r-",
X = base_df[All][features]
# X_dopant = base_df[X_all]['dopant']
X = X.convert_objects(convert_numeric=True,copy=True)
# X = preprocessing.minmax_scale(X)
X = X.apply(lambda x:(x-np.average(x))/np.std(x))
print("X:",X)
# X = preprocessing.normalize(X)
# print('2:',X.info())
Y = base_df[All]['ebh']
# print("Y1:",Y)
Y = Y.convert_objects(convert_numeric=True,copy=True)
# Y = Y.apply(lambda x:(x-np.average(x))/np.std(x))
# print("Y2:",Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.9,random_state=3)
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

pca = PCA(n_components=2)
pca.fit(X)
# print("attr of pca:",dir(pca))
# print("get_covariance:",pca.get_covariance())
print("n_components_:",pca.n_components_)
print("n_features_:",pca.n_features_)
print("n_samples_:",pca.n_samples_)
# print("X:",X.values)
# print("fit_transform:",pca.fit_transform(X))
# print("singular_values_:",pca.singular_values_)
# print("X*components_[0].transpose:",np.dot(X,pca.components_[0].transpose()))
# print("explained_variance_[0]*components_[0].transpose:",pca.explained_variance_[0]*pca.components_[0].transpose())
print("components_:",pca.components_)
print("explained_variance_:",pca.explained_variance_)
print("explained_variance_ratio_:",pca.explained_variance_ratio_)

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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure(figsize=(10,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    # print("train_sizes:",train_sizes)
    # print("train_scores:",train_scores)
    # print("test_scores:",test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    # print("train_scores_mean:",train_scores_mean)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    # print("test_scores_mean:",test_scores_mean)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def try_different_model(model):
    model.fit(X_train,Y_train)
    # print("dir of X_train:",dir(X_train))
    print("params:%s"%model.get_params())
    # print("intercep:%s"%model.intercept_)
    # print("coefficient:%s"%model.coef_)
    score = model.score(X_test,Y_test)
    # print(len(X_test),len(Y_test))
    result = model.predict(X_test)
    if hasattr(model,'feature_importances_'):
        print(str(model).split('(')[0]+'\'s'+' '+"feature importances:%s"%model.feature_importances_)
        model_features=str(model).split('(')[0]+'\'s'+' '+"feature importances"
        plt.figure(figsize=(10,8))
        index = list(range(len(model.feature_importances_)))
        # print("index:",model.feature_importances_[6])
        plt.bar(index,model.feature_importances_)
        plt.xticks(index,features,rotation=90)
        for a,b in zip(index,model.feature_importances_):
            plt.text(a,b,'%.6f'%b,ha='center',va='bottom',fontsize=12)
        plt.title(model_features)
    if str(model).split('(')[0] == 'DecisionTreeRegressor':
        dot_data = tree.export_graphviz(model,out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("decision.pdf")
        # dot_data = tree.export_graphviz(model,out_file=None,filled=True,rounded=True,special_characters=True)#feature_names=X_test.feature_names,class_names=Y_test.feature_names,
        # print("dot_data:",dot_data)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # Image(graph.create_png())
        
    print(str(model).split('(')[0]+'\'s'+' '+"score:%s"%score)
    plt.figure(figsize=(10,8))
    plt.plot(range(len(result)),result,'b',label="predict")
    plt.plot(range(len(result)),Y_test.values,'r',label="test")
    plt.legend(loc="upper right") 
    plt.xticks(range(len(result)),X_test_dopant)
    plt.xlabel("dopant")
    plt.title(str(model).split('(')[0]+'\'s'+' '+'score:%f'%score)
    plt.figure(figsize=(10,8))
    plt.scatter(Y_test,result,s=50, c='blue',marker='<',label='test')
    # print((X_train))#, len(Y_train))
    plt.scatter(Y_train, Y_train, s=80, c='red', marker='*',label='train')
    plt.legend(loc="upper left") 
    plt.plot(Y_train,Y_train,'black',label='y=x')
    plt.xlabel("test data")
    plt.ylabel("predicted data")
    plt.title(str(model).split('(')[0]+'\'s'+' '+'score:%f'%score)
    # cv1 = ShuffleSplit(n_splits=100,test_size=0.1,random_state=3)
    # train_sizes,train_scores,test_scores = learning_curve(model,X,Y,train_sizes = [0.1,0.25,0.5,0.75,1],cv=9,scoring ='mean_square_error',n_jobs=4)
    plot_learning_curve(model,title=str(model).split('(')[0],X=X,y=Y,ylim=None,cv=10,n_jobs=4)

    plt.show()
if __name__ == '__main__':
    for i in [tree.DecisionTreeRegressor(),linear_model.LinearRegression(),neighbors.KNeighborsRegressor(n_neighbors = 4),ensemble.RandomForestRegressor(n_estimators=10,oob_score=True),ensemble.AdaBoostRegressor(n_estimators=10),ensemble.GradientBoostingRegressor(n_estimators=10),BaggingRegressor(),ExtraTreeRegressor(),KernelRidge(alpha=0.05,kernel='rbf')]:#,,linear_model.ridge_regression(alpha=1.0,X=X_train,y=Y_train),ensemble.GradientBoostingRegressor(n_estimators=10,learning_rate=0.1,max_depth=7,random_state=0,loss='ls')，svm.SVR(kernel='rbf',C=1e1,gamma=0.1)
        try_different_model(i)
    





    
