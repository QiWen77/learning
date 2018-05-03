import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import time

train_data = pd.read_excel('case2_training.xlsx')
# print("train data info:",train_data.head())
test_data = pd.read_excel('case2_testing.xlsx')
min_max_scalar = preprocessing.MinMaxScaler()
np_scaled = min_max_scalar.fit_transform(train_data)
train_data_norm = pd.DataFrame(np_scaled)
train_data_norm.columns = [
    'ID', 'Region', 'Date', 'Weekday', 'Apartment', 'Beds', 'Review',
    'Pic Quality', 'Price', 'Accept'
]

features = [
    'Region', 'Date', 'Weekday', 'Apartment', 'Beds', 'Review', 'Pic Quality',
    'Price'
]
label = 'Accept'

X, Y = train_data_norm[features], train_data_norm[label]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=3)


###  Try different model as a start
def try_different_model(model):
    start_time = time.time()
    model.fit(X_train, Y_train)
    predictions = model.predict_proba(X_test)
    Y_pred = model.predict(X_test)
    end_time = time.time()
    name = str(model).split('(')[0] + '\'s'
    train_and_predict_time = end_time - start_time
    print(name + ' ' +
          "train and predict time: {}s".format(train_and_predict_time))
    confusion_matrix_ = confusion_matrix(Y_test, Y_pred)
    # plt.figure()
    plt.matshow(confusion_matrix_)
    plt.title(name + ' ' + 'Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Real type')
    plt.xlabel('Predicted type')
    false_positive_rate, recall, thresholds = roc_curve(
        Y_test, predictions[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    plt.figure()
    plt.title(name + ' ' + 'Receiving Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall out')
    plt.show()


if __name__ == '__main__':
    for i in [LogisticRegression(),GaussianNB(),MultinomialNB(),KNeighborsClassifier(),SVC(probability=True),DecisionTreeClassifier(),BaggingClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier()]:#
        try_different_model(i)

###From the previous test,I can get the AUC value of the tested moddel and the time consumption.it come to the result that GradientBoostingClassifier is the best classifier for this case.

### optimize the parameters of GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV

def modelfit(alg,
             X_train,
             Y_train,
             X_test,
             Y_test,
             performCV=True,
             printFeatureImportance=True,
             cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(X_train, Y_train)
    #Predict training set:
    dtrain_predictions = alg.predict(X_test)
    dtrain_predprob = alg.predict_proba(X_test)[:, 1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(
            alg, X, Y, cv=cv_folds, scoring='roc_auc', n_jobs=4)

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Y_test.values,
                                                     dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(
        Y_test, dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score),
               np.max(cv_score)))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_,
                             predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()


predictors = features
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(
    gbm0,
    X_train,
    Y_train,
    X_test,
    Y_test,
    performCV=True,
    printFeatureImportance=True,
    cv_folds=5)

# ###searching for the best n_estimators
# gsearch1 = GridSearchCV(
#     estimator=GradientBoostingClassifier(
#         learning_rate=0.1,
#         min_samples_split=500,
#         min_samples_leaf=50,
#         max_depth=8,
#         max_features='sqrt',
#         subsample=0.8,
#         random_state=10),
#     param_grid={
#         'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#     },
#     scoring='roc_auc',
#     n_jobs=4,
#     iid=False,
#     cv=5)
# print("X:", X_train, '\n', "Y:", Y_train)
# gsearch1.fit(X_train.values, Y_train.values)
# print("grid_scores:", gsearch1.grid_scores_, "best_params:",
#       gsearch1.best_params_, "best_score:", gsearch1.best_score_)
# ## best_params: {'n_estimators': 100} best_score: 0.784707067122819

# ###searching for max_depth and min_samples_split
# param_test2 = {
#     'max_depth': [5, 7, 9, 11, 13, 15, 17],
#     'min_samples_split': [200, 400, 600, 800, 1000]
# }
# gsearch2 = GridSearchCV(
#     estimator=GradientBoostingClassifier(
#         learning_rate=0.1,
#         n_estimators=100,
#         max_features='sqrt',
#         subsample=0.8,
#         random_state=10),
#     param_grid=param_test2,
#     scoring='roc_auc',
#     n_jobs=4,
#     iid=False,
#     cv=5)
# gsearch2.fit(X_train.values, Y_train.values)
# print("grid_scores:", gsearch2.grid_scores_, "best_params:",
#       gsearch2.best_params_, "best_score:", gsearch2.best_score_)
# ## best_params: {'max_depth': 5, 'min_samples_split': 800} best_score: 0.7859788638730565

### searching for min_samples_leaf
# param_test3 = {'min_samples_leaf': [10, 20, 30, 40, 50, 60, 70]}
# gsearch3 = GridSearchCV(
#     estimator=GradientBoostingClassifier(
#         learning_rate=0.1,
#         n_estimators=100,
#         max_depth=5,
#         max_features='sqrt',
#         subsample=0.8,
#         min_samples_split=800,
#         random_state=10),
#     param_grid=param_test3,
#     scoring='roc_auc',
#     n_jobs=4,
#     iid=False,
#     cv=5)
# gsearch3.fit(X_train.values, Y_train.values)
# print("grid_scores:", gsearch3.grid_scores_, "best_params:",
#       gsearch3.best_params_, "best_score:", gsearch3.best_score_)
# ##best_params: {'min_samples_leaf': 20} best_score: 0.7857621625266176

# ###searching for maxfeature
# param_test4 = {'max_features': [1, 2, 3, 4, 5, 6, 7, 8]}
# gsearch4 = GridSearchCV(
#     estimator=GradientBoostingClassifier(
#         learning_rate=0.1,
#         n_estimators=100,
#         max_depth=5,
#         min_samples_split=800,
#         min_samples_leaf=20,
#         subsample=0.8,
#         random_state=10),
#     param_grid=param_test4,
#     scoring='roc_auc',
#     n_jobs=4,
#     iid=False,
#     cv=5)
# gsearch4.fit(X_train.values, Y_train.values)
# print("grid_scores:", gsearch4.grid_scores_, "best_params:",
#       gsearch4.best_params_, "best_score:", gsearch4.best_score_)
# ## best_params: {'max_features': 5} best_score: 0.7864639168831798

###serching for subsample
# param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
# gsearch5 = GridSearchCV(
#     estimator=GradientBoostingClassifier(
#         learning_rate=0.1,
#         n_estimators=100,
#         max_depth=5,
#         min_samples_split=800,
#         min_samples_leaf=20,
#         random_state=10,
#         max_features=5),
#     param_grid=param_test5,
#     scoring='roc_auc',
#     n_jobs=4,
#     iid=False,
#     cv=5)
# gsearch5.fit(X_train.values, Y_train.values)
# print("grid_scores:", gsearch5.grid_scores_, "best_params:",
#       gsearch5.best_params_, "best_score:", gsearch5.best_score_)
# ## best_params: {'subsample': 0.7} best_score: 0.7865946920754865

### Trying to reduce learning_rate and increase n_estimators correspondingly
# gbm_tuned_1 = GradientBoostingClassifier(
#     learning_rate=0.05,
#     n_estimators=200,
#     max_depth=5,
#     min_samples_split=800,
#     min_samples_leaf=20,
#     subsample=0.7,
#     random_state=10,
#     max_features=5)
# modelfit(gbm_tuned_1, X_train,Y_train,X_test,Y_test)
## Accuracy : 0.7394
## AUC Score (Train): 0.790584
## CV Score : Mean - 0.7879291 | Std - 0.005423245 | Min - 0.78325 | Max - 0.7966294

### Continue to reduce learning_rate to one fifth of gbm_tuned_1,and increase n_estimators to five times 
gbm_tuned_2 = GradientBoostingClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=5,
    min_samples_split=800,
    min_samples_leaf=20,
    subsample=0.7,
    random_state=10,
    max_features=5)
modelfit(gbm_tuned_2, X_train,Y_train,X_test,Y_test)
#Model Report
# Accuracy : 0.7388
# AUC Score (Train): 0.790813
# CV Score : Mean - 0.7884053 | Std - 0.005468836 | Min - 0.7829971 | Max - 0.7971555

### Continue to reduce learning_rate to one fifth of gbm_tuned_2,and increase n_estimators to five times 
# gbm_tuned_3 = GradientBoostingClassifier(
#     learning_rate=0.002,
#     n_estimators=4000,
#     max_depth=5,
#     min_samples_split=800,
#     min_samples_leaf=20,
#     subsample=0.7,
#     random_state=10,
#     max_features=5)
# modelfit(gbm_tuned_3, X_train,Y_train,X_test,Y_test)
# Model Report
# Accuracy : 0.7376
# AUC Score (Train): 0.790198
# CV Score : Mean - 0.7878539 | Std - 0.005352489 | Min - 0.7827583 | Max - 0.7962942

###So we shall take gbm_tuned_2 as the optimum model to predict the testing data###


###Predicting the output probability of the testing data
import csv
pred_acceptance=gbm_tuned_2.predict(test_data[features])
pred_probability = gbm_tuned_2.predict_proba(test_data[features])[:,1]
# with open('probability.csv','w') as f:
#     writer = csv.writer(f)
#     for line in pred_probability:
#         writer.writerows(line)
dataframe = pd.DataFrame({'ID':test_data['ID'],'Probability':pred_probability})
dataframe.to_csv("test_data_ID-Prob.csv",index=False)    

feat_imp = pd.Series(gbm_tuned_2.feature_importances_,predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Final testing data Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

