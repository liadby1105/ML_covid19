#liad ben yechiel - 207637414

import numpy as np
import  statsmodels.api as  sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from numpy import arange
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

r_results=[]

def clean_high_correlation(X_train,X_test): #function to clean columnes with high correlation between them.
    df_temp_train = pd.DataFrame(X_train)
    df_temp_test = pd.DataFrame(X_test)
    corr_matrix = df_temp_train.corr().abs() # Create correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) # Select upper triangle of correlation matrix
    to_drop = [column for column in upper.columns if any(upper[column] > 0.975)] # Find features with correlation greater than 0.95
    df_temp_train.drop(to_drop, axis=1, inplace=True)# Drop features
    df_temp_test.drop(to_drop, axis=1, inplace=True)# Drop features
    X_train=df_temp_train.iloc[:,:].values
    X_test=df_temp_test.iloc[:,:].values
    return X_train,X_test

def measuring_metrics(y_test,y_pred): #Some mesuring metrics for the models
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print("mse:" + '\t', mse)
    print("rmse:" + '\t', rmse)
    print("mae:" + '\t', mae)
    print("r2:" + '\t', r2, '\n')
    metricss=[mse,rmse,mae,r2]
    return(metricss)


def backward_elimination(x,X_ols_test,y_dependent,SL): #The stage of backward elimination features with high pvalue.
    var=np.arange(x.shape[1])
    x_ols_array=x[:,var]
    regressor=sm.OLS(y_dependent,x_ols_array).fit()
    for i in range(sum(regressor.pvalues>0)):
        if sum(regressor.pvalues>SL)>0:
            arg=regressor.pvalues.argmax()
            var=np.delete(var,arg)
            x_ols_array=x[:,var]
            X_ols_test2=X_ols_test[:,var]
            regressor=sm.OLS(y_dependent,x_ols_array).fit()
    return (var[:],regressor,x_ols_array,X_ols_test2)
    

def reg_lin(X_train, X_test, y_train, y_test): #function that do the stage of the linear regresson.

    X_train=np.append(arr=np.ones((508,1)).astype(int),values=X_train ,axis=1)
    X_test=np.append(arr=np.ones((127,1)).astype(int),values=X_test ,axis=1)
 
    SL=0.05
    var,OLS,x_ols_array,X_ols_test = backward_elimination(X_train,X_test,y_train,SL)    
    y_pred=OLS.predict(X_ols_test)
    return(y_pred)

def predict(X,y): #function that calculate and run all the code (in function to do it twice, Once for the positve and the other time for the negative)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #split the data to train and test set 80-20.
    mm_x = MinMaxScaler() #make all the data run in the same scale.
    X_train = mm_x.fit_transform(X_train)
    X_test  = mm_x.transform(X_test)
    X_train,X_test=clean_high_correlation(X_train,X_test) #clean the data
    
    
    
    kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True) #the kfold for the cv

    norma = [True,False]
    models = [] #entering all the models in a list.
    models.append(('PR', LinearRegression(normalize=norma)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('LASSO', LassoCV(alphas=arange(0, 1, 0.01), cv=kfold)))
    models.append(('RR', Ridge()))
    models.append(('RF', RandomForestRegressor(random_state=1)))
    
    global r_results
    
    
    lr_reg = reg_lin(X_train, X_test, y_train, y_test)
    print('Linear Regresson metrics results:')
    measuring_metrics(y_test,lr_reg)
    r_results.append(('LR', r2_score(y_test,lr_reg)))
    
    for name, model in models: #run all the models from the list reading by their names.
        if (name=='PR'):
            degs = [1,2,3,4,5] #diffrent degrees that checked out.
            best_score = 0
            best_deg = 0
            for deg in degs:
                for nor in norma:
                    poly_features= PolynomialFeatures(degree = deg)
                    X_train_poly = poly_features.fit_transform(X_train)
                    model.fit(X_train_poly, y_train)
                    scores = cross_val_score(model, X_train_poly, y_train,cv=kfold, scoring='r2')
                    if max(scores) > best_score:
                        best_score = max(scores)
                        best_deg = deg
            print('Polyminal Regresson metrics results:')
            print("according to r2 in Polyminal Regresson the best degree is: ", best_deg)
            print("according to r2 in Polyminal Regresson the r2 best score is: ", best_score,'\n')
            r_results.append((name, best_score))
        if (name=='RF'):
            param_grid = {'n_estimators': range(50,130,10),
                          'max_features': ['auto'],
                          'max_depth' : np.arange(4,15),
                          'criterion' :['mse']}
            search = GridSearchCV(estimator=model,param_grid=param_grid, scoring='r2', cv=kfold)
            results_rf = search.fit(X_train, y_train)   
            y_pred = results_rf.predict(X_test)
            print('in RandomForest regresson {} is the Best depth according to {} '.format(search.best_params_['max_depth'],
                                                                                           'r2'.replace("_"," ")))
            print('in RandomForest regresson {} is the Best number of trees according to {} '.format(search.best_params_['n_estimators'],
                                                                                                     'r2'.replace("_"," ")))
            print('RandomForest metrics results:')
            measuring_metrics(y_test,y_pred)
            r_results.append((name, r2_score(y_test,y_pred)))
        if (name=='LASSO'):
            model.fit(X_train, y_train)
            y_pred=model.predict(X_test)
            print('LASSO regresson metrics results:')
            print('according to r2 best alpha in LASSO Regresson: %f' % model.alpha_)
            measuring_metrics(y_test,y_pred)
            r_results.append((name, r2_score(y_test,y_pred)))
        if (name=='KNN'):
            p=[1,2] #p = 1, this is equivalent to using manhattan distance and p = 2 for euclidean distance.
            n_neighbors = list(range(1,30))
            hyperparameters = dict(n_neighbors=n_neighbors,p=p)
            clf = GridSearchCV(model, hyperparameters, cv=kfold, scoring = 'r2')
            best_model = clf.fit(X_train,y_train)
            print('KNN metrics results:')
            print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
            print('Best p:', best_model.best_estimator_.get_params()['p'])
            y_pred = clf.predict(X_test)
            r_results.append((name, r2_score(y_test,y_pred)))
            measuring_metrics(y_test,y_pred)
        if (name=='RR'):
            grid = dict()
            grid['alpha'] = arange(0,0.1,0.01)
            search = GridSearchCV(model, grid, scoring='r2', cv=kfold)
            results = search.fit(X_train, y_train)
            y_pred = results.predict(X_test)
            print('{} is the Best alpha in Ridge Regresson according to {} '.format(results.best_params_['alpha'],
                                                                                    'r2'.replace("_"," ")))
            measuring_metrics(y_test,y_pred)
            r_results.append((name, r2_score(y_test,y_pred)))
    return(r_results)

df = pd.read_csv('COVID-19_Daily_Testing_-_By_Test.csv') #read the data to df
X = df.iloc[:, 5:23].values #slice all the independent variable
y_pos = df.iloc[:, 2].values #slice the results of the positive tests.
print('The Positive Results:', '\n')
predict(X,y_pos)

y_neg = df.iloc[:, 3].values #slice the results of the positive tests.
print('\n','The Negative Results:', '\n')
predict(X,y_neg)

i=0
best_score_pos = 0
best_score_neg = 0
for result in r_results: #checking the best model in the positive and negative section accrodint to the r^2 score.
    if i<6:
        if result[1] > best_score_pos:
            best_score_pos = result[1]
            best_reg_name_pos = result[0]
    else:
        if result[1] > best_score_neg:
            best_score_neg = result[1]
            best_reg_name_neg = result[0]
    i+=1

#print final results.
print('Accroding to r2 in the positive section the best model is ',best_reg_name_pos,"\n",
      "and his r2 score is: ",best_score_pos, "\n")
print('Accroding to r2 in the negative section the best model is ',best_reg_name_neg,"\n",
      "and his r2 score is: ",best_score_neg)
