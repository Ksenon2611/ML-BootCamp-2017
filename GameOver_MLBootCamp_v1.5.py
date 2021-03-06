
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble # RF, GBM
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score


X_train_init = pd.read_csv('/home/pavel/anaconda3/Scripts/MLBootCamp 2017/x_train.csv',sep=";")
Y_train_init = pd.read_csv('/home/pavel/anaconda3/Scripts/MLBootCamp 2017/y_train.csv',header=None)
Y_train_init.columns = ['IS_IT_GAMER'] # Rename our Target column

Y_test = pd.read_csv('/home/pavel/anaconda3/Scripts/MLBootCamp 2017/x_test.csv',sep=";")

data_train = pd.concat([X_train_init,Y_train_init],axis=1)

y_train_cl = data_train.iloc[:,12:13]
X_train_cl = data_train.iloc[:,:]
del X_train_cl['IS_IT_GAMER']

X_train_cl['Scores'] = X_train_cl['totalScore'] - X_train_cl['totalBonusScore']
# New feature - Intensivity of levels = number of attempted levels per day
X_train_cl['Levels_day'] = X_train_cl['numberOfAttemptedLevels']/X_train_cl['numberOfDaysActuallyPlayed']
# data_valide['Levels_day'] = data_valide['numberOfAttemptedLevels']/data_valide['numberOfDaysActuallyPlayed']
#New feature - Instensivity of attempts = number of attempts per day
X_train_cl['Attempts_day'] = X_train_cl['totalNumOfAttempts']/X_train_cl['numberOfDaysActuallyPlayed']

X_corr = X_train_cl.corr()

### Delete all correlate features with less importance
del X_train_cl['numberOfAttemptedLevels']
# 0.176618 - best result for now
del X_train_cl['totalStarsCount']
# 0.177936
del X_train_cl['totalScore']
# 0.177936
del X_train_cl['totalBonusScore']

# xtrain, xtest,ytrain, ytest = train_test_split(X_train_cl,y_train_cl,test_size = 0.3,random_state = 13) 


### HyperOpt Optimization ###

def objective(space):
    
    numfolds = 20
    total = 0
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=13)
    
    clf = xgb.XGBClassifier(n_estimators = 5000, 
                            max_depth = space['max_depth'],
                            learning_rate = space['learning_rate'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'])
    
    for train_index, test_index in kf.split(X_train_cl,y_train_cl.IS_IT_GAMER):
        xtrain, xtest = X_train_cl.iloc[train_index], X_train_cl.iloc[test_index]
        ytrain, ytest = y_train_cl.iloc[train_index], y_train_cl.iloc[test_index]
        
        eval_set = [(xtrain, ytrain),(xtest, ytest)]

        clf.fit(xtrain, ytrain, eval_metric="logloss",eval_set = eval_set, early_stopping_rounds=300)
        pred = clf.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
#        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)
    return{'loss':total, 'status': STATUS_OK }


space ={
        'max_depth': hp.choice('max_depth', np.arange(1, 5, dtype=int)),
        'learning_rate': hp.quniform('learning_rate', 0, 0.03, 0.001),
        'min_child_weight': hp.quniform ('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.8, 1)
    }


  
    
trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10, 
            trials=trials) 

print (best)   

trials.results
trials.trials

### RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
y_check = y_train_cl.values.ravel()


def objective_rf(space):
    
    numfolds = 10
    total = 0
    kf2 = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=13) 
    
    rf = RandomForestClassifier(n_estimators = space['n_estimators'], 
                            max_depth = space['max_depth'],
                            max_features = space['max_features'],
                            criterion = space['criterion'],
                            min_impurity_split = space['min_impurity_split'],
             #               scale = space['scale'],
             #               normalize = space['normalize'],
             #               min_samples_leaf = space['min_samples_leaf'],
             #               min_weight_fraction_leaf  = space['min_weight_fraction_leaf'],
             #               min_impurity_split = space['min_impurity_split'],
                            random_state = 13,
                            warm_start = True,                            
                            n_jobs = -1
                            )
    
    for train_index, test_index in kf2.split(X_train_cl,y_train_cl.IS_IT_GAMER):
        xtrain, xtest = X_train_cl.iloc[train_index], X_train_cl.iloc[test_index]
        ytrain, ytest = y_train_cl.iloc[train_index], y_train_cl.iloc[test_index]
        
     #   eval_set = [(xtrain, ytrain),(xtest, ytest)]

        rf.fit(xtrain, ytrain.values.ravel())
        pred = rf.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
#        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)
    return{'loss':total, 'status': STATUS_OK }


space ={
    'max_depth': hp.choice('max_depth', range(1,30)),
    'max_features': hp.choice('max_features', range(1,11)),
    'n_estimators': hp.choice('n_estimators', range(1,50)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'min_impurity_split': hp.uniform('min_impurity_split',0,0.1)                       
 #   'scale': hp.choice('scale', [0, 1]),
#    'normalize': hp.choice('normalize', [0, 1])
    }


  
    
trials = Trials()

best = fmin(fn=objective_rf,
            space=space,
            algo=tpe.suggest,
            max_evals=20, 
            trials=trials) 

print (best)   

trials.results
trials.trials

### Let's Try ExtraTreeClassifier ###

def objective_etree(space):
    numfolds = 10
    total = 0
    kf2 = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=13) 
    
    etree = ExtraTreesClassifier(n_estimators = space['n_estimators'], 
                            max_depth = space['max_depth'],
                            max_features = space['max_features'],
                            criterion = space['criterion'],
                            min_impurity_split = space['min_impurity_split'],
             #               scale = space['scale'],
             #               normalize = space['normalize'],
             #               min_samples_leaf = space['min_samples_leaf'],
             #               min_weight_fraction_leaf  = space['min_weight_fraction_leaf'],
             #               min_impurity_split = space['min_impurity_split'],
                            random_state = 13,
                            warm_start = True,                            
                            n_jobs = -1
                            )
    
    for train_index, test_index in kf2.split(X_train_cl,y_train_cl.IS_IT_GAMER):
        xtrain, xtest = X_train_cl.iloc[train_index], X_train_cl.iloc[test_index]
        ytrain, ytest = y_train_cl.iloc[train_index], y_train_cl.iloc[test_index]
        
     #   eval_set = [(xtrain, ytrain),(xtest, ytest)]

        etree.fit(xtrain, ytrain.values.ravel())
        pred = etree.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
#        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)
    return{'loss':total, 'status': STATUS_OK }

space ={
    'max_depth': hp.choice('max_depth', range(1,30)),
    'max_features': hp.choice('max_features', range(1,11)),
    'n_estimators': hp.choice('n_estimators', range(1,50)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'min_impurity_split': hp.uniform('min_impurity_split',0,0.1)                       
 #   'scale': hp.choice('scale', [0, 1]),
#    'normalize': hp.choice('normalize', [0, 1])
    }


  
    
trials = Trials()

best = fmin(fn=objective_rf,
            space=space,
            algo=tpe.suggest,
            max_evals=20, 
            trials=trials) 

print (best)   

trials.results
trials.trials






#1. n_estimators = 100;{'learning_rate': 0.05, 'x_subsample': 0.9160329614890378, 'min_child_weight': 6.0, 'max_depth': 3}
#2. n_estimators = 5000; {'min_child_weight': 6.0, 'learning_rate': 0.003, 'x_subsample': 0.8396633805360239, 'max_depth': 2}
#3. 0.3804 numfolds = 20; n_estimators=5000;{'min_child_weight': 5.0, 'learning_rate': 0.007, 'x_subsample': 0.8013942737365293, 'max_depth': 3}

# Hypothesis: NN, LightGBM, Extratrees, RandomForests

########################### Best solution for XGBoost ############################################
numfolds = 20
kf = StratifiedKFold(n_splits= numfolds,shuffle=True,random_state = 13)

rf_model = RandomForestClassifier(n_estimators = 1000,n_jobs=-1)
gbm_model = GradientBoostingClassifier(n_estimators = 5000,max_depth = 2,learning_rate=0.002)
etree_model = ExtraTreesClassifier()

xgb_model = xgb.XGBClassifier(n_estimators=5000,max_depth=3, learning_rate=0.007,
                              min_child_weight=5, subsample=0.8013942737365293)
total = 0   
   
for train_index,test_index in kf.split(X_train_cl,y_train_cl.IS_IT_GAMER):
 
    xtrain,xtest = X_train_cl.iloc[train_index], X_train_cl.iloc[test_index]
    ytrain, ytest = y_train_cl.iloc[train_index], y_train_cl.iloc[test_index]
    
    eval_set = [(xtrain, ytrain),(xtest, ytest)]
#    rf_model.fit(xtrain,ytrain)
    xgb_model.fit(xtrain, ytrain, eval_metric="logloss",eval_set = eval_set, early_stopping_rounds=300)

    pred = xgb_model.predict_proba(xtest)  
    pred = pred[:,1]
    logloss = log_loss(ytest,pred)
    print (logloss)
    
    total += logloss
    
result = total/numfolds 
 
print ("Logloss of model is: ", result) 
# 0.3822 - numfolds = 5
# 0.3804 - numfolds = 20

result

pred[:,1]
gbm_model.get_params

#############################################################################################

###   RandomForestClassifier ### DIDN'T GET BETTER LOG LOSS

numfolds = 10
kf = StratifiedKFold(n_splits= numfolds,shuffle=True,random_state = 13)

scores = cross_val_score(rf2_model,X_train_cl,y_train_cl.IS_IT_GAMER,scoring="neg_log_loss",cv = kf)

scores.mean(), scores.std()


 rf_model = RandomForestClassifier(n_estimators = 200,random_state = 13,max_depth = 2, n_jobs=-1)
 
 rf2_model = RandomForestClassifier(n_estimators = 40,min_impurity_split = 0.05484315966586634, max_features = 4,
                                    max_depth = 5, criterion = "gini",random_state = 13,warm_start = True, n_jobs = -1)

 rf3_model = RandomForestClassifier(n_estimators = 180, min_impurity_split = 0.07398323057139528, max_features = 8,
                                    max_depth = 28, criterion = "gini",random_state=13,n_jobs = -1)

total = 0   
   
for train_index,test_index in kf.split(X_train_cl,y_train_cl.IS_IT_GAMER):
 
    xtrain,xtest = X_train_cl.iloc[train_index], X_train_cl.iloc[test_index]
    ytrain, ytest = y_train_cl.iloc[train_index], y_train_cl.iloc[test_index]
    
#    eval_set = [(xtrain, ytrain),(xtest, ytest)]
#    rf_model.fit(xtrain,ytrain)
    rf2_model.fit(xtrain, ytrain.values.ravel())

    pred = rf2_model.predict_proba(xtest)  
    pred = pred[:,1]
    logloss = log_loss(ytest,pred)
    print (logloss)
#    print ('{:f}'.format(logloss.std()))
    #  score = cross_val_score(rf2_model,xtest,ytest, cv = kf)
 #   print (score.mean, score.std)
    total += logloss
    
result = total/numfolds 
 
print ("Logloss of model is: ", result) 



### Transform our Test data to submit predictions

Y_test['Scores'] = Y_test['totalScore'] - Y_test['totalBonusScore']
Y_test['Levels_day'] = Y_test['numberOfAttemptedLevels']/Y_test['numberOfDaysActuallyPlayed']
Y_test['Attempts_day'] = Y_test['totalNumOfAttempts']/Y_test['numberOfDaysActuallyPlayed']

del Y_test['numberOfAttemptedLevels']
del Y_test['totalStarsCount']
del Y_test['totalScore']
del Y_test['totalBonusScore']


# Best Score was taken by XGBoost Model with KFold = 20

### Prediction of Test Data ###
Pred_prob = xgb_model.predict_proba(Y_test)

Pred_xgb = np.delete(Pred_prob,0,1)
Pred_prob.shape


np.savetxt('submission1.0.txt',Pred_xgb, '%.9f') 
