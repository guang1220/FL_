import sys
from sklearn.neural_network import MLPClassifier
import joblib
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
import os
from common_utils import get_logger
from sklearn.ensemble import RandomForestClassifier

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from common_utils import col,independent_test

def RandomForest():
    clf = RandomForestClassifier(random_state=0,class_weight='balanced',max_depth=7,n_estimators=20)
    clf = clf.fit(Xtrain, Ytrain)
    independent_test(logger, clf, 'all')

def Logistic():
    model = LogisticRegression(penalty="l2",class_weight='balanced',max_iter=5000, warm_start=True)
    x_train = Xtrain.copy()

    ss = StandardScaler()
    ss = ss.fit(x_train.loc[:, col])
    x_train.loc[:, col] = ss.transform(x_train.loc[:, col])
    model=model.fit(x_train, Ytrain)
    independent_test(logger, model, 'all')


def xgboost():
    weight = (Ytrain == 0).sum() / (Ytrain == 1).sum()
    model = XGBClassifier(learning_rate=0.1, n_estimators=20,scale_pos_weight=weight)
    model.fit(Xtrain, Ytrain)
    independent_test(logger, model,  'all')

def SVM():
    model = SVC(kernel="linear",class_weight = "balanced", cache_size=10000,probability=True)

    x,x_train,x_test=X.copy(),Xtrain.copy(),Xtest.copy()

    ss = StandardScaler()
    ss = ss.fit(x_train.loc[:, col])
    x.loc[:, col] = ss.transform(x.loc[:, col])
    x_train.loc[:, col] = ss.transform(x_train.loc[:, col])
    x_test.loc[:, col] = ss.transform(x_test.loc[:, col])

    model = model.fit(x_train, Ytrain)
    independent_test(logger, model,'all')

def MLP():
    model = MLPClassifier(hidden_layer_sizes=(200,200),max_iter=200,random_state=0)
    x,x_train,x_test=X.copy(),Xtrain.copy(),Xtest.copy()

    ss = StandardScaler()
    ss = ss.fit(x_train.loc[:, col])
    x.loc[:, col] = ss.transform(x.loc[:, col])
    x_train.loc[:, col] = ss.transform(x_train.loc[:, col])
    x_test.loc[:, col] = ss.transform(x_test.loc[:, col])

    ros = RandomOverSampler(random_state=0)
    x_train_resampled, Ytrain_resampled = ros.fit_resample(x_train, Ytrain)
    model = model.fit(x_train_resampled, Ytrain_resampled)
    independent_test(logger, model,  'all')

if __name__ == "__main__":
    dir_path=yaml.load(open('../config.yaml'),Loader=yaml.FullLoader)['use_data_dir_path']
    classifier = sys.argv[1]
    parent_path = os.path.dirname(os.path.abspath(__file__))
    logger = get_logger('','all_center','all')
    logger.info('all_center')
    logger.info('#start')
    count = 0
    for file in os.listdir(dir_path):  # file 表示的是文件名
        if 'client' in file:
            count = count + 1
    all_data=joblib.load(dir_path+'/client_0'+'_data.pkl')
    for client in range(1,count):
        data = joblib.load(dir_path+'/client_'+str(client)+'_data.pkl')
        all_data=pd.concat([all_data,data])
    # logger.info('#=========================all_data.shape {}====================='.format(all_data.shape))
    X = all_data.iloc[:, :-1]
    y = all_data.iloc[:, -1]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)
    if 'r' in classifier:
        logger.info('Classifier: {} start'.format('RandomForest'))
        RandomForest()
        logger.info('Classifier: {} end'.format('RandomForest'))
    if 'l' in classifier:
        logger.info('Classifier: {} start'.format('Logistic'))
        Logistic()
        logger.info('Classifier: {} end'.format('Logistic'))
    if 's' in classifier:
        logger.info('Classifier: {} start'.format('svm'))
        SVM()
        logger.info('Classifier: {} end'.format('svm'))
    if 'x' in classifier:
        logger.info('Classifier: {} start'.format('xgboost'))
        xgboost()
        logger.info('Classifier: {} end'.format('xgboost'))
    if 'm' in classifier:
        logger.info('Classifier: {} start'.format('mlp'))
        MLP()
        logger.info('Classifier: {} end'.format('mlp'))
    logger.info('#end')
print("OK")
