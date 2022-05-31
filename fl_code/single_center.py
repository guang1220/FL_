import sys
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import os
from common_utils import get_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from common_utils import col,result_log,independent_test,year_by_year_test


def RandomForest(cur_client):
    model = RandomForestClassifier(random_state=0,class_weight='balanced',max_depth=7,n_estimators=20)
    model = model.fit(Xtrain, Ytrain)
    y_pred = model.predict(Xtest)
    y_score = model.predict_proba(Xtest)[:,1]
    logger.info('test_method: {} start'.format('local_test'))
    result_log(logger, Ytest, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))
    independent_test(logger, model, cur_client)
    year_by_year_test(cur_client, logger, model, dir_path)

def Logistic(cur_client):
    model = LogisticRegression(penalty="l2",class_weight='balanced',max_iter=5000)
    x, x_train, x_test = X.copy(), Xtrain.copy(), Xtest.copy()

    ss = StandardScaler()
    ss = ss.fit(x_train.loc[:, col])
    x.loc[:, col] = ss.transform(x.loc[:, col])
    x_train.loc[:, col] = ss.transform(x_train.loc[:, col])
    x_test.loc[:, col] = ss.transform(x_test.loc[:, col])

    model = model.fit(x_train, Ytrain)
    y_pred = model.predict(x_test)
    y_score = model.decision_function(x_test)
    logger.info('test_method: {} start'.format('local_test'))
    result_log(logger, Ytest, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))
    independent_test(logger, model,  cur_client)
    year_by_year_test(cur_client, logger, model, dir_path)

def SVM(cur_client):
    model = SVC(kernel="linear", class_weight = "balanced", cache_size=50000)
    x,x_train,x_test=X.copy(),Xtrain.copy(),Xtest.copy()

    ss = StandardScaler()
    ss = ss.fit(x_train.loc[:, col])
    x.loc[:, col] = ss.transform(x.loc[:, col])
    x_train.loc[:, col] = ss.transform(x_train.loc[:, col])
    x_test.loc[:, col] = ss.transform(x_test.loc[:, col])

    model = model.fit(x_train, Ytrain)
    y_pred = model.predict(x_test)
    y_score = model.decision_function(x_test)
    logger.info('test_method: {} start'.format('local_test'))
    result_log(logger, Ytest, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))
    independent_test(logger, model, cur_client)
    year_by_year_test(cur_client, logger, model, dir_path)

def xgboost(cur_client):
    weight=(Ytrain==0).sum()/(Ytrain==1).sum()
    model = XGBClassifier(learning_rate=0.1, n_estimators=20,scale_pos_weight=weight)
    model.fit(Xtrain, Ytrain)
    y_pred = model.predict(Xtest)
    y_score = model.predict_proba(Xtest)[:,1]
    logger.info('test_method: {} start'.format('local_test'))
    result_log(logger, Ytest, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))
    independent_test(logger, model,  cur_client)
    year_by_year_test(cur_client, logger, model, dir_path)

def MLP(cur_client):
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
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:,1]
    logger.info('test_method: {} start'.format('local_test'))
    result_log(logger, Ytest, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))
    independent_test(logger, model, cur_client)
    year_by_year_test(cur_client, logger, model, dir_path)

if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
    dir_path=yaml.load(open(os.path.dirname(parent_path)+'/config.yaml'),Loader=yaml.FullLoader)['use_data_dir_path']
    start_id=int(sys.argv[1])
    end_id=int(sys.argv[2])
    classifier=sys.argv[3]
    clients = range(start_id,end_id+1)
    logger = get_logger('','single_center','single')
    logger.info('single_center')
    for client in clients:
        logger.info('#start')
        logger.info('client_id: {}'.format(client))
        data = joblib.load(dir_path+'client_'+str(client)+'_data.pkl')
        X=data.iloc[:,:-1]
        y=data.iloc[:,-1]
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3,random_state=0)
        if 'r' in classifier:
            logger.info('Classifier: {} start'.format('RandomForest'))
            RandomForest(client)
            logger.info('Classifier: {} end'.format('RandomForest'))
        if 'l' in classifier:
            logger.info('Classifier: {} start'.format('Logistic'))
            Logistic(client)
            logger.info('Classifier: {} end'.format('Logistic'))
        if 's' in classifier:
            logger.info('Classifier: {} start'.format('svm'))
            SVM(client)
            logger.info('Classifier: {} end'.format('svm'))
        if 'x' in classifier:
            logger.info('Classifier: {} start'.format('xgboost'))
            xgboost(client)
            logger.info('Classifier: {} end'.format('xgboost'))
        if 'm' in classifier:
            logger.info('Classifier: {} start'.format('mlp'))
            MLP(client)
            logger.info('Classifier: {} end'.format('mlp'))
        logger.info('#end')
print("OK")

