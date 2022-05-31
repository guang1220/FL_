import os
import sys
import warnings
import flwr as fl
import joblib
import yaml
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

FL_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from fl_code.common_utils import get_logger, col, result_log, independent_test,year_by_year_test


def get_model_parameters(model):
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 2
    n_features = 308
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data():
    data = joblib.load(path + 'client_' + str(client_id) + '_data.pkl')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    ss = StandardScaler()
    ss = ss.fit(x_train.loc[:, col])
    x_train.loc[:, col] = ss.transform(x_train.loc[:, col])
    x_test.loc[:, col] = ss.transform(x_test.loc[:, col])
    return (x_train, y_train), (x_test, y_test)


def local_test():
    logger.info('test_method: {} start'.format('local_test'))
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    result_log(logger, y_test, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))


class Client(fl.client.NumPyClient):
    def get_parameters(self):
        return get_model_parameters(model)

    def fit(self, parameters, config):
        set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        return get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(model, parameters)
        y_score = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
        return auc, len(X_test), {"auc": auc}


if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
    client_id = str(sys.argv[1])
    ca = str(sys.argv[2])
    config_path = os.path.dirname(os.path.dirname(parent_path)) + '/config.yaml'
    path = yaml.load(open(config_path), Loader=yaml.FullLoader)['use_data_dir_path']
    logger = get_logger('logistic', 'logic_client_' + client_id, ca)

    if ca == '-1': ca = '0'
    server_ip = open(parent_path + '/server_IP').read().split("\n")[int(ca)].strip()

    (X_train, y_train), (X_test, y_test) = load_data()
    logger.info('Logistic_client_' + client_id)
    logger.info('#start')

    model = LogisticRegression(
        penalty="l2",
        max_iter=5000,
        warm_start=True,
        class_weight='balanced',
        n_jobs=-1
    )
    set_initial_params(model)

    fl.client.start_numpy_client(server_ip, client=Client())
    local_test()
    independent_test(logger, model,client_id)

    logger.info('#end')

    # local fine_tuning
    logger.info('local fine_tuning #start')
    model.fit(X_train, y_train)
    local_test()
    independent_test(logger, model, client_id)
    year_by_year_test(client_id,logger,model,path)
    logger.info('local fine_tuning #end')