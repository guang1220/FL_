import os
import sys
import flwr as fl
import joblib
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from fl_code.common_utils import get_logger,col,result_log,independent_test

def get_model_parameters(model,y_train):
    support_vectors = model.support_vectors_.tolist()
    support_vectors_idx = model.support_
    for i in range(len(support_vectors)):
        support_vectors[i].append(y_train[support_vectors_idx[i]])
    params = support_vectors
    return params

def load_data():
    data = joblib.load(path+'/client_'+str(client_id)+'_data.pkl')
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
        return get_model_parameters(model,y_train.values)

    def fit(self, parameters, config):
        if config['node_num'] == 2:
            final_vector.append(parameters)
        elif config['node_num'] > 2:
            final_vector[0]=parameters

        return get_model_parameters(model,y_train.values), len(X_train), {}

    def evaluate(self, parameters, config):
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        return 1 - auc, len(X_test), {"auc": auc}


if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
    client_id = str(sys.argv[1])
    ca = str(sys.argv[2])
    config_path = os.path.dirname(os.path.dirname(parent_path)) + '/config.yaml'
    path = yaml.load(open(config_path), Loader=yaml.FullLoader)['use_data_dir_path']
    logger = get_logger('svm', 'svm_client_' + client_id, ca)

    if ca == '-1': ca = '0'
    server_ip = open(parent_path + '/server_IP').read().split("\n")[int(ca)].strip()

    logger.info('svm_client_'+client_id)
    logger.info('#start')
    (X_train, y_train), (X_test, y_test) = load_data()
    final_vector, intercept = [], []

    model = SVC(kernel="linear",cache_size=50000, class_weight = "balanced")
    model.fit(X_train, y_train)

    fl.client.start_numpy_client(server_ip, client=Client())

    final_vector = np.array(final_vector[0])
    x1 = final_vector[:, :-1]
    y1 = final_vector[:, -1]

    model = SVC(kernel="linear", cache_size=50000, class_weight = "balanced")
    model.fit(x1, y1)
    local_test()
    independent_test(logger, model, path, client_id)
    logger.info('#end')

