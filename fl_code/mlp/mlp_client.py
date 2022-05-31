import os
import sys
import warnings
import flwr as fl
import joblib
import yaml
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from fl_code.common_utils import get_logger,col,result_log,independent_test


def get_model_parameters(model) :
    params = (model.coefs_, model.intercepts_)
    return params

def set_model_params(model, params) :
    model.coefs_ = list(params[0])
    model.intercepts_ = list(params[1])
    return model


def load_data():
    data = joblib.load(path+'client_'+str(client_id)+'_data.pkl')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    ss = StandardScaler()
    ss = ss.fit(X.loc[:, col])
    X.loc[:, col] = ss.transform(X.loc[:, col])
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    return (x_train, y_train), (x_test, y_test)

def local_test():
    logger.info('test_method: {} start' .format('local_test'))
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    result_log(logger, y_test, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))

class Client(fl.client.NumPyClient):
    def get_parameters(self):
        return []

    def fit(self, parameters, config):
        if config['node_num']>1:
            set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train_resampled, y_train_resampled)
        return get_model_parameters(model), len(x_train_resampled), {}

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
    logger = get_logger('mlp', 'mlp_client_' + client_id, ca)

    if ca == '-1': ca = '0'
    server_ip = open(parent_path + '/server_IP').read().split("\n")[int(ca)].strip()

    (X_train, y_train), (X_test, y_test) = load_data()
    ros = RandomOverSampler(random_state=0)
    x_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    logger.info('mlp_client_'+client_id)
    logger.info('#start')

    model = MLP(hidden_layer_sizes=(200,200),max_iter=20,random_state=0,warm_start=True)

    fl.client.start_numpy_client(server_ip, client=Client())
    local_test()
    independent_test(logger, model,client_id)
    logger.info('#end')


