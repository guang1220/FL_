import os
import sys
import warnings
import yaml
from sklearn import mixture
import flwr as fl
import joblib
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from fl_code.common_utils import get_logger,col


class ClusterClient(fl.client.NumPyClient):
    def get_parameters(self):
        score, means, covariances, weights, precisions, precisions_cholesky = [], [], [], [], [], []
        for n_components in range(2, 7):
            if n_components >= len(X): continue
            cluster = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=3000).fit(X)
            labels_ = cluster.predict(X)
            if (len(np.unique(labels_)) == 1):
                score.append(1)
            else:
                score.append(silhouette_score(X, labels_))
            means.append(cluster.means_)
            covariances.append(cluster.covariances_)
            weights.append(cluster.weights_)
            precisions.append(cluster.precisions_)
            precisions_cholesky.append(cluster.precisions_cholesky_)
        score = np.array(score)
        index = np.where(score == score.max())[0][0]
        cluster_mean = means[index]
        cluster_covariance = covariances[index]
        best_weights = weights[index]
        best_precisions = precisions[index]
        best_precisions_cholesky = precisions_cholesky[index]
        return [cluster_mean, cluster_covariance, best_weights, best_precisions, best_precisions_cholesky]

    def fit(self, parameters, config):
        logger.info("==================fit {} round===============".format(int(config['node_num'])))
        if config['node_num'] > 1:
            n_components = len(parameters[0])  # (n_components,n_features,n_features)
            clf = mixture.GaussianMixture(n_components=n_components, covariance_type='full', warm_start=True,
                                          max_iter=3000)
            clf.means_ = parameters[0]
            clf.covariances_ = parameters[1]
            clf.weights_ = parameters[2]
            clf.precisions_ = parameters[3]
            clf.precisions_cholesky_ = parameters[4]
            if len(cluster_model) == 0:
                cluster_model.append(clf)
            else:
                cluster_model[0] = clf
            return [clf.covariances_, clf.means_], len(X), {}
        else:
            params = self.get_parameters()
            return [params[1], params[0]], len(X), {}

    def evaluate(self, parameters, config):
        n_components = len(parameters[0])  # (n_components,n_features,n_features)
        clf = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
        clf.means_ = parameters[0]
        clf.covariances_ = parameters[1]
        clf.weights_ = parameters[2]
        clf.precisions_ = parameters[3]
        clf.precisions_cholesky_ = parameters[4]
        if len(cluster_model) == 0:
            cluster_model.append(clf)
        else:
            cluster_model[0] = clf
        return 1.0, len(X), {"silhouette_score": 1.0}


if __name__ == "__main__":
    path = yaml.load(open('../../config.yaml'), Loader=yaml.FullLoader)['use_data_dir_path']
    server_ip = open('./server/server_IP').readline().strip()
    client_id = str(sys.argv[1])
    parent_path = os.path.dirname(os.path.abspath(__file__))
    # save_path = parent_path + '/2d_result/clusters/'
    data = joblib.load(path + '/client_' + str(client_id) + '_data.pkl')

    X = data[col]
    y = data.iloc[:, -1]
    logger = get_logger(client_id + '_cluster_client')
    cluster_model = []
    fl.client.start_numpy_client(server_ip, client=ClusterClient())
    model = cluster_model[0]
    label_ = model.predict(X)
    if (len(np.unique(label_)) == 1):
        score = 1
    else:
        score = silhouette_score(X, label_)
    logger.info("n_components={}".format(len(model.means_)))
    logger.info("silhouette_score={}".format(score))
    logger.info(pd.DataFrame(label_).value_counts())

    data["cluster_label"] = label_
    v_c = data["cluster_label"].value_counts()
    # for i in range(len(v_c.values)):
    #     cluster_i = data[data["cluster_label"] == v_c.index[i]]
    #     joblib.dump(cluster_i, save_path + 'client_' + client_id + "_cluster_" + str(i) + "_data.pkl")
