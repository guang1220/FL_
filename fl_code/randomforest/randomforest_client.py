import os
import sys
from math import sqrt
import yaml
from sklearn import tree
from sklearn.model_selection import train_test_split
import flwr as fl
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib
import random

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from fl_code.common_utils import get_logger,result_log,independent_test,majorityElement


def local_test():
    logger.info('test_method: {} start'.format('local_test'))
    y_pred = client.predict(x_test)
    y_score = client.predict_proba(x_test)[:, 1]
    result_log(logger, y_test, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))


def sample_data(data):
    train = data.sample(frac=1.0, replace=True)
    test = data.loc[data.index.difference(train.index)].copy()
    train.index = range(train.shape[0])
    test.index = range(test.shape[0])
    return train, test


class FeatureTree:
    def __init__(self, selected_features, DecisionTreeClassifier):
        self.selected_features = selected_features
        self.DecisionTreeClassifier = DecisionTreeClassifier


class RandomforestClient(fl.client.NumPyClient):
    def __init__(self):
        self.train = None
        self.test = None
        self.selected_features = None
        self.trees = []
        self.max_iter = 3
        self.max_depth = 10
        self.class_weight = {0: 1, 1: 1}
        self.max_class_weight = 15

    def init(self, max_iter=3, max_depth=7, max_class_weight=10):
        self.train, self.test = sample_data(final_train)
        # logN，N/3，sqrtN，N
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_class_weight = max_class_weight

    def get_parameters(self):
        self.selected_features = random.sample(features, int(sqrt(n_features) + 1))
        Xtrain = self.train.iloc[:, :-1]
        ytrain = self.train.iloc[:, -1]
        clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, class_weight='balanced').fit(
            Xtrain.iloc[:, self.selected_features], ytrain)
        flf = FeatureTree(self.selected_features, clf)
        return flf

    def fit(self, parameters, config):
        if config['node_num']>1:
            self.trees.append(parameters[0][0])
        best_clf, best_auc,best_features = None, 0,None
        for i in range(self.max_iter):
            self.train, self.test = sample_data(final_train)
            Xtrain = self.train.iloc[:, :-1]
            ytrain = self.train.iloc[:, -1]
            Xtest = self.test.iloc[:, :-1]
            ytest = self.test.iloc[:, -1]
            self.selected_features = random.sample(features, int(sqrt(n_features) + 1))
            # max_depth = random.randint(2, self.max_depth)
            max_depth = self.max_depth
            self.class_weight[1] = random.randint(1, self.max_class_weight)
            clf = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced").fit(Xtrain, ytrain)
            y_score = clf.predict_proba(Xtest)[:,1]
            auc = roc_auc_score(ytest, y_score)
            if auc >= best_auc:
                best_auc = auc
                best_clf = clf
                best_features=self.selected_features
        flf = FeatureTree(best_features, best_clf)
        return flf, len(self.train), {"auc": best_auc}

    def evaluate(self, parameters, config):
        return 1.0, len(self.train), {"auc": 1.0}

    def predict(self, Xtest):
        all_tree_pred = [t.DecisionTreeClassifier.predict(Xtest) for t in self.trees]
        if len(all_tree_pred) < 3: return all_tree_pred[0]
        y_pred = []
        for i in range(len(Xtest)):
            single_ypred = [pred[i] for pred in all_tree_pred]
            y_pred.append(majorityElement(single_ypred))
        return y_pred

    def predict_proba(self, Xtest):
        '''
一、对于决策树
比如多分类的例子中，某一个叶子节点，包含的各类样本分别是{‘A’:1, ‘B’:2, ‘C’:7}，那么预测的概率predict_proba就分别是{‘A’:0.1, ‘B’:0.2, ‘C’:0.7}，如果使用的是predict方法，那就会选择概率最大的即C类。

二、对于随机森林
随机森林就是多个决策树的集成决策，此时predict_proba就是对于当前测试样本，所有树给各类的投票 / 所有树的数量。
比如有100棵树，给当前测试样本共计投给A类10票，B类20票，C类70票。那么预测的概率predict_proba就分别是{‘A’:0.1, ‘B’:0.2, ‘C’:0.7}，如果使用的是predict方法，那就会选择概率最大的即C类。
        '''
        all_tree_pred = [t.DecisionTreeClassifier.predict(Xtest) for t in self.trees]
        prob = []
        for i in range(len(Xtest)):
            single_ypred = np.array([pred[i] for pred in all_tree_pred])
            positive_prob = (single_ypred == 1).sum() / len(single_ypred)
            negative_prob = (single_ypred == 0).sum() / len(single_ypred)
            prob.append([negative_prob, positive_prob])
        return np.array(prob)


if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
    client_id = str(sys.argv[1])
    ca = str(sys.argv[2])
    config_path = os.path.dirname(os.path.dirname(parent_path)) + '/config.yaml'
    path = yaml.load(open(config_path), Loader=yaml.FullLoader)['use_data_dir_path']
    logger = get_logger('randomForest', 'random_client_' + client_id, ca)

    if ca == '-1': ca = '0'
    server_ip = open(parent_path + '/server_IP').read().split("\n")[int(ca)].strip()

    logger.info('RandomForest_client_' + client_id)
    logger.info('#start')
    client_data = joblib.load(path + 'client_' + str(client_id) + '_data.pkl')

    client_data.columns = [i for i in range(client_data.shape[1])]
    X = client_data.iloc[:, :-1]
    y = client_data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    final_train = pd.concat([x_train, y_train], axis=1)

    random.seed(0)
    n_features = 308
    features = range(0, n_features)
    client = RandomforestClient()
    client.init(max_iter=1,max_depth=7)

    fl.client.start_numpy_client(server_ip, client=client)

    local_test()
    independent_test(logger, client,client_id)
    logger.info('#end')
