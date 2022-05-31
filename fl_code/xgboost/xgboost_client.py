import os
import sys
import joblib
import pandas as pd
import numpy as np
import multiprocessing
import flwr as fl
import yaml
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FL_test=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FL_test)
from fl_code.common_utils import get_logger,col,result_log,independent_test


def local_test():
    logger.info('test_method: {} start'.format('local_test'))
    y_pred = client.predict(x_test)
    y_score=client.predict_proba(x_test)[:,1]
    result_log(logger, y_test, y_pred, y_score)
    logger.info('test_method: {} end'.format('local_test'))


class Node:
    split_list=[]
    gain_list=[]
    cover_list=[]
    def __init__(self, x, samples, grad, hess, feature_sel=0.8 , min_num_leaf=5, min_child_weight=1, depth=0, max_depth=10, reg=1, gamma=1):
        self.x = x
        self.grad = grad
        self.hess = hess
        self.samples = samples
        self.depth = depth
        self.min_num_leaf = min_num_leaf
        self.reg = reg
        self.gamma  = gamma
        self.min_child_weight = min_child_weight
        self.feature_sel = feature_sel
        self.max_depth = max_depth
        self.n_samples = len(samples)
        self.n_features = self.x.shape[1]
        # self.subsampled_features = np.random.choice(np.arange(self.n_features), int(np.round(self.feature_sel * self.n_features)))
        self.subsampled_features =np.arange(self.n_features)
        self.val = -np.sum(self.grad[self.samples])/(np.sum(self.hess[self.samples]) + self.reg)
        self.rhs = None
        self.lhs = None
        self.score = float('-inf')
    def compute_gamma(self, gradient, hessian):
        return -np.sum(gradient)/(np.sum(hessian) + self.reg)

    def muti_find_greedy_split(self,features,queue):
        split_feature,score,split_val=None,float('-inf'),None
        for feature in features:
            x = self.x[self.samples, feature]
            values = list(set(x))
            for v in values:
                lhs = x <= v
                rhs = x > v
                lhs_idxs = np.nonzero(x <= v)[0]
                rhs_idxs = np.nonzero(x > v)[0]
                if (rhs.sum() > self.min_num_leaf) and (lhs.sum() > self.min_num_leaf):
                    if (self.hess[lhs_idxs].sum() > self.min_child_weight) and (self.hess[rhs_idxs].sum() > self.min_child_weight):
                        curr_score = self.get_gain(lhs, rhs)
                        if curr_score > score:
                            split_feature = feature
                            score = curr_score
                            split_val = v
        queue.put([split_feature, score,split_val])
        return queue

    def grow_tree(self,thread_num=20):
        features_step = int(len(self.subsampled_features) / thread_num)
        start, end = 0, 0
        qs, ps = [], []
        for i in range(1, thread_num + 1):
            start = end
            end += features_step
            if i == thread_num: end = len(self.subsampled_features)
            q = multiprocessing.Queue()
            qs.append(q)
            p = multiprocessing.Process(target=self.muti_find_greedy_split, args=(self.subsampled_features[start:end], q))
            ps.append(p)
        for p in ps: p.start()
        for p in ps: p.join()
        ans = []
        for q in qs:
            while not q.empty():
                ans.append(q.get())
        for feature,curr_score,value in ans:
            if curr_score > self.score:
                self.split_feature = feature
                self.score = curr_score
                self.split_val = value

        if not self.is_leaf:
            x = self.x[self.samples , self.split_feature]
            lhs = np.nonzero(x <= self.split_val)[0]
            rhs = np.nonzero(x > self.split_val)[0]
            self.lhs = Node(self.x, self.samples[lhs], self.grad, self.hess, self.feature_sel, self.min_num_leaf, self.min_child_weight, self.depth+1, self.max_depth, self.reg, self.gamma)
            self.rhs = Node(self.x, self.samples[rhs], self.grad, self.hess, self.feature_sel, self.min_num_leaf, self.min_child_weight, self.depth+1, self.max_depth, self.reg, self.gamma)
            Node.split_list.append(self.split_feature) ## feature index each split
            Node.gain_list.append(self.get_gain(lhs, rhs)) ## gain each split
            Node.cover_list.append(len(self.samples)) ## samples in node before each split
            self.lhs.grow_tree()
            self.rhs.grow_tree()

    def find_greedy_split(self, feature):
        x = self.x[self.samples, feature]
        values=list(set(x))
        for v in values:
            lhs = x <= v
            rhs = x > v
            lhs_idxs = np.nonzero(x <=v)[0]
            rhs_idxs = np.nonzero(x > v)[0]
            if (rhs.sum() > self.min_num_leaf) and (lhs.sum() > self.min_num_leaf):
                # check purity score
                if (self.hess[lhs_idxs].sum() > self.min_child_weight) and (self.hess[rhs_idxs].sum() > self.min_child_weight):
                    curr_score = self.get_gain(lhs, rhs)
                    if curr_score > self.score:
                        self.split_feature = feature
                        self.score = curr_score
                        self.split_val =v

    def get_gain(self, lhs, rhs):
        gradient = self.grad[self.samples]
        hessian  = self.hess[self.samples]
        gradl = gradient[lhs].sum()
        hessl  = hessian[lhs].sum()
        gradr = gradient[rhs].sum()
        hessr  = hessian[rhs].sum()
        return 0.5 * (gradl**2/(hessl + self.reg) + gradr**2/(hessr + self.reg) - (gradl + gradr)**2/(hessr + hessl + self.reg)) - self.gamma

    @property
    def is_leaf(self):
        return self.score == float('-inf') or self.depth >= self.max_depth

    def predict(self, x):
        pred = np.zeros(x.shape[0])
        for sample in range(x.shape[0]):
            pred[sample] = self.predict_sample(x[sample,:])
        return pred

    def predict_sample(self, sample):
        if self.is_leaf:
            return self.val
        if sample[self.split_feature] <= self.split_val:
            next_node = self.lhs
        else:
            next_node = self.rhs
        return next_node.predict_sample(sample)


class XGBoostTree:
    def fit(self, x, grad, hess, feature_sel, min_num_leaf, min_child_weight, max_depth, reg, gamma):
        self.tree = Node(x, np.array(np.arange(x.shape[0])), grad, hess, feature_sel, min_num_leaf, min_child_weight, depth=0, max_depth=max_depth, reg=reg, gamma=gamma)
        self.tree.grow_tree()
        return self
    def predict(self, x):
        return self.tree.predict(x)


class XGBoostClassifierClient(fl.client.NumPyClient):
    def __init__(self,feature_sel=1,max_depth=6,min_child_weight=1,min_num_leaf=5,lr=0.1,reg=1,gamma=0):
        self.dec_trees = []
        self.feature_sel = feature_sel
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_num_leaf = min_num_leaf
        self.lr = lr
        self.reg = reg
        self.gamma = gamma
        self.x, self.y = x_train_resampled, y_train_resampled
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return preds - labels
    def hess(self, preds):
        preds = self.sigmoid(preds)
        return preds * (1 - preds)

    @staticmethod
    def log_odds(column):
        binary_yes = np.count_nonzero(column == 1)
        binary_no  = np.count_nonzero(column == 0)
        return(np.log(binary_yes/binary_no))

    def predict(self, x):
        if isinstance(x,pd.DataFrame): x=x.values
        pred = np.zeros(x.shape[0])
        for tree in self.dec_trees:
            pred += self.lr * tree.predict(x)
        predicted_probas = self.sigmoid(np.full((x.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return preds

    def predict_proba(self, x):
        if isinstance(x, pd.DataFrame): x = x.values
        pred = np.zeros(x.shape[0])
        for tree in self.dec_trees:
            pred += self.lr * tree.predict(x)
        predicted_probas = self.sigmoid(np.full((x.shape[0], 1), 1).flatten().astype('float64') + pred)
        probs=[]
        for p in predicted_probas:
            probs.append([0,p])
        return np.array(probs)

    def get_parameters(self):
        return []

    def fit(self,  parameters, config):
        boosting_rounds=config["node_num"]
        old_auc=0
        if boosting_rounds==1:
            self.base_pred = np.full((self.x.shape[0], 1), 1).flatten().astype('float64')
        else:
            boosting_tree=parameters[0][0]
            self.dec_trees.append(boosting_tree)
            self.base_pred += self.lr * boosting_tree.predict(self.x)
            old_pred=self.lr * boosting_tree.predict(x_test)
            old_predicted_probas = self.sigmoid(np.full((x_test.shape[0], 1), 1).flatten().astype('float64') + old_pred)
            old_auc = roc_auc_score(y_test, old_predicted_probas)
        Grad = self.grad(self.base_pred, self.y)
        Hess = self.hess(self.base_pred)
        new_boosting_tree = XGBoostTree().fit(self.x, grad=Grad, hess=Hess, feature_sel=self.feature_sel, min_num_leaf=self.min_num_leaf, min_child_weight=self.min_child_weight, max_depth=self.max_depth, reg=self.reg, gamma=self.gamma)
        y_pred=self.lr * new_boosting_tree.predict(x_test)
        predicted_probas = self.sigmoid(np.full((x_test.shape[0], 1), 1).flatten().astype('float64') + y_pred)
        auc=roc_auc_score(y_test, predicted_probas)
        return new_boosting_tree,abs(int((auc-old_auc)*1000)),{"auc":auc}

    def evaluate(self, parameters, config):
        auc = 1.0
        return auc,0, {"auc": auc}


if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
    client_id = str(sys.argv[1])
    ca = str(sys.argv[2])
    config_path = os.path.dirname(os.path.dirname(parent_path)) + '/config.yaml'
    path = yaml.load(open(config_path), Loader=yaml.FullLoader)['use_data_dir_path']
    logger = get_logger('xgboost', 'xgb_client_' + client_id, ca)

    if ca == '-1': ca = '0'
    server_ip = open(parent_path + '/server_IP').read().split("\n")[int(ca)].strip()

    client_data = joblib.load(path+'/client_'+str(client_id)+'_data.pkl')
    logger.info('xgboost_client_'+client_id)
    logger.info('#start')

    client_data.columns = [i for i in range(client_data.shape[1])]
    X = client_data.iloc[:, :-1].values  # the last column contains labels
    y = client_data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    ros = RandomOverSampler(random_state=0)
    x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

    client=XGBoostClassifierClient()

    fl.client.start_numpy_client(server_ip, client=client)

    local_test()
    independent_test(logger, client,client_id)
    logger.info('#end')