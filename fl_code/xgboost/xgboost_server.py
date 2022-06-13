import sys
import flwr as fl
from typing import Dict
from xgboostfedtree import FedTree
import numpy as np
import multiprocessing
import socket

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

    def grow_tree(self,thread_num=6):
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

def fit_round(node_num: int) -> Dict:
    """Send round number to client."""
    return {"node_num": node_num}

if __name__ == "__main__":
    port = str(sys.argv[1])
    client_num = int(sys.argv[2])

    server_ip = socket.gethostbyname(socket.getfqdn(socket.gethostname())) + ':' + port
    with open('./server_IP', 'a', encoding='utf-8') as f:
        f.write(server_ip + '\n')

    strategy = FedTree(
        on_fit_config_fn=fit_round,
        min_available_clients=client_num,
        min_fit_clients=client_num,
        min_eval_clients=2
    )
    fl.server.start_server("0.0.0.0:"+port,strategy=strategy,config={"num_rounds": 21})
