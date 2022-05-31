import itertools
import os
import random
from sklearn import tree
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score, \
    roc_curve
from sklearn.svm import SVC
import pandas as pd
from xgboost.sklearn import XGBClassifier
import common_utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shap
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from common_utils import get_logger,col
from sklearn.neural_network import MLPClassifier as MLP

pd.set_option('display.max_columns', 1000000)  # 可以在大数据量下，没有省略号
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_colwidth', 1000000)
pd.set_option('display.width', 1000000)
np.set_printoptions(threshold=np.inf)


#
# log = get_logger("bmi_age")
# dir_path=parent_path + '/2d_result/processed_data'
# for client in range(1,8):
#     data = joblib.load(dir_path+'/client_'+str(client)+'_data.pkl')
#
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#     feature_name = data.columns
#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)
#     best_depth = 2
#     best_auc = 0
#     for i in range(2, 7):
#         clf = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=i, n_estimators=10)
#         clf = clf.fit(Xtrain, Ytrain)
#         y_score = clf.predict_proba(Xtest)[:,1]
#         auc = roc_auc_score(Ytest, y_score)
#         if auc > best_auc:
#             best_auc = auc
#             best_depth = i
#     clf = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=best_depth, n_estimators=10)
#     clf = clf.fit(Xtrain, Ytrain)
#     log.info("----------------client_{}---------------".format(client))
#     log.info("best_depth: {} , best_auc: {}".format(best_depth, best_auc))
#     feature_importances = zip(feature_name, clf.feature_importances_)
#     sort_feature_importances = sorted(feature_importances, key=lambda item: item[1], reverse=True)
#     log.info('top 20 feature_importances : ')
#     top20=np.array(sort_feature_importances)[:20,0]
#
#     # res=['px13550|PROCEDURE|81001', 'px13829|PROCEDURE|84300', 'px14105|PROCEDURE|87086', 'ccs101|DIAGNOSIS|101', 'ccs103|DIAGNOSIS|103', 'ccs104|DIAGNOSIS|104', 'px14099|PROCEDURE|87070', 'px13554|PROCEDURE|81025']
#     res=list(top20)
#     log.info('{}'.format(res))
#
#     fig = plt.figure(figsize=(8, 8))
#     plt.suptitle("Data dimensionality reduction and visualization", fontsize=14)
#     X = data.loc[:, res].values
#
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#     Y = tsne.fit_transform(X)  # 转换后的输出
#     ax = fig.add_subplot(1, 1, 1)
#     plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#     plt.title("all_data")
#     ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
#     ax.yaxis.set_major_formatter(NullFormatter())
#     plt.savefig(parent_path + '/2d_result/visualization/client_'+str(client)+'_data_visualization.png')
# plt.show()

parent_path = os.path.dirname(os.path.abspath(__file__))
# logger = get_logger('test')
selected_features=[]
file=open('/home/guoyouguang/PycharmProjects/FL_test/data_feature/selected_features.txt')
while True:
    line=file.readline().strip()
    if not line: break
    selected_features.append(line)

# print(len(selected_features))
# print(len(set(selected_features)))
d=pd.DataFrame(selected_features)
print(d.value_counts())



