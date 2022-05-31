import itertools
import os
import yaml
from sklearn import tree
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import pandas as pd
from matplotlib.ticker import NullFormatter
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import shap

col=['demo1|AGE', 'vital3|BMI', 'vital7|BP_SYSTOLIC', 'vital8|BP_DIASTOLIC', 'lab127|LAB_RESULT_CM|19212-0', 'lab156|LAB_RESULT_CM|2019-8', 'lab157|LAB_RESULT_CM|2020-6', 'lab158|LAB_RESULT_CM|2021-4', 'lab45|LAB_RESULT_CM|14003-8', 'lab196|LAB_RESULT_CM|2069-3', 'lab197|LAB_RESULT_CM|2070-1', 'lab198|LAB_RESULT_CM|2072-7', 'lab199|LAB_RESULT_CM|2075-0', 'lab200|LAB_RESULT_CM|2078-4', 'lab201|LAB_RESULT_CM|2079-2', 'lab82|LAB_RESULT_CM|15158-9', 'lab243|LAB_RESULT_CM|2339-0', 'lab244|LAB_RESULT_CM|2340-8', 'lab245|LAB_RESULT_CM|2342-4', 'lab246|LAB_RESULT_CM|2344-0', 'lab247|LAB_RESULT_CM|2345-7', 'lab248|LAB_RESULT_CM|2348-1', 'lab249|LAB_RESULT_CM|2350-7', 'lab250|LAB_RESULT_CM|2351-5', 'lab164|LAB_RESULT_CM|20436-2', 'lab613|LAB_RESULT_CM|4269-7', 'lab656|LAB_RESULT_CM|50555-2', 'lab292|LAB_RESULT_CM|26464-8', 'lab293|LAB_RESULT_CM|26465-5', 'lab294|LAB_RESULT_CM|26466-3', 'lab295|LAB_RESULT_CM|26469-7', 'lab296|LAB_RESULT_CM|26472-1', 'lab297|LAB_RESULT_CM|26473-9', 'lab431|LAB_RESULT_CM|30410-5', 'lab663|LAB_RESULT_CM|51487-7', 'lab672|LAB_RESULT_CM|53279-6', 'lab317|LAB_RESULT_CM|2703-7', 'lab318|LAB_RESULT_CM|2704-5', 'lab319|LAB_RESULT_CM|2705-2', 'lab126|LAB_RESULT_CM|19211-2', 'lab67|LAB_RESULT_CM|14864-3', 'lab289|LAB_RESULT_CM|26454-9', 'lab290|LAB_RESULT_CM|26455-6', 'lab291|LAB_RESULT_CM|26458-0', 'lab781|LAB_RESULT_CM|771-6', 'lab788|LAB_RESULT_CM|789-8', 'lab789|LAB_RESULT_CM|798-9', 'lab783|LAB_RESULT_CM|785-6', 'lab784|LAB_RESULT_CM|786-4', 'lab785|LAB_RESULT_CM|787-2', 'lab786|LAB_RESULT_CM|788-0', 'lab671|LAB_RESULT_CM|53278-8', 'lab353|LAB_RESULT_CM|2821-7', 'lab354|LAB_RESULT_CM|2823-3', 'lab355|LAB_RESULT_CM|2828-2', 'lab356|LAB_RESULT_CM|2829-0', 'lab738|LAB_RESULT_CM|6298-4', 'lab84|LAB_RESULT_CM|15202-5', 'lab123|LAB_RESULT_CM|19123-9', 'lab124|LAB_RESULT_CM|19124-7', 'lab257|LAB_RESULT_CM|24447-5', 'lab275|LAB_RESULT_CM|2518-9', 'lab276|LAB_RESULT_CM|2520-5', 'lab277|LAB_RESULT_CM|2524-7', 'lab108|LAB_RESULT_CM|17861-6', 'lab109|LAB_RESULT_CM|17862-4', 'lab488|LAB_RESULT_CM|32305-5', 'lab755|LAB_RESULT_CM|6874-2', 'lab81|LAB_RESULT_CM|15155-5', 'lab192|LAB_RESULT_CM|20657-3', 'lab128|LAB_RESULT_CM|19213-8', 'lab337|LAB_RESULT_CM|2744-1', 'lab338|LAB_RESULT_CM|2745-8', 'lab339|LAB_RESULT_CM|2746-6', 'lab340|LAB_RESULT_CM|2748-2', 'lab341|LAB_RESULT_CM|2755-7', 'lab342|LAB_RESULT_CM|2756-5', 'lab658|LAB_RESULT_CM|50560-2', 'lab68|LAB_RESULT_CM|14873-4', 'lab361|LAB_RESULT_CM|2861-3', 'lab362|LAB_RESULT_CM|2862-1', 'lab36|LAB_RESULT_CM|13980-8', 'lab387|LAB_RESULT_CM|30003-8', 'lab41|LAB_RESULT_CM|13992-3', 'lab71|LAB_RESULT_CM|14957-5', 'lab72|LAB_RESULT_CM|14959-1', 'lab732|LAB_RESULT_CM|61151-7', 'lab97|LAB_RESULT_CM|1747-5', 'lab343|LAB_RESULT_CM|2777-1', 'lab344|LAB_RESULT_CM|2778-9', 'lab345|LAB_RESULT_CM|2779-7', 'lab287|LAB_RESULT_CM|26450-7', 'lab288|LAB_RESULT_CM|26451-5', 'lab496|LAB_RESULT_CM|32593-6', 'lab769|LAB_RESULT_CM|711-2', 'lab770|LAB_RESULT_CM|713-8', 'lab10|LAB_RESULT_CM|11031-2', 'lab298|LAB_RESULT_CM|26474-7', 'lab299|LAB_RESULT_CM|26478-8', 'lab300|LAB_RESULT_CM|26479-6', 'lab774|LAB_RESULT_CM|731-0', 'lab775|LAB_RESULT_CM|736-9', 'lab376|LAB_RESULT_CM|2947-0', 'lab377|LAB_RESULT_CM|2950-4', 'lab378|LAB_RESULT_CM|2951-2', 'lab379|LAB_RESULT_CM|2955-3', 'lab380|LAB_RESULT_CM|2956-1', 'lab85|LAB_RESULT_CM|15207-4', 'lab5|LAB_RESULT_CM|10839-9', 'lab771|LAB_RESULT_CM|71693-6', 'lab782|LAB_RESULT_CM|777-3', 'lab457|LAB_RESULT_CM|3094-0', 'lab739|LAB_RESULT_CM|6299-2', 'lab392|LAB_RESULT_CM|30180-4', 'lab767|LAB_RESULT_CM|704-7', 'lab768|LAB_RESULT_CM|706-2', 'lab304|LAB_RESULT_CM|26511-6', 'lab779|LAB_RESULT_CM|751-8', 'lab780|LAB_RESULT_CM|770-8', 'lab494|LAB_RESULT_CM|3255-7', 'lab238|LAB_RESULT_CM|2276-4', 'lab125|LAB_RESULT_CM|1920-8', 'lab219|LAB_RESULT_CM|21563-2', 'lab106|LAB_RESULT_CM|17849-1', 'lab506|LAB_RESULT_CM|33037-3', 'lab606|LAB_RESULT_CM|41276-7', 'lab778|LAB_RESULT_CM|742-7', 'lab716|LAB_RESULT_CM|5905-5', 'lab524|LAB_RESULT_CM|33959-8', 'lab495|LAB_RESULT_CM|32552-2', 'lab493|LAB_RESULT_CM|32546-4', 'lab704|LAB_RESULT_CM|57734-6', 'med1036|PRESCRIBING|11160', 'med1918|PRESCRIBING|1359936', 'med1927|PRESCRIBING|1360172', 'med1928|PRESCRIBING|1360172', 'med2515|PRESCRIBING|1551652', 'med2516|PRESCRIBING|1551652', 'med2614|PRESCRIBING|1604539', 'med2615|PRESCRIBING|1604539', 'med2616|PRESCRIBING|1604540', 'med2689|PRESCRIBING|1652239', 'med2692|PRESCRIBING|1652639', 'med2693|PRESCRIBING|1652639', 'med2694|PRESCRIBING|1652644', 'med2695|PRESCRIBING|1652644', 'med2705|PRESCRIBING|1653202', 'med2706|PRESCRIBING|1653202', 'med2717|PRESCRIBING|1654862', 'med2718|PRESCRIBING|1654862', 'med2970|PRESCRIBING|1670021', 'med2971|PRESCRIBING|1670021', 'med3561|PRESCRIBING|1858995', 'med3570|PRESCRIBING|1860167', 'med3571|PRESCRIBING|1860167', 'med5198|PRESCRIBING|237527', 'med5199|PRESCRIBING|237528', 'med5530|PRESCRIBING|242120', 'med5531|PRESCRIBING|242120', 'med5817|PRESCRIBING|259111', 'med5818|PRESCRIBING|259111', 'med5955|PRESCRIBING|260265', 'med6692|PRESCRIBING|311028', 'med6693|PRESCRIBING|311028', 'med6694|PRESCRIBING|311034', 'med6695|PRESCRIBING|311034', 'med6696|PRESCRIBING|311040', 'med6697|PRESCRIBING|311040', 'med6698|PRESCRIBING|311041', 'med6699|PRESCRIBING|311041', 'med6700|PRESCRIBING|311048', 'med6701|PRESCRIBING|311048', 'med6702|PRESCRIBING|311054', 'med7720|PRESCRIBING|351297', 'med7721|PRESCRIBING|351297', 'med8170|PRESCRIBING|431196', 'med8171|PRESCRIBING|431248', 'med8172|PRESCRIBING|431248', 'med8173|PRESCRIBING|431252', 'med8174|PRESCRIBING|431264', 'med8329|PRESCRIBING|484322', 'med8330|PRESCRIBING|484322', 'med8342|PRESCRIBING|485210', 'med8343|PRESCRIBING|485210', 'med9838|PRESCRIBING|847187', 'med9841|PRESCRIBING|847203', 'med9842|PRESCRIBING|847203', 'med9849|PRESCRIBING|847259', 'med9850|PRESCRIBING|847259', 'med9851|PRESCRIBING|847417', 'med1170|PRESCRIBING|1151127', 'med1171|PRESCRIBING|1151127', 'med1562|PRESCRIBING|1251190', 'med1563|PRESCRIBING|1251190']
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
independent_data_path=yaml.load(open(parent_path+'/config.yaml'),Loader=yaml.FullLoader)['independent_data_path']

def get_logger(cf,name,ca):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    if ca=='all' or ca=='single':
        log_path = parent_path + "/log/"
    elif ca=='-1':
        log_path = parent_path + "/log/"+cf+"/all/"
    else:
        log_path=parent_path + "/log/"+cf+"/lack_"+ca+"/"
    os.makedirs(log_path, exist_ok=True)
    handler = logging.FileHandler(log_path + name + ".log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def result_log(logger,Ytest, y_pred,y_score):
    c_matrix = pd.DataFrame(confusion_matrix(Ytest, y_pred), index=["Real-0", "Real-1"],
                            columns=["Predict-0", "Predict-1"])
    logger.info("{}".format('confusion_matrix'))
    logger.info("{}".format(c_matrix))
    p = precision_score(Ytest, y_pred)
    r = recall_score(Ytest, y_pred)
    f1 = f1_score(Ytest, y_pred)
    auc = roc_auc_score(Ytest, y_score)
    logger.info('auc={}'.format(auc))
    logger.info('p={}'.format(p))
    logger.info('r={}'.format(r))
    logger.info('f1={} '.format(f1))

def year_by_year_test(curclient,logger,model,dir_path):
    logger.info('test_method: {} start'.format('year_by_year_test'))
    count = 0
    for file in os.listdir(dir_path):  # file 表示的是文件名
        if 'client' in file:
            count = count + 1
    for client in range(count):
        logger.info('center_id: {}'.format(client))
        client_data = joblib.load(dir_path+'/client_'+str(client)+'_data.pkl')
        X = client_data.iloc[:, :-1]
        y = client_data.iloc[:, -1]
        if isinstance(model,SVC) or isinstance(model,MLP) or isinstance(model,LogisticRegression):
            ss = StandardScaler()
            ss = ss.fit(X.loc[:, col])
            X.loc[:, col] = ss.transform(X.loc[:, col])
        if curclient==client:
            _, X, _, y = train_test_split(X, y, test_size=0.3, random_state=0)
        all_y_pred = model.predict(X)
        if isinstance(model, SVC):
            all_y_score = model.decision_function(X)
        else:
            all_y_score = model.predict_proba(X)[:, 1]
        result_log(logger,y,all_y_pred,all_y_score)
    logger.info('test_method: {} end'.format('year_by_year_test'))

def independent_test(logger,model,cur_client):
    logger.info('test_method: {} start'.format('independent_test'))
    independent_data = joblib.load(independent_data_path)
    X = independent_data.iloc[:, :-1]
    y = independent_data.iloc[:, -1]
    if isinstance(model,SVC) or isinstance(model,MLP) or isinstance(model,LogisticRegression):
        ss = StandardScaler()
        ss = ss.fit(X.loc[:, col])
        X.loc[:, col] = ss.transform(X.loc[:, col])
    all_y_pred = model.predict(X)
    if isinstance(model, SVC):
        y_score = model.decision_function(X)
    else:
        y_score=model.predict_proba(X)[:,1]
    result_log(logger,y,all_y_pred,y_score)
    logger.info('test_method: {} end'.format('independent_test'))
    # curve_roc(parent_path + '/2d_result/'+path+'/roc_img/',cur_client,y,y_score)

def majorityElement(nums):
    if len(nums) == 1:
        return nums[0]
    numDic = {}
    for i in nums:
        if i in numDic:
            numDic[i] += 1
        else:
            numDic[i] = 1
    res, maxcount = None, 0
    for key, value in numDic.items():
        if res == None: res = key
        if value > maxcount:
            res = key
            maxcount = value
    return res

def get_client_feature_importance(logger,path):
    logger.info('test_method: {} start'.format('year_by_year_test'))
    dir_path = parent_path + '/2d_result/' + path
    clients = []
    for file in os.listdir(dir_path):
        if 'client' in file:
            clients.append(file[:file.index('.')])
    for client in clients:
        logger.info('------------center_id: {}---------------'.format(client))
        client_data = joblib.load(dir_path + '/' + client+ 'pkl')
        X = client_data.iloc[:, :-1]
        y = client_data.iloc[:, -1]
        feature_name = client_data.columns
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)
        best_depth = 2
        best_auc = 0
        for i in range(2, 21):
            clf = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=i, n_estimators=100)
            clf = clf.fit(Xtrain, Ytrain)
            y_score = clf.predict_proba(Xtest)[:,1]
            auc = roc_auc_score(Ytest, y_score)
            if auc > best_auc:
                best_auc = auc
                best_depth = i
        clf = RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=best_depth, n_estimators=100)
        clf = clf.fit(Xtrain, Ytrain)
        logger.info("best_depth: {} , best_auc: {}".format(best_depth, best_auc))
        # print(clf.feature_importances_)
        # [*zip(feature_name, clf.feature_importances_)]
        feature_importances = zip(feature_name, clf.feature_importances_)
        sort_feature_importances = sorted(feature_importances, key=lambda item: item[1], reverse=True)
        # print(sort_feature_importances[:20])
        logger.info('top 20 feature_importances : ')
        logger.info('{}'.format(sort_feature_importances[:20]))
        logger.info('')

def curve_roc(path,cur_client,Ytest,y_score):
    lw = 2
    plt.figure(figsize=(5, 5))
    fpr, tpr, threshold = roc_curve(Ytest, y_score)
    auc=roc_auc_score(Ytest,y_score)
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (AUC = %0.2f)' % auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('t >= ')
    plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig(path+'/client_'+str(cur_client)+'_roc.png')

def TSNE_visualization(path):
    dir_path = parent_path + '/2d_result/' + path
    clients=[]
    for file in os.listdir(dir_path):
        if 'client' in file:
            clients.append(file[:file.index('.')])
    for client in clients:
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle("Data dimensionality reduction and visualization", fontsize=14)
        data = joblib.load(parent_path + '/2d_result/' + path + '/' + client+'.pkl')

        important_features = ['med11520|PRESCRIBING|998740', 'med10671|PRESCRIBING|876193', 'med496|PRESCRIBING|1049621', 'med1005|PRESCRIBING|1115005', 'med11155|PRESCRIBING|966571', 'med3498|PRESCRIBING|1807627', 'med1966|PRESCRIBING|1362057', 'med102|PRESCRIBING|1007073', 'med10949|PRESCRIBING|89905', 'px14786|PROCEDURE|99238', 'med10658|PRESCRIBING|866508', 'med7182|PRESCRIBING|313323', 'med7304|PRESCRIBING|313782', 'med2822|PRESCRIBING|1659137', 'med2922|PRESCRIBING|1665515', 'med5028|PRESCRIBING|204508', 'med1275|PRESCRIBING|1191222', 'med3272|PRESCRIBING|1740467', 'med4483|PRESCRIBING|198440', 'med6812|PRESCRIBING|311670']
        X = data[important_features].values

        # X = data.iloc[:, :-1].values
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = tsne.fit_transform(X)  # 转换后的输出
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.title(client)
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        # plt.axis('tight')
        plt.savefig(parent_path + '/2d_result/visualization/' + client + '_visualization.png')

def plot_confusion_matrix(cm,path,total, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    classes = ['Neg', 'Pos']
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("black")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        text = str(cm[i, j])+' ('+str(round((cm[i,j]/total)*100,1))+'%)'
        plt.text(j, i, text,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(parent_path+'/2d_result/processed_data/'+path+'/client_0_cm.png')
    # plt.show()

def heatmap():
    sns.set()
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示，必须放在sns.set之后
    np.random.seed(0)
    uniform_data = np.random.rand(10, 12)  # 设置二维矩阵 (n_client,k_features)
    uniform_data = uniform_data.T

    top_k_featrues = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    pd_data = pd.DataFrame(uniform_data, index=top_k_featrues)
    f, ax = plt.subplots(figsize=(9, 6))

    # heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
    # 参数annot=True表示在对应模块中注释值
    # 参数linewidths是控制网格间间隔
    # 参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
    # 参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
    sns.heatmap(pd_data, ax=ax, vmin=0, vmax=1, cmap='YlOrRd', annot=True, linewidths=2, cbar=True)

    ax.set_title('features heatmap')  # plt.title('热图'),均可设置图片标题
    ax.set_ylabel('Top K features')  # 设置纵轴标签
    ax.set_xlabel('client')  # 设置横轴标签

    # 设置坐标字体方向，通过rotation参数可以调节旋转角度
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, horizontalalignment='right')
    # plt.savefig('cluster.tif',dpi = 300)
    plt.show()

def shap_analyze():
    data = joblib.load(parent_path + '/2d_result/processed_data/independent_data.pkl')
    X = data.iloc[:, :20]
    y = data.iloc[:, -1]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)
    ss = StandardScaler()
    ss = ss.fit(Xtrain)
    Xtrain = ss.transform(Xtrain)
    Xtest = ss.transform(Xtest)

    # XGB
    # weight=(Ytrain==0).sum()/(Ytrain==1).sum()
    # model = XGBClassifier(learning_rate=0.1, n_estimators=10,scale_pos_weight=weight)
    # model.fit(Xtrain, Ytrain)
    # explainer = shap.Explainer(model)
    # shap_values = explainer(Xtest)

    # LinearExplainer logistic&svm
    # model = SVC(kernel="linear",class_weight='balanced', cache_size=50000)
    # model = LogisticRegression(penalty="l2",class_weight='balanced',max_iter=5000)
    # model = model.fit(Xtrain, Ytrain)
    # explainer = shap.LinearExplainer(model, Xtrain, feature_dependence="independent")
    # shap_values = explainer.shap_values(Xtest)

    # any model mlp,randomforest
    # model = RandomForestClassifier(random_state=0,class_weight='balanced',max_depth=7,n_estimators=10)
    model = MLP(hidden_layer_sizes=(200, 200), max_iter=200, random_state=0)
    model.fit(Xtrain, Ytrain)
    explainer = shap.KernelExplainer(model.predict_proba, Xtrain, link="logit")
    shap_values = explainer.shap_values(Xtest, nsamples=100)
    #
    shap.summary_plot(shap_values, Xtest, feature_names=X.columns)
    # plt.savefig('./shap.png')

    # shap.force_plot(explainer.expected_value[0], shap_values[0][0,:],X.columns,link="logit",matplotlib=True,show=False)
    # shap.force_plot(explainer.expected_value[0], shap_values[0], x_test, link="logit",matplotlib=True,show=False)
