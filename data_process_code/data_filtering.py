# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from FunctionUtils import get_logger,get_filelist
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import mutual_info_classif as MIC
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def missing_value_processing(data, c):
    # 离散值填补为0
    if c == 'discrete_data':
        data = data.fillna(value=0)
        return data
    else:
        # 随机森林填补缺失值
        X_missing_reg = data.copy()
        column_list = X_missing_reg.isnull().sum(axis=0).index
        sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
        for i in sortindex:
            df = X_missing_reg
            fillc = df.loc[:, column_list[i]]
            if fillc.isnull().sum() == 0: continue
            df = pd.concat([df.loc[:, df.columns != column_list[i]], pd.DataFrame(label)], axis=1)
            df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]
            if len(Xtrain)!=0:
                rfc = RandomForestRegressor(n_estimators=10)
                rfc = rfc.fit(Xtrain, Ytrain)
                Ypredict = rfc.predict(Xtest)
            else:
                Ypredict = [0] * len(Xtest)
            X_missing_reg.loc[X_missing_reg.loc[:, column_list[i]].isnull(), column_list[i]] = Ypredict
        return X_missing_reg

def get_features():
    selected_features=[]
    file=open(parent_path+'/data_feature/selected_features.txt')
    while True:
        line=file.readline().strip()
        if not line: break
        selected_features.append(line)
    return selected_features

if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features =get_features()
    logger = get_logger(parent_path, 'data_filtering')
    pd_dir_path = parent_path + "/processed_data/raw_pd_data/"
    pd_files = get_filelist(pd_dir_path)

    save_dir_path = parent_path + "/processed_data/filter_data/"
    os.makedirs(save_dir_path, exist_ok=True)

    num=0
    for pd_file in pd_files:
        logger.info("load data : {}".format(pd_file))
        raw_data = joblib.load(pd_file)
        label = raw_data.iloc[:, -1]
        label[label >= 1] = 1
        raw_data = raw_data.iloc[:, :-1]
        discrete_column, continuous_column, =  [], []
        for name in features:
            if "DIAGNOSIS" in name or "PROCEDURE" in name or 'vital4' in name or 'vital5' in name or 'vital6' in name or 'demo2' in name or 'demo3' in name or 'demo4' in name:
                discrete_column.append(name)
            else:
                continuous_column.append(name)

        discrete_data = raw_data[discrete_column]
        continuous_data = raw_data[continuous_column]

        discrete_data = missing_value_processing(discrete_data, 'discrete_data')
        continuous_data = missing_value_processing(continuous_data, 'continuous_data')

        data = pd.concat([discrete_data, continuous_data, label], axis=1)  # AKI_label
        data = data.apply(pd.to_numeric, errors='ignore', downcast="float")

        save_file_name = 'client_'+str(num)+'_data.pkl'
        num+=1
        joblib.dump(data, save_dir_path +save_file_name )
        logger.info("data size: {}".format(data.shape))
        logger.info("")

print("OK")
