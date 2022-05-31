# -*- coding: utf-8 -*-
import sys

import numpy as np
from FunctionUtils import get_filelist
import os
import joblib
import FunctionUtils as fu
import pandas as pd

# label
def get_label(labels, advance_day):
    # 首次爆发 AKI 患者 AKI 状态及预测时间
    status = int(labels[0][0])
    day = int(labels[0][1])
    day = day - advance_day
    return [status, day]


if __name__ == "__main__":
    pre_day=2
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger=fu.get_logger(parent_path,'get_raw_data')
    logger.info("get_raw_data")

    list_dir_path = parent_path+"/processed_data/list_data/"
    list_files = get_filelist(list_dir_path)
    feature_dict_path = parent_path + "/data_feature/feature_dict_map.pkl"
    feature_num_path = parent_path + "/data_feature/feature_num.pkl"
    feature_name_path = parent_path+"/data_feature/feature_name.csv"

    save_dir_path = parent_path + "/processed_data/raw_pd_data/"
    os.makedirs(save_dir_path, exist_ok=True)

    feature_names = pd.read_csv(feature_name_path)["name"].values
    for list_file in list_files:
        data = joblib.load(list_file)
        logger.info("load data : {}".format(list_file))
        feature_dict = joblib.load(feature_dict_path)
        feature_num = joblib.load(feature_num_path)
        column_len = len(feature_dict)
        data_num = len(data)
        raw_data = [0]*data_num
        num=0
        for i in range(data_num):
            encounter_id, demo, vital, lab, ccs, px, med, label = data[i]
            AKI_status, pred_day = get_label(label, pre_day)

            if pred_day < 0:
                continue

            aki_label_index, id_index =  feature_dict["AKI_label"], feature_dict["encounter_id"]

            instance = fu.get_instance(feature_num)
            instance[ aki_label_index] = AKI_status
            instance[ id_index] = encounter_id

            instance = fu.get_demo(demo, feature_dict, instance)
            instance = fu.get_vital(vital, pred_day, feature_dict, instance)
            instance = fu.get_lab(lab, pred_day, feature_dict, instance)
            instance = fu.get_med(med, pred_day, feature_dict, instance)
            instance = fu.get_ccs(ccs, pred_day, feature_dict, instance)
            instance = fu.get_px(px, pred_day, feature_dict, instance)

            raw_data[num] = instance
            num+=1

        encounter_id, demo, vital, lab, ccs, px, med, label, instance, data, feature_dict, instances_list = [], [], [], [], [], [], [], [], [], [], [], []
        del encounter_id, demo, vital, lab, ccs, px, med, label, instance, data, feature_dict, instances_list
        raw_data1 = [raw_data[i] for i in range(num)]
        raw_data_df = fu.get_DataFrame(raw_data1, feature_num)
        raw_data_df.columns = feature_names

        dynamic_column, discrete_column, continuous_column, = [], [], []
        for name in raw_data_df.columns:
            if "LAB_RESULT_CM" in name or 'vital7' in name or 'vital8' in name:
                dynamic_column.append(name)
            elif "DIAGNOSIS" in name or "PROCEDURE" in name or 'vital4' in name or 'vital5' in name or 'vital6' in name or 'demo2' in name or 'demo3' in name or 'demo4' in name:
                discrete_column.append(name)
            elif "PRESCRIBING" in name or "demo1" in name or 'vital1' in name or 'vital2' in name or 'vital3' in name:
                continuous_column.append(name)

        discrete_data = raw_data_df[discrete_column]
        continuous_data = raw_data_df[continuous_column]
        dynamic_data = raw_data_df[dynamic_column]
        label=raw_data_df.iloc[:,-2]

        dynamic_data_closest=fu.get_dynamic_closest(dynamic_data)

        raw_data_df=pd.concat([discrete_data, continuous_data, dynamic_data_closest, label], axis=1)
        save_file_name = list_file[list_file.rindex('/') + 1:list_file.rindex('_')]+"_pd.pkl"
        joblib.dump(raw_data_df, save_dir_path + save_file_name)

        logger.info("data size: {}".format(raw_data_df.shape) )
        logger.info("labels|count")
        logger.info(raw_data_df.iloc[:, -1].value_counts())
        logger.info("")
        del raw_data, raw_data_df
    print("OK")


