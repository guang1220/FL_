# -*- coding: utf-8 -*-#
"""
This is the common function used in the program

"""
import os
import pickle
import math
import joblib
import numpy as np
import pandas as pd
import logging



def get_logger(parent_path, name):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(parent_path + "/log/" + name + ".log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_filelist(dir):
    filelist = []
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            filelist.extend(get_filelist(newDir))
    return filelist


def get_map(path):
    map_f = open(path, 'rb')
    feature_dict = pickle.load(map_f)
    map_f.close()
    return feature_dict


# demo
def get_demo(dat, feature_dict, value_all):
    temp = np.array(dat)
    # 获取下标索引对应的特征值
    for m in range(len(temp)):
        demo_index = "demo"
        if m == 0:
            demo_index = demo_index + str(1)
        else:
            demo_index = demo_index + str(m + 1) + str(temp[m])
            temp[m] = 1

        index_num = feature_dict[demo_index]
        value_all[index_num] = temp[m]
    return value_all


# vital 时间窗口
def get_vital(dat, t, feature_dict, value_all):
    for m in range(len(dat)):
        try:
            temp = np.asarray(dat[m]).astype(float)
            meas_t = temp[:, -1]
            threshold = np.min(meas_t) - 1
            meas_t[meas_t > t] = threshold  # filter data_feature, which is after occur_time
            if np.max(meas_t) == threshold:
                continue
            nearest_t = np.where(meas_t == meas_t.max())[0][-1]  # 身高，体重，体重指数 取t天之前的最接近的数据的下标
            if m == 0 or m == 1 or m == 2:
                if m == 0:  # HT
                    temp[nearest_t][0] = temp[nearest_t][0] if 0 < temp[nearest_t][0] <= 95 else 0
                if m == 1:  # WT
                    temp[nearest_t][0] = temp[nearest_t][0] if 0 < temp[nearest_t][0] <= 1400 else 0
                if m == 2:  # BMI
                    temp[nearest_t][0] = temp[nearest_t][0] if 0 < temp[nearest_t][0] <= 70 else 0
                vital_index = 'vital' + str(m + 1)
                index_num = feature_dict[vital_index]
                value_all[index_num] = temp[nearest_t][0]

            if m == 3 or m == 4 or m == 5:
                # vital4,vital5,vital6 是一个定性变量
                vital_index = 'vital' + str((m + 1) * 10) + str(int(temp[nearest_t][0]))
                index_num = feature_dict[vital_index]
                value_all[index_num] = 1

            if m == 6 or m == 7:
                bp_index = np.where(meas_t > threshold)[0]
                bp_list = []
                if m == 6:  # BP_SYSTOLIC
                    for i in bp_index:
                        if not 40 <= temp[i][0] <= 210: temp[i][0] = 0
                        bp_list.append(temp[i].tolist())
                if m == 7:  # BP_DIASTOLIC
                    for i in bp_index:
                        if not (temp[i][0] <= 120 or temp[i][0] >= 40): temp[i][0] = 0
                        bp_list.append(temp[i].tolist())
                vital_index = 'vital' + str(m + 1)
                index_num = feature_dict[vital_index]
                value_all[index_num] = bp_list

        except Exception as e:
            # logger.exception("vital handle error: {}".format(e))
            continue
    return value_all


# lab 时间窗口
def get_lab(dat, t, feature_dict, value_all):
    for m in range(len(dat)):
        try:
            labIndex = dat[m][0][0][0]
            index_num = feature_dict[labIndex]
            temp = dat[m][1]
            meas_t = np.array(temp)[:, -1].astype(float)
            threshold = np.min(meas_t) - 1
            meas_t[meas_t > t] = threshold  # filter data_feature, which is after occur_time
            if np.max(meas_t) == threshold:
                continue
            lab_index = np.where(meas_t > threshold)[0]
            lab_list = []
            for i in lab_index:
                lab_list.append([temp[i][0], temp[i][-1]])
            value_all[index_num] = lab_list
        except Exception as e:
            # logger.exception("lab handle error: {}".format(e))
            continue
    return value_all

# med
def get_med(dat, t, feature_dict, value_all):
    for m in range(len(dat)):
        try:
            medIndex = dat[m][0][0][0]
            index_num = feature_dict[medIndex]
            temp1 = np.asarray(dat[m][1]).astype(float)
            temp2 = temp1[:, -1]
            temp3 = [x for x in temp2 if x <= t]
            if len(temp3) == 0:
                continue
            value_all[index_num] = len(temp3)  # 服用药物的次数
        except Exception as e:
            # logger.exception("med handle error: {}".format(e))
            # logger.exception("med missing: {}".format(dat[m][0][0][0]))
            continue
    return value_all

# ccs
def get_ccs(dat, t, feature_dict, value_all):
    for m in range(len(dat)):
        try:
            ccsTimes = dat[m][1]
            ccsTimes = list(map(float, ccsTimes))
            ccsTime = np.min(ccsTimes)
            if ccsTime <= t:  # 记录为最早出现的时间
                ccsIndex = dat[m][0][0]
                index_num = feature_dict[ccsIndex]
                value_all[index_num] = 1
        except Exception as e:
            # logger.exception("ccs handle error: {}".format(e))
            # logger.exception("ccs missing: {}".format(dat[m][0][0]))
            continue
    return value_all

# px
def get_px(dat, t, feature_dict, value_all):
    for m in range(len(dat)):
        try:
            pxTimes = dat[m][1]
            pxTimes = list(map(float, pxTimes))
            pxTime = np.min(pxTimes)
            if pxTime <= t:  # 记录为最早出现的时间
                pxIndex = dat[m][0][0]
                index_num = feature_dict[pxIndex]
                value_all[index_num] = 1
        except Exception as e:
            # logger.exception("px handle error: {}".format(e))
            # logger.exception("px missing: {}".format(dat[m][0][0]))
            continue
    return value_all

def get_instance(list_num):
    demo_vital_num, lab_num, ccs_px_num, med_num, new_feature_num = list_num
    demo_vital = np.full([1, demo_vital_num], np.nan)
    lab = np.full([1, lab_num], np.nan)
    ccs_px = np.full([1, ccs_px_num], np.nan)
    med = np.full([1, med_num], np.nan)
    new_feature = np.zeros([1, new_feature_num], dtype=np.float32)
    return np.hstack((demo_vital, lab, ccs_px, med, new_feature))[0].tolist()

def get_DataFrame(data, list_num):
    demo_vital_num, lab_num, ccs_px_num, med_num, new_feature_num = list_num
    d_v_start, d_v_end = 0, demo_vital_num
    lab_start, lab_end = demo_vital_num, demo_vital_num + lab_num
    c_p_start, c_p_end = demo_vital_num + lab_num, demo_vital_num + lab_num + ccs_px_num
    med_start, med_end = demo_vital_num + lab_num + ccs_px_num, demo_vital_num + lab_num + ccs_px_num + med_num
    n_f_start, n_f_end = demo_vital_num + lab_num + ccs_px_num + med_num, demo_vital_num + lab_num + ccs_px_num + med_num + new_feature_num

    demo_vital_df = get_data(data, d_v_start, d_v_end)
    lab_df = get_data(data, lab_start, lab_end)
    ccs_px_df = get_data(data, c_p_start, c_p_end)
    med_df = get_data(data, med_start, med_end)
    new_feature_df = get_data(data, n_f_start, n_f_end)
    dataframe = pd.concat([demo_vital_df, lab_df, ccs_px_df, med_df, new_feature_df], axis=1)
    return dataframe

def get_data(data, start, end):
    pd_data = []
    for d in data:
        r_data = []
        for i in range(start, end):
            r_data.append(d[i])
        pd_data.append(r_data)
    return pd.DataFrame(pd_data)

def get_dynamic_max(data):
    m,n=data.shape[0],data.shape[1]
    for j in range(n):
        for i in range(m):
            _list=data.iloc[i,j]
            if type(_list).__name__=='list' and len(_list)>0:
                Max=0
                for v,d in _list:
                    v=float(v)
                    if v>Max:Max=v
                data.iloc[i,j]=Max
    return data

def get_dynamic_min(data):
    m,n=data.shape[0],data.shape[1]
    for j in range(n):
        for i in range(m):
            _list=data.iloc[i,j]
            if type(_list).__name__=='list' and len(_list)>0:
                Max=0
                for v,d in _list:
                    v=float(v)
                    if v<Max:Max=v
                data.iloc[i,j]=Max
    return data

def get_dynamic_average(data):
    m,n=data.shape[0],data.shape[1]
    for j in range(n):
        for i in range(m):
            _list=data.iloc[i,j]
            if type(_list).__name__=='list' and len(_list)>0:
                v_list=[]
                for v,d in _list:
                    v_list.append(float(v))
                data.iloc[i,j]=round(np.mean(v_list), 2)
    return data

def get_dynamic_closest(data):
    m,n=data.shape[0],data.shape[1]
    for j in range(n):
        for i in range(m):
            _list=data.iloc[i,j]
            if type(_list).__name__=='list' and len(_list)>0:
                data.iloc[i,j]=_list[-1][0]
    return data

