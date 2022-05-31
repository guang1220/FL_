# -*- coding: utf-8 -*-
"""
Process the TXT format data_feature into structured data_feature (list object).
The program requires two parameters:
parameters 1: The folder path of TXT format data_feature (one TXT file for each year) of the current site (eg. /path-to-dir/KUMC-TXT-data_feature/ )
parameters 2: The site_name of the current site (eg. KUMC)

output: multiple "string2list.pkl" files prefixed by year, saved into ../resutl/site_name/data_feature/

Example:
    python P1_data_string_to_list_process.py  /path-to-dir/KUMC-TXT-data_feature/  KUMC
"""
import os
import sys
import joblib
from FunctionUtils import get_filelist
import yaml

# Structure the demo class feature data_feature
def demo_process(data):
    return data.split('_')


# Structure the vital class feature data_feature
def vitals_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(';') for i in range(len(first_split))]
    third_split = [[second_split[i][j].split(',') for j in range(len(second_split[i]))] for i in
                   range(len(second_split))]
    return third_split


# Structure the ccs and px class feature data_feature
def css_px_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(':') for i in range(len(first_split))]
    third_split = [[second_split[i][j].split(',') for j in range(len(second_split[i]))] for i in
                   range(len(second_split))]
    return third_split


# Structure the lab and med class feature data_feature
def lab_med_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(':') for i in range(len(first_split))]
    third_split = [[second_split[i][j].split(';') for j in range(len(second_split[i]))] for i in
                   range(len(second_split))]
    fourth_split = [
        [[third_split[i][j][k].split(',') for k in range(len(third_split[i][j]))] for j in range(len(third_split[i]))]
        for i in range(len(third_split))]
    return fourth_split


# Structure the label class feature data_feature
def label_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(',') for i in range(len(first_split))]
    return second_split


if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_file_path =yaml.load(open('../config.yaml'),Loader=yaml.FullLoader)['txt_data_dir_path']

    save_dir_path = parent_path + "/processed_data/list_data/"
    os.makedirs(save_dir_path, exist_ok=True)

    txt_file_list = get_filelist(txt_file_path)

    for txt_file in txt_file_list:
        X = []
        if not txt_file.endswith('.txt'):
            continue
        print("txt_file:", txt_file)

        with open(txt_file, 'r') as f:
            for line in f.readlines():
                encounter_id = line.strip().split('"')[1]
                demo, vital, lab, ccs, px, med, label = line.strip().split('"')[3].split('|')

                demo = demo_process(demo)
                vital = vitals_process(vital)
                lab = lab_med_process(lab)
                med = lab_med_process(med)
                ccs = css_px_process(ccs)
                px = css_px_process(px)
                label = label_process(label)

                X.append([encounter_id, demo, vital, lab, ccs, px, med, label])

        save_file_name=txt_file[txt_file.rindex('/')+1:-4]+ '_list.pkl'
        joblib.dump(X, save_dir_path +  save_file_name)

print("OK")
