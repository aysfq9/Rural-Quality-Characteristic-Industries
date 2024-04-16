#! -*- coding: utf-8 -*-
import re
import numpy as np
from tqdm import tqdm
from metrics import *
from train import *
import json



def load_eval_data(data_path, max_len):
    X = []
    y = []
    sentence = []
    labels = []
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split()
            if (not line or len(line) < 2):
                X.append(sentence)
                y.append(labels)
                sentence = []
                labels = []
                continue
            word, tag = line[0], line[1]
            if split_pattern.match(word) and len(sentence) + 8 >= max_len:
                sentence.append(word)
                labels.append(tag)
                X.append(sentence)
                y.append(labels)
                sentence = []
                labels = []
            else:
                sentence.append(word)
                labels.append(tag)
    if len(sentence):
        X.append(sentence)
        sentence = []
        y.append(labels)
        labels = []
    return X, y

def predict_label(data, y_true, output_folder):
    y_pred = []
    merged_text = ""
    for d in data:
        text = ''.join([i[0] for i in d])
        entity_mentions = NER.recognize(text)
        pred = ['O' for _ in range(len(text))]
        b = 0
        for item in entity_mentions:
            word, typ = item[0], item[1]
            start = text.find(word, b)
            end = start + len(word)
            pred[start] = 'B-' + typ
            for i in range(start + 1, end):
                pred[i] = 'I-' + typ
            b += len(word)
        y_pred.append(pred)
        merged_text += text

    merged_pred = []
    for i, pred in enumerate(y_pred):
        merged_pred.extend(pred)
    merged_pred_str = ' '.join(merged_pred)
    print("Merged Predicted Labels: ", merged_pred_str)
    print("Merged Text: ", merged_text)

    output_file_name = '{}.json'.format(predict_label.file_num)
    output_path = os.path.join(output_folder, output_file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        output = {
            'sentence': merged_text,
            'text': list(merged_text),
            'y_pred': merged_pred
        }
        json.dump(output, f, ensure_ascii=False, indent=4)

    print("Saved JSON file:", output_path)
    print()

    predict_label.file_num += 1

    return y_pred, merged_text

def evaluate():
    data_folder = 'H:/study/pycharm/YEDDA-master/data/分割ann/after/save'
    output_folder = 'H:/study/pycharm37/one/B-L-C/data/relation/predictions/entity2'
    files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    predict_label.file_num = 10000  # 初始化文件编号
    for file in files:
        file_path = os.path.join(data_folder, file)
        test_data, y_true = load_eval_data(file_path, max_len)
        y_pred = predict_label(test_data, y_true, output_folder)

if __name__ == '__main__':
    evaluate()