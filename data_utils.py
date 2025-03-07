#! -*- coding: utf-8 -*-
import re
import pandas as pd
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer

entity_labels = ['产品', '产地', '产量','面积','类别','重量','历史','别名','荣誉','地理坐标','扩展产品','产地美名','品种']

id2label = {i:j for i,j in enumerate(sorted(entity_labels))}
label2id = {j: i for i, j in id2label.items()}
num_labels = len(entity_labels) * 2 + 1 # b i o

vocab_path = 'H:/study/pycharm37/one/bert_for_ner/chinese-bert-m/vocab.txt'
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
max_len = 200

def load_data(data_path,max_len):
    """加载数据    """
    datasets = []
    samples_len = []
    
    X = []
    y = []
    sentence = []
    labels = []
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    with open(data_path,'r',encoding = 'utf8') as f:
        for line in f.readlines():
            #每行为一个字符和其tag，中间用tab或者空格隔开
            line = line.strip().split()
            if(not line or len(line) < 2): 
                X.append(sentence)
                y.append(labels)
                sentence = []
                labels = []
                continue
            word = line[0]
            tag = re.sub(r'^M','I',line[1]) # BMESO -> BIO
            tag = re.sub(r'^E','I',tag)
            tag = re.sub(r'^S','B',tag)
            tag = tag.replace('_','-')
            if split_pattern.match(word) and len(sentence)+8 >= max_len:
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

    for token_seq,label_seq in zip(X,y):
        if len(token_seq) < 2:
            continue
        sample_seq, last_flag = [], ''
        for token, this_flag in zip(token_seq,label_seq):
            if this_flag == 'O' and last_flag == 'O':
                sample_seq[-1][0] += token
            elif this_flag == 'O' and last_flag != 'O':
                sample_seq.append([token, 'O'])
            elif this_flag[:1] == 'B':
                sample_seq.append([token, this_flag[2:]]) # B-city
            else:
                if sample_seq:
                    sample_seq[-1][0] += token
            last_flag = this_flag

        datasets.append(sample_seq)
        samples_len.append(len(token_seq))
        if len(token_seq) > 200:
            print(token_seq)

    df = pd.DataFrame(samples_len)
    print(data_path,'\n',df.describe())
    print(sorted(set([i for arr in y for i in arr])))
    return datasets,y


class data_generator(DataGenerator):
    """数据生成器    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0] #[CLS]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < max_len:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id] # [sep]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == '__main__':

    data_path = 'H:/study/pycharm37/one/B-L-C/data/train.txt'
    d,y= load_data(data_path,max_len)
    print(d[:6])