#! -*- coding: utf-8 -*-
import os
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from bert4keras.backend import K,keras,search_layer
from bert4keras.snippets import ViterbiDecoder, to_array

from data_utils import *
from build_model import bert_bilstm_crf



seed = 233
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


epochs = 1
max_len = 200
batch_size = 16
lstm_units = 128
drop_rate = 0.1
leraning_rate = 5e-5

config_path = 'H:/study/pycharm37/one/bert_for_ner/chinese-bert-m/bert_config.json'
checkpoint_path = 'H:/study/pycharm37/one/bert_for_ner/chinese-bert-m/bert_model.ckpt'
checkpoint_save_path = './checkpoint/bert_bilstm_crf.weights'


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > max_len:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids]) # ndarray
        nodes = model.predict([token_ids, segment_ids])[0] # [sqe_len,23]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities]

def ner_metrics(data):
    X,Y,Z = 1e-6,1e-6,1e-6
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        pred = NER.recognize(text)
        R = set(pred)
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self,epoch,logs=None):
        NER.trans = K.eval(CRF.trans)
        f1, precision, recall = ner_metrics(valid_data)
        if f1 > self.best_val_f1:
            model.save_weights(checkpoint_save_path)
            self.best_val_f1 = f1 
            print('save model to {}'.format(checkpoint_save_path))
        else:
            global leraning_rate
            leraning_rate = leraning_rate / 5 
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

def adversarial_training(model, embedding_name, epsilon=1):


    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):
        # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs
    model.train_function = train_function  # 覆盖原训练函数


model,CRF = bert_bilstm_crf(
        config_path,checkpoint_path,num_labels,lstm_units,drop_rate,leraning_rate)

NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

if __name__ == '__main__':

    train_path = './data/train.txt'
    test_path = './data/test.txt'
    train_data,_ = load_data(train_path,max_len)
    valid_data,_ = load_data(test_path,max_len)

    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size*5)

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_save_path, 
        monitor='val_sparse_accuracy', 
        verbose=1, 
        save_best_only=True,
        mode='max'
        )

    evaluator = Evaluator()

    def scheduler(epoch):
        return leraning_rate /(max(2*(epoch-1),1))

    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        epochs=epochs,
        callbacks=[evaluator,lr_scheduler]
    )

    print(K.eval(CRF.trans))
    print(K.eval(CRF.trans).shape)
    pickle.dump(K.eval(CRF.trans),open('./checkpoint/crf_trans.pkl','wb'))

else:
    model.load_weights(checkpoint_save_path)
    NER.trans = pickle.load(open('./checkpoint/crf_trans.pkl','rb'))