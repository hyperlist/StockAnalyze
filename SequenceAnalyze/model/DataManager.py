# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: DataManager.py
# @Project: sequence-lableing-vex

import os, logging,csv
import numpy as np
import pandas as pd

class DataManager:
    def __init__(self, configs, logger):
        self.configs=configs
        self.train_file = configs.train_file
        self.logger = logger

        self.UNKNOWN = "<UNK>"
        self.PADDING = "<PAD>"

        self.train_file = configs.datasets_fold + "/" + configs.train_file
        self.dev_file = configs.datasets_fold + "/" + configs.dev_file
        self.test_file = configs.datasets_fold + "/" + configs.test_file
        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim

        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + "/token2id"
        self.label2id_file = self.vocabs_dir + "/label2id"

        self.token2id, self.id2token, self.label2id, self.id2label = self.loadVocab()

        self.max_token_number = len(self.token2id)
        self.max_label_number = len(self.label2id)

        self.logger.info("dataManager initialed...\n")

    def loadVocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info("vocab files not exist, building vocab...")
            return self.buildVocab()

        self.logger.info("loading vocab...")
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return token2id, id2token, label2id, id2label

    def buildVocab(self):
        df_train = pd.read_csv(self.train_file, sep=" ", quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=["sentence", "label"])
        tokens = set()
        for items in df_train["sentence"][df_train["sentence"].notnull()]:
            tokens.update(list(items))
        tokens = list(tokens)

        labels = list(set(df_train["label"][df_train["label"].notnull()]))
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1
        
        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")
        with open(self.label2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(str(id2label[idx]) + "\t" + str(idx) + "\n")
        
        return token2id, id2token, label2id, id2label
        

                
    def getEmbedding(self, embed_file):
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=(self.embedding_dim))

        with open(embed_file, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), "embedding dim must be consistent with the one in `token_emb_dir'."
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec

        return emb_matrix

    

    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, tokens, labels, is_padding=True):
        x = []
        y = []
        y_psyduo = []
        tmp_x = []
        tmp_y = []
        tmp_y_psyduo = []

        for record in zip(tokens, labels):
            label = self.label2id[record[1]]
            token = []
            for word in record[0]:
                token.append(self.token2id[word])
            x.append(token)
            y.append(label)
        if is_padding:
            x = np.array(self.padding(x))
        else:
            x = np.array(x)
        y = np.array(y)
        return x, y

    def getTrainingSet(self, train_val_ratio=0.9):
        df_train = pd.read_csv(self.train_file, sep=" ", quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=["token", "label"])
        # map the token and label into id
        #df_train["token_id"] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        #df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        
        # convert the data in maxtrix
        X, y = self.prepare(df_train["token"], df_train["label"])
        
        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]
        if os.path.exists(self.dev_file):
            X_train = X
            y_train = y
            X_val, y_val = self.getValidingSet()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]

        self.logger.info("\ntraining set size: %d, validating set size: %d\n" % (len(X_train), len(y_val)))
        
        return X_train, y_train, X_val, y_val
        
        
    def getValidingSet(self):
        df_val = pd.read_csv(self.dev_file, sep=" ", quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=["token", "label"])

        X_val, y_val = self.prepare(df_val["token"], df_val["label"])
        return X_val, y_val

    def getTestingSet(self):
        df_test = pd.read_csv(self.dev_file, sep=" ", quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=None)

        if len(list(df_test.columns)) == 2:
            df_test.columns = ["token", "label"]
            df_test = df_test[["token"]]
        elif len(list(df_test.columns)) == 1:
            df_test.columns = ["token"]

        df_test["token_id"] = df_test.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_test["token"] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else x)

        X_test_id, y_test_psyduo_label = self.prepare(df_test["token_id"], df_test["token_id"],
                                                      return_psyduo_label=True)
        X_test_token, _ = self.prepare(df_test["token"], df_test["token"])

        self.logger.info("\ntesting set size: %d\n" % (len(X_test_id)))
        return X_test_id, y_test_psyduo_label, X_test_token

    def mapFunc(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        sentence = list(sentence)
        gap = self.batch_size - 1

        x_ = []
        y_ = []

        for token in sentence:
            try:
                x_.append(self.token2id[token])
            except:
                x_.append(self.token2id[self.UNKNOWN])
            y_.append(self.label2id["O"])

        if len(x_) < self.max_sequence_length:
            sentence += ['x' for _ in range(self.max_sequence_length - len(sentence))]
            x_ += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x_))]
            y_ += [self.label2id["O"] for _ in range(self.max_sequence_length - len(y_))]
        elif len(x_) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x_ = x_[:self.max_sequence_length]
            y_ = y_[:self.max_sequence_length]

        X = [x_]
        Sentence = [sentence]
        Y = [y_]
        X += [[0 for j in range(self.max_sequence_length)] for i in range(gap)]
        Sentence += [['x' for j in range(self.max_sequence_length)] for i in range(gap)]
        Y += [[self.label2id['O'] for j in range(self.max_sequence_length)] for i in range(gap)]
        X = np.array(X)
        Sentence = np.array(Sentence)
        Y = np.array(Y)
        return X, Sentence, Y
        
    def batch_iter(x, y, batch_size=64):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
            
    def nextBatch(self, X, y, start_index):
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch
        
    def check_contain_chinese(self, check_str):
        for ch in list(check_str):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
