import os
from time import time
import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import svm
from sklearn.externals import joblib
from sklearn import metrics

class SVM(object):
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger
        self.pos_file = os.path.join(configs.datasets_fold, configs.pos_corpus)
        self.neg_file = os.path.join(configs.datasets_fold, configs.neg_corpus)
        self.modelpath = os.path.join(configs.checkpoints_dir, "SVM.pkl")
        self.vectpath = os.path.join(configs.checkpoints_dir, "vect.v")
        
    def load_dataset_tokenized(self):
        pos_sents = []
        with open(self.pos_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.split(' ')
                sent = []
                for t in tokens:
                    if t.strip():
                        sent.append(t.strip())
                pos_sents.append(sent)

        neg_sents = []
        with open(self.neg_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.split(' ')
                sent = []
                for t in tokens:
                    if t.strip():
                        sent.append(t.strip())
                neg_sents.append(sent)

        balance_len = min(len(pos_sents), len(neg_sents))

        texts = pos_sents + neg_sents
        labels = [1] * balance_len + [0] * balance_len

        return texts, labels
    def dummy_fun(self,doc):
            return doc

    def train(self):
        self.logger.info('Loading dataset...')
        clf = svm.LinearSVC()
        X, y = self.load_dataset_tokenized()

        vectorizer = TfidfVectorizer(analyzer='word',
                                     tokenizer=self.dummy_fun,
                                     preprocessor=self.dummy_fun,
                                     token_pattern=None)
        X = vectorizer.fit_transform(X)
        
        self.logger.info('Train model...')
        clf.fit(X, y)
        
        self.logger.info('Saving Model...')
        joblib.dump(clf, self.modelpath)
        joblib.dump(vectorizer, self.vectpath)
        
    
    def analyze(self,id,file):
        self.logger.info('Loading %s comments at %s...' %(id,file))
        path = os.path.join('ID-'+id, file)
        comment_file = os.path.join(self.configs.datasets_fold, path)
        print(comment_file)
        df = pd.read_csv(comment_file)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['created_time'] = pd.to_datetime(df['created_time'], format='%Y-%m-%d %H:%M:%S')
        df['polarity'] = 0
        df['title'].apply(lambda x: [w.strip() for w in x.split()])
        
        texts = df['title']
        texts = self.vect.transform(texts)
        
        preds = self.clf.predict(texts)
        df['polarity'] = preds
        self.logger.info('Saving Result...')
        respath = os.path.join(self.configs.results_dir, id+'.csv')
        if not os.path.exists(respath):
            df.to_csv(respath, index=False,mode='a')
        else:
            df.to_csv(respath, index=False, header=False,mode='a')
            self.clean(respath)
        
    def clean(self, filepath):
        df = pd.read_csv(filepath)
        df.drop_duplicates(subset='id', keep='first', inplace=True)
        df.to_csv(filepath, index=False)
        
    def soft_load(self):
        self.clf = joblib.load(self.modelpath)
        self.vect = joblib.load(self.vectpath)
        
    def predict_single(self, sent):
        sent = list(jieba.cut(sent))
        sent = ' '.join(sent)
        texts= [sent]
        texts = self.vect.transform(texts)
        preds = self.clf.predict(texts)
        self.logger.info("%s    %s" %(sent, str(preds)))
        return preds
        
        