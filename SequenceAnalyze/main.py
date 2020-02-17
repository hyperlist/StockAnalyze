import argparse
import random
import numpy as np
import os
import re, logging, datetime, csv, time
from model.Model import SVM
from model.DataManager import DataManager
from model.Configer import Configer

def get_logger(log_dir):
    log_file = log_dir + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d: %H %M %S'))
    return logger

def set_env(configs):
    random.seed(configs.seed)
    np.random.seed(configs.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning with BiLSTM+CRF')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configer(config_file=args.config_file)
    if not os.path.exists(configs.results_dir):
        os.makedirs(configs.results_dir)
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    set_env(configs)

    mode = configs.mode.lower()

    if mode == 'api_service':
        logger.info("mode: api service")
        try:
            cmd_new = r'python ./newapp/manage.py runserver %s:%s' % (configs.ip, configs.port)
            res = os.system(cmd_new)
        except Exception as err:
            print(err)
    else:
        #dataManager = DataManager(configs, logger)
        model = SVM(configs, logger)
        if mode == 'train':
            logger.info("mode: train")
            model.train()
        elif mode == 'analyze':
            logger.info("mode: analyze")
            model.soft_load()
            urls = csv.DictReader(open(os.path.join(configs.datasets_fold, "stocks.csv"), 'r', encoding='utf-8'))
            for item in list(urls):
                file = datetime.datetime.now().strftime('%Y-%m-%d')+'.csv'
                model.analyze(item['id'],file)
        elif mode == 'interactive_predict':
            logger.info("mode: predict_one")
            model.soft_load()
            while True:
                print("please input a sentence (enter 'exit' to exit.)\n")
                sentence = input()
                if sentence == 'exit':
                    break
                res = model.predict_single(sentence)
                
        