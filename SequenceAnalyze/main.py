import argparse
import random
import numpy as np
import os
import re, logging, datetime, csv, time
from model.Model import Model
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
        dataManager = DataManager(configs, logger)
        model = Model(configs, logger, dataManager)
        if mode == 'train':
            logger.info("mode: train")
            model.train()
        elif mode == 'test':
            logger.info("mode: test")
            model.test()
        elif mode == 'interactive_predict':
            logger.info("mode: predict_one")
            model.soft_load()
            while True:
                print("please input a sentence (enter 'exit' to exit.)\n")
                sentence = input()
                if sentence == 'exit':
                    break
                logger.info(sentence)
                sentence_tokens, entities, entities_type, entities_index = model.predict_single(sentence)
                logger.info("\nExtracted entities:\n %s\n" % ("\n".join([a + "\t(%s)" % b for a, b in zip(entities, entities_type)])))
                
        elif mode == 'learn':
            #X_train, y_train, Xs_val, y_val = dataManager.getTrainingSet()
            dataManager.getTrainingSet()
            """
            build_time = time.time()
            logger.info("model learn start")
            model.learn()
            logger.info("model learn end, time consumption:%.2f(min)\n",((time.time() - build_time) / 60))
            """