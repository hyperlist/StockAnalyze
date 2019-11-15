import math, os
import numpy as np
import tensorflow as tf
import pandas as pd
import time

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(object):
    def __init__(self, configs, logger, dataManager):
        start_time = time.time()
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.CUDA_VISIBLE_DEVICES
        self.configs = configs
        self.logger = logger
        self.logger.info("model init...\n")
        self.logdir = configs.log_dir
        self.dataManager = dataManager
        if configs.mode == "train":
            self.is_training = True
        else:
            self.is_training = False
        #保存目录
        self.checkpoint_name = configs.checkpoint_name
        self.checkpoints_dir = configs.checkpoints_dir
        self.max_to_keep = configs.checkpoints_max_to_keep
        self.print_per_batch = configs.print_per_batch
        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file
        #
        self.num_layers = configs.encoder_layers
        self.is_crf = configs.use_crf
        #
        self.learning_rate = configs.learning_rate
        self.dropout_rate = configs.dropout
        #
        self.batch_size = configs.batch_size
        self.emb_dim = configs.embedding_dim
        self.hidden_dim = configs.hidden_dim
        #attention机制
        self.is_attention = configs.use_self_attention
        self.attention_dim = configs.attention_dim
        self.num_epochs = configs.epoch
        self.max_time_steps = configs.max_sequence_length
        self.num_tokens = dataManager.max_token_number
        self.num_classes = dataManager.max_label_number
        #is_early_stop
        self.is_early_stop = configs.is_early_stop
        self.patient = configs.patient
        #评价指标使用f1值
        self.best_f1_val = -2
        self.best_accuracy_val = 0
        
        #模型初始化
        self.build()
        
        time_span = (time.time() - start_time) / 60
        self.logger.info("model init end, time consumption:%.2f(min)\n",time_span)

    def build(self):               
        self.inputs = tf.placeholder(tf.int32, [None, self.max_time_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.max_time_steps])
        
        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)

        self.initializer = tf.uniform_unit_scaling_initializer()
        self.embedding = tf.get_variable("emb", [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)
        
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.max_time_steps, 0)
        
        
        # LSTM cell
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
        
        #优化器
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        
        #前向LSTM
        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        #反向LSTM
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)
        
        #前向LSTM
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
        #反向LSTM
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))
        
        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)
        
        outputs, _, _ = tf.nn.static_bidirectional_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.inputs_emb,
            dtype=tf.float32,
            sequence_length=self.length
        )
        
        #print(outputs)
        # outputs: list_steps[batch, 2*dim]
        outputs = tf.concat(outputs, 1)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
        
        #print(outputs)
        #print(type(self.length))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            #print(outputs)
            # forward and backward
            

        # linear
        self.outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes],
                                         initializer=self.initializer)
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.initializer)
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        self.logits = tf.reshape(self.logits, [self.batch_size, self.max_time_steps, self.num_classes])
        # print(self.logits.get_shape().as_list())
        if not self.is_crf:
            # softmax
            softmax_out = tf.nn.softmax(self.logits, axis=-1)

            self.batch_pred_sequence = tf.cast(tf.argmax(softmax_out, -1), tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            mask = tf.sequence_mask(self.length)
            self.losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        else:
            # crf
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.length)
            self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.logits,
                                                                                           self.transition_params,
                                                                                           self.length)

            self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.dev_summary = tf.summary.scalar("loss", self.loss)
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        
    def train(self):
        X_train, y_train, X_val, y_val = self.dataManager.getTrainingSet()
        tf.initialize_all_variables().run(session=self.sess)

        saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.logdir + "/training_loss", self.sess.graph)
        dev_writer = tf.summary.FileWriter(self.logdir + "/validating_loss", self.sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))

        cnt = 0
        cnt_dev = 0
        unprogressed = 0
        very_start_time = time.time()
        best_at_epoch = 0
        self.logger.info("\ntraining starting" + ("+" * 20))
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # shuffle train at each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            self.logger.info("\ncurrent epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                X_train_batch, y_train_batch = self.dataManager.nextBatch(X_train, y_train,
                                                                          start_index=iteration * self.batch_size)
                _, loss_train, train_batch_viterbi_sequence, train_summary = \
                    self.sess.run([
                        self.opt_op,
                        self.loss,
                        self.batch_pred_sequence,
                        self.train_summary
                    ],
                        feed_dict={
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                        })

                if iteration % self.print_per_batch == 0:
                    cnt += 1
                    train_writer.add_summary(train_summary, cnt)

                    measures = metrics(X_train_batch, y_train_batch,
                                       train_batch_viterbi_sequence,
                                       self.measuring_metrics, self.dataManager)

                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ": %.3f " % v)
                    self.logger.info("training batch: %5d, loss: %.5f, %s" % (iteration, loss_train, res_str))

            # validation
            loss_vals = list()
            val_results = dict()
            for measu in self.measuring_metrics:
                val_results[measu] = 0

            for iterr in range(num_val_iterations):
                cnt_dev += 1
                X_val_batch, y_val_batch = self.dataManager.nextBatch(X_val, y_val, start_index=iterr * self.batch_size)

                loss_val, val_batch_viterbi_sequence, dev_summary = \
                    self.sess.run([
                        self.loss,
                        self.batch_pred_sequence,
                        self.dev_summary
                    ],
                        feed_dict={
                            self.inputs: X_val_batch,
                            self.targets: y_val_batch,
                        })

                measures = metrics(X_val_batch, y_val_batch, val_batch_viterbi_sequence,
                                   self.measuring_metrics, self.dataManager)
                dev_writer.add_summary(dev_summary, cnt_dev)

                for k, v in measures.items():
                    val_results[k] += v
                loss_vals.append(loss_val)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_f1_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iterations
                val_res_str += (k + ": %.3f " % val_results[k])
                if k == 'f1': dev_f1_avg = val_results[k]

            self.logger.info("time consumption:%.2f(min),  validation loss: %.5f, %s" %
                             (time_span, np.array(loss_vals).mean(), val_res_str))
            if np.array(dev_f1_avg).mean() > self.best_f1_val:
                unprogressed = 0
                self.best_f1_val = np.array(dev_f1_avg).mean()
                best_at_epoch = epoch
                saver.save(self.sess, self.checkpoints_dir + "/" + self.checkpoint_name, global_step=self.global_step)
                self.logger.info("saved the new best model with f1: %.3f" % (self.best_f1_val))
            else:
                unprogressed += 1

            if self.is_early_stop:
                if unprogressed >= self.patient:
                    self.logger.info("early stopped, no progress obtained within %d epochs" % self.patient)
                    self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
                    self.logger.info(
                        "total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
                    self.sess.close()
                    return
        self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
        self.logger.info("total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
        self.sess.close()
        
    def metrics(self, y_true, y_pred):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        accuracy = -1
        
        hit_num = 0
        pred_num = 0
        true_num = 0

        correct_label_num = 0
        total_label_num = 0
        for i in range(len(y_true)):
            x = [str(dataManager.id2token[val]) for val in X[i] if val != dataManager.token2id[dataManager.PADDING]]
            y = [str(dataManager.id2label[val]) for val in y_true[i] if val != dataManager.label2id[dataManager.PADDING]]
            y_hat = [str(dataManager.id2label[val]) for val in y_pred[i] if
                     val != dataManager.label2id[dataManager.PADDING]]  # if val != 5

            correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
            total_label_num += len(y)

            true_labels, labled_labels, _ = extractEntity(x, y, dataManager)
            pred_labels, labled_labels, _ = extractEntity(x, y_hat, dataManager)

            hit_num += len(set(true_labels) & set(pred_labels))
            pred_num += len(set(pred_labels))
            true_num += len(set(true_labels))

        if total_label_num != 0:
            accuracy = 1.0 * correct_label_num / total_label_num

        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_numv
            f1 = 2.0 * (precision * recall) / (precision + recall)
        results = dict()
        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = f1
        results["accuracy"] = accuracy
        return results
    
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

