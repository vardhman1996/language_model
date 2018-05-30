import tensorflow as tf
import data_reader
from data_reader import DataReader
import numpy as np
import sys
import math
import os

MAX_LENGTH = 25
NUM_SENTENCES = 14532
UNK_CHAR = '\u0001'
STOP_CHAR = '\u0003'
START_CHAR = '\u0002'
V = 136755

class LangModel(object):
    def __init__(self, X_dim=32, h_dim=256, max_epoch=10, batch_size=64, keep_param = 0.2):
        self.dr = DataReader('final_sentences.csv', batch_size=batch_size)
        self.max_epoch = max_epoch
        self.X_dim = X_dim
        self.h_dim = h_dim
        self.y_dim = len(self.dr.char_to_num) + 1
        self.batch_size = batch_size
        self.keep_param = keep_param
        self.build_model()
        self.sess = tf.Session()


    def lstm_cell(self, reuse=False):

        with tf.variable_scope('lstm') as vs:
            if reuse:
                vs.reuse_variables()

            return tf.contrib.rnn.BasicLSTMCell(self.h_dim)


    def fc_layer(self, inp, keep_prob= 1.0, reuse=False):

        with tf.variable_scope('fc') as vs:
            if reuse:
                vs.reuse_variables()

            Wt = tf.get_variable(name='weight', shape=[self.h_dim, self.y_dim], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer)
            bias = tf.get_variable(name='bias', shape=[self.y_dim], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            drop_inp = tf.nn.dropout(inp, self.keep_prob)           

            return tf.matmul(drop_inp, Wt) + bias


    def build_model(self):
        # Nodes during Training :
        self.X_train = tf.placeholder(tf.float32, shape=[None, MAX_LENGTH, self.X_dim], name='input')
        self.Y_train = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='labels')
        self.keep_prob = tf.placeholder(tf.float32)

        self.lstm_cell = self.lstm_cell()
        outputs, _ = tf.nn.dynamic_rnn(
            self.lstm_cell, inputs= self.X_train, dtype=tf.float32)

        output_list = tf.unstack(outputs, axis = 1)
        logits = self.fc_layer(output_list[-1])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_train)
        tf.summary.scalar('cross_entropy_loss', tf.reduce_mean(loss))

        self.optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        self.saver = tf.train.Saver(max_to_keep = 4)

        self.merged = tf.summary.merge_all()

        # Nodes during Inference :
        self.X_infer = tf.placeholder(tf.float32, shape=[1, 1, self.X_dim], name='infer_inp')
        # self.initial_state = tf.placeholder(tf.float32, shape=[2, 1, self.h_dim], name='init_state')

        self.init_c = tf.placeholder(tf.float32, shape= [1, self.h_dim],  name='init_c')
        self.init_h = tf.placeholder(tf.float32, shape= [1, self.h_dim],  name='init_h')
        lstm_obj = tf.contrib.rnn.LSTMStateTuple(self.init_c, self.init_h)
        _, (self.infer_state, self.infer_output) = tf.nn.dynamic_rnn(
            self.lstm_cell, inputs=self.X_infer, initial_state=lstm_obj, dtype=tf.float32)

        infer_logits = self.fc_layer(self.infer_output, reuse=True)
        self.y_hat = tf.nn.softmax(infer_logits)

    def train(self, train_id):
        train_writer = tf.summary.FileWriter('train_logs/{}'.format(train_id), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        batches = 80000
        for ep in range(self.max_epoch):
            print("Epoch: {}".format(ep))
            for i, (bx, by) in enumerate(self.dr.get_data(num_batches=batches)):
                summary, _ = self.sess.run([self.merged, self.optim], feed_dict={self.X_train : bx, self.Y_train : by, self.keep_prob : self.keep_param})
                if (i + 1) % 1000 == 0:
                    train_writer.add_summary(summary, ep * batches + i)
                    print("Batch Number: {}".format(i + 1))
            if (ep + 1) % 5 == 0 or (ep + 1) == self.max_epoch:
                self.save(ep + 1, train_id)

    def save(self, ep, train_id):
        checkpoint_dir = 'checkpoint_dir/lstm_h' + str(self.h_dim) + '_b' + str(self.batch_size) + '_T' + str(
            MAX_LENGTH)+'_keep'+str(self.keep_param) + "_" + train_id
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model'), global_step = ep)
        print('Saved model in Epoch {}'.format(ep))

    def load(self, ep, train_id):
        checkpoint_dir = 'checkpoint_dir/lstm_h'+str(self.h_dim)+'_b'+str(self.batch_size)+'_T'+str(MAX_LENGTH) +'_keep'+str(self.keep_param) + "_" + train_id
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model-{}'.format(ep)))
        print('Restored model weights from Epoch {}'.format(ep))

    def infer(self):
        # for data in sys.stdin:
        while True:
            user_input = 'ohoeololqpggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg'
            user_input_chars = [c for c in user_input]
            i = 0

            next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
            first_char = START_CHAR # first is always start
            char_bits = np.array(data_reader.char_to_bit(first_char))
            char_bits = char_bits.reshape((1, 1, 32))

            # get the initial predictions given the start of a sequence
            y_pred, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                   feed_dict={self.X_infer: char_bits, self.init_c: next_c,
                                                              self.init_h: next_h, self.keep_prob : 1.0})

            char_indices = np.arange(len(self.dr.char_to_num) + 1)
            while i < len(user_input_chars):
                if user_input_chars[i] == 'o': # observation
                    next_char = user_input_chars[i + 1]

                    if next_char == STOP_CHAR:
                        # reset next_c and next_h if STOP is observed
                        next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
                        y_prob = y_pred[self.dr.get_char_to_num(next_char)]
                        print(u"// added a character to the history! Probability = {0}".format(math.log(y_prob, 2)))
                        next_char = START_CHAR # since hitory was cleared.
                        char_bits = np.array(data_reader.char_to_bit(next_char))
                        char_bits = char_bits.reshape((1, 1, 32))
                        y_pred_new, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                                   feed_dict={self.X_infer: char_bits,
                                                                              self.init_c: next_c,
                                                                              self.init_h: next_h, self.keep_prob : 1.0})
                        y_pred = y_pred_new
                        i += 2
                        continue

                    char_bits = np.array(data_reader.char_to_bit(next_char))
                    char_bits = char_bits.reshape((1,1,32))

                    # this character's prob is found from previous time step's probability distribution
                    y_prob = y_pred.flatten()[self.dr.get_char_to_num(next_char)]
                    print("// added a character to the history! Probability = {0}".format(math.log(y_prob, 2)))
                    # get the new predictions given the current observation
                    y_pred_new, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                   feed_dict={self.X_infer: char_bits, self.init_c: next_c,
                                                              self.init_h: next_h, self.keep_prob : 1.0})
                    y_pred = y_pred_new
                    i += 1
                elif user_input_chars[i] == 'q':
                    next_char = user_input_chars[i + 1]
                    # use the current probability distribution to get the query prob
                    if next_char not in self.dr.char_to_num:
                        unk_prob = y_pred.flatten()[len(self.dr.char_to_num)]
                        y_prob = unk_prob / (V - len(self.dr.char_to_num))
                        print(math.log(y_prob, 2))
                        i += 2
                        continue

                    y_prob = y_pred.flatten()[self.dr.get_char_to_num(next_char)]
                    print(math.log(y_prob, 2))
                    i += 1
                elif user_input_chars[i] == 'g':
                    # use the current probability distribution to generate a char
                    char_index = np.random.choice(char_indices, 1, p=y_pred.flatten())
                    while char_index == len(self.dr.char_to_num):
                        char_index = np.random.choice(char_indices, 1, p=y_pred.flatten())

                    gen_char = self.dr.get_num_to_char(char_index[0])

                    # clear history if generates a stop char.
                    if gen_char == STOP_CHAR:
                        next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))

                    # if generated STOP then next_char is START
                    next_char = gen_char if gen_char != STOP_CHAR else START_CHAR
                    char_bits = np.array(data_reader.char_to_bit(next_char))
                    char_bits = char_bits.reshape((1, 1, 32))

                    # record generated char in history by getting new next_c and next_h
                    y_pred_new, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                               feed_dict={self.X_infer: char_bits, self.init_c: next_c,
                                                                          self.init_h: next_h, self.keep_prob : 1.0})
                    y_pred = y_pred_new
                    print(u""+gen_char)
                elif user_input_chars[i] == 'x':
                    exit(0)
                i+=1
            break

if __name__=='__main__':
    lm = LangModel(X_dim=32, h_dim=256, max_epoch=10, batch_size=512, keep_param = 0.7)
    run_id = str(input("enter a run id: "))
    lm.train(run_id)
    #lm.load(10, run_id)
    #lm.infer()

