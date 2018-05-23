import tensorflow as tf
import data_reader
from data_reader import DataReader
import numpy as np
import sys
import math

MAX_LENGTH = 25
NUM_SENTENCES = 14532
UNK_CHAR = '\u0001'
STOP_CHAR = '\u0003'
START_CHAR = '\u0002'


class LangModel(object):

    def __init__(self, fw_dic, bk_dic, X_dim = 32, h_dim = 256, y_dim = 13695, max_epoch = 10, batch_size = 32):

        self.max_epoch = max_epoch
        self.X_dim = X_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.fw_indx_lookup = fw_dic
        self.rev_indx_lookup = bk_dic

        self.build_model()
        self.sess = tf.Session()

        self.dr = DataReader('final_sentences_sample.csv', batch_size=32)

    def lstm_cell(self, reuse = False):

        with tf.variable_scope('lstm') as vs:
            if reuse:
                vs.reuse_variables()

            return tf.contrib.rnn.BasicLSTMCell(self.h_dim)


    def fc_layer(self, inp, reuse = False):

        with tf.variable_scope('fc') as vs:
            if reuse:
                vs.reuse_variables()

            Wt = tf.get_variable(name='weight', shape=[self.h_dim, self.y_dim], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer)
            bias = tf.get_variable(name='bias', shape=[self.y_dim], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))

            return tf.matmul(inp, Wt) + bias


    def build_model(self):
        # Nodes during Training :
        self.X_train = tf.placeholder(tf.float32, shape=[None, MAX_LENGTH, self.X_dim], name='input')
        self.Y_train = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='labels')

        self.lstm_cell = self.lstm_cell()
        outputs, _ = tf.nn.dynamic_rnn(
            self.lstm_cell, inputs= self.X_train, dtype=tf.float32)

        output_list = tf.unstack(outputs, axis = 1)
        logits = self.fc_layer(output_list[-1])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_train)

        self.optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        self.saver = tf.train.Saver()

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

    def train(self):

        self.sess.run(tf.global_variables_initializer())
        for ep in range(self.max_epoch):
            print("Epoch: {}".format(ep))
            for i, (bx, by) in enumerate(self.dr.get_data(num_batches=2000)):
                self.sess.run(self.optim, feed_dict={self.X_train : bx, self.Y_train : by})
                if (i + 1) % 100 == 0:
                    print("Batch Number: {}".format(i + 1))

        # self.save()


    def save(self):

        pass

    def load(self):
        pass

    def infer(self):
        # for data in sys.stdin:
        while True:
            user_input = 'ohoaonodolggggggggggg'
            user_input_chars = [c for c in user_input]
            i = 0


            next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
            first_char = START_CHAR # first is always start
            char_bits = np.array(data_reader.char_to_bit(first_char))
            char_bits = char_bits.reshape((1, 1, 32))

            # get the initial predictions given the start of a sequence
            y_pred, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                   feed_dict={self.X_infer: char_bits, self.init_c: next_c,
                                                              self.init_h: next_h})

            char_indices = np.arange(len(self.dr.char_to_num))
            while i < len(user_input_chars):
                if user_input_chars[i] == 'o': # observation
                    next_char = user_input_chars[i + 1]

                    if next_char == STOP_CHAR:
                        # reset next_c and next_h if STOP is observed
                        next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
                        y_prob = y_pred[self.dr.get_char_to_num(next_char)]
                        print(u"// added a character to the history! Probability = {0}".format(math.log(y_prob, 2)))
                        i+=2 # check this again
                        continue

                    char_bits = np.array(data_reader.char_to_bit(next_char))
                    char_bits = char_bits.reshape((1,1,32))
                    # get the new predictions given the current observation
                    y_pred_new, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                   feed_dict={self.X_infer: char_bits, self.init_c: next_c,
                                                              self.init_h: next_h})

                    # this character's prob is found from previous time step's probability distribution
                    y_prob = y_pred.flatten()[self.dr.get_char_to_num(next_char)]
                    print("// added a character to the history! Probability = {0}".format(math.log(y_prob, 2)))
                    y_pred = y_pred_new
                    i += 1

                elif user_input_chars[i] == 'q':
                    next_char = user_input_chars[i + 1]
                    # use the current probability distribution to get the query prob
                    y_prob = y_pred.flatten()[self.dr.get_char_to_num(next_char)]
                    print(math.log(y_prob, 2))
                    i += 1

                elif user_input_chars[i] == 'g':
                    # use the current probability distribution to generate a char
                    char_index = np.random.choice(char_indices, 1, p=y_pred.flatten())
                    gen_char = self.dr.get_num_to_char(char_index[0])

                    # clear history if generates a stop char.
                    if gen_char == STOP_CHAR:
                        next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))


                    # while not is_valid_bmp(ord(gen_char)):
                    #     char_index = np.random.choice(char_indices, 1, p=y_pred)
                    #     gen_char = self.dr.get_num_to_char(char_index)

                    # if generated STOP then next_char is START
                    next_char = gen_char if gen_char != STOP_CHAR else START_CHAR
                    char_bits = np.array(data_reader.char_to_bit(next_char))
                    char_bits = char_bits.reshape((1, 1, 32))

                    # record generated char in history by getting new next_c and next_h
                    y_pred_new, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                               feed_dict={self.X_infer: char_bits, self.init_c: next_c,
                                                                          self.init_h: next_h})
                    y_pred = y_pred_new
                    print(u""+gen_char)
                elif user_input_chars[i] == 'x':
                    exit(0)
                i+=1
            break

if __name__=='__main__':
    lm = LangModel(None, None, X_dim = 32, h_dim = 256, y_dim = 13695, max_epoch = 1, batch_size = 32)
    lm.train()
    lm.infer()