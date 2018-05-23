
import tensorflow as tf
from data_reader import  DataReader
import numpy as np

MAX_LENGTH = 25
NUM_SENTENCES = 14532

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


        # infer_output_list = tf.unstack(infer_outputs, axis=1)
        # self.infer_output = tf.reshape(infer_output, shape=[1, self.h_dim])

        # infer_logits = self.fc_layer(infer_output_list[-1], reuse=True)
        infer_logits = self.fc_layer(self.infer_output, reuse=True)
        self.y_hat = tf.nn.softmax(infer_logits)

        # logits = self.fc_layer(output_list[0])
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_train[:,0,:])
        # for t in range(1, len(output_list)):
        #     logits = self.fc_layer(output_list[t], reuse=True)
        #     loss += tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_train[:,t,:])




        # output_list = tf.unstack(outputs, axis=1, name='list')
        # y_hat_list = []
        # for t in range(1, len(output_list)):
        #     logits = self.fc_layer(output_list[t], reuse=True)
        #     y_hat_list.append(tf.nn.softmax(logits))
        # return y_hat_list, final_state

    def train(self):

        self.sess.run(tf.global_variables_initializer())
        dr = DataReader('final_sentences_sample.csv', batch_size=32)
        for ep in range(self.max_epoch):
            for i, (bx, by) in enumerate(dr.get_data(num_batches=10)):
                self.sess.run(self.optim, feed_dict={self.X_train : bx, self.Y_train : by})
                print(i)

        self.save()




    def save(self):

        pass

    def load(self):
        pass

    def infer(self):
        next_c, next_h = np.zeros((1, self.h_dim)), np.zeros((1, self.h_dim))
        for i, (bx, by) in enumerate(dr.get_data(num_batches=10)):
            bx = bx[0, 0, :]
            bx = np.reshape(bx, newshape=(1, 1, 32))

            y_pred, next_c, next_h = self.sess.run([self.y_hat, self.infer_state, self.infer_output],
                                                   feed_dict={self.X_infer: bx, self.init_c: next_c,
                                                              self.init_h: next_h})



if __name__=='__main__':


    lm = LangModel(None, None, X_dim = 32, h_dim = 256, y_dim = 13695, max_epoch = 1, batch_size = 32)
    lm.train()