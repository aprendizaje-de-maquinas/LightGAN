'''
This module implements the LightLSTM approach
'''
import tensorflow as tf
import numpy as np

class LightLSTM(tf.nn.rnn_cell.RNNCell):
    '''
    This class implements a single cell using the LigthLSTM design
    Descends from RNNCell so that we can use it with tf RNN ops
    '''
    def __init__(self, input_size, lstm_size, dtype=tf.float32, scope=None, reuse=False):
        '''
        Creates the cell and saves the size of the input and hidding units
        '''
        super(LightLSTM, self).__init__(_reuse=reuse)

        self._input_size = input_size
        self._lstm_size = lstm_size
        self._dtype = dtype

    @property
    def state_size(self):
        '''
        Used by tf RNN ops, returns the size of the state (the -1 of the shape)
        '''
        return self._lstm_size

    @property
    def output_size(self):
        '''
        used by tf RNN ops used to know the size of the output (the -1 of the shape)
        '''
        return self._lstm_size

    def __call__(self, inputs, state, input_dropout=None, output_dropout=None, scope=None):
        '''
        Used by tf RNN ops for a single call to this cell

        inputs = row stacked on top of col
        state = dh stacked on top of dc, necesary for internal computations
        '''
        # for inits
        dim = self._input_size
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        # get the three params for the LightLSTM similar to the params in regular LSTM
        W = tf.get_variable('Matrix', [dim, self._lstm_size*4],\
                                    dtype=self._dtype, initializer=initializer)
        b = tf.get_variable('bias', [self._lstm_size*4],\
                                    dtype=self._dtype, initializer=initializer)
        H = tf.get_variable('H', [self._lstm_size, self._lstm_size*4],\
                                    dtype=self._dtype, initializer=initializer)


        # split the state and the inputs
        dh, dc = tf.split(state, 2, 0)
        x, y = tf.split(inputs, 2, 0)

        proj4 = b + tf.matmul(x, W) + tf.matmul(dh, H)

        it_proj, bit_proj, ft_proj, ot_proj = tf.split(proj4, 4, -1)

        it = tf.sigmoid(it_proj)
        bit = it * tf.tanh(bit_proj)

        ft = tf.sigmoid(ft_proj)
        bft = ft * dc

        ct = tf.add(bft, bit)
        ot = tf.sigmoid(ot_proj)
        ht = ot * tf.tanh(ct)

        proj4_2 = b + tf.matmul(y, W) + tf.matmul(ht, H)

        it_proj_2, bit_proj_2, ft_proj_2, ot_proj_2 = tf.split(proj4_2, 4, -1)

        it_2 = tf.sigmoid(it_proj_2)
        bit_2 = it_2 * tf.tanh(bit_proj_2)

        ft_2 = tf.sigmoid(ft_proj_2)
        bft_2 = ft_2 * ct

        ct2 = tf.add(bft_2, bit_2)
        ot_2 = tf.sigmoid(ot_proj_2)
        ht2 = ot_2 * tf.tanh(ct2)

        # concat row and col, concat the two states
        return tf.concat([ht, ht2], axis=0), tf.concat([ht2, ct2], axis=0)
