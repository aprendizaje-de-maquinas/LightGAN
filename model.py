'''
This Module implements the Generator and Discriminator Netoworks
'''
import tensorflow as tf
from lightrnn import LightLSTM
from config import *

# this simplifies the rnn
embed_dim = FLAGS.DISC_STATE_SIZE

DROPOUT = 0.75


def discriminator(inputs, wordmap_len, seq_len, reuse=False):
    '''
    This dunction creates a discriminator network

    inputs = the input to the discriminator could be from generator or ground truth
    wordmap_len = the dimmension of the embedding table (ie ciel(sqrt(|V|)))
    seq_len = the length of the sequences to be classified
    reuse = whether tf should reuse parameters
    '''
    with tf.variable_scope('Discriminator', reuse=reuse):

        num_neurons = FLAGS.DISC_STATE_SIZE

        # the embeding layers. mapping from the seqrt of vocab size to the num units in LSTM
        init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        weight_r = tf.get_variable('embedding_r', shape=[wordmap_len, embed_dim], initializer=init)
        weight_c = tf.get_variable('embedding_c', shape=[wordmap_len, embed_dim], initializer=init)

        # these are extra weights for if we make embed_dim different from num_neurons
        #w1 = tf.get_variable('w_r', shape=[embed_dim, num_neurons], initializer=init)
        #w2 = tf.get_variable('w_c', shape=[embed_dim, num_neurons], initializer=init)


        # create the LightLSTM cells for the rnn
        cells = []
        for scope in range(FLAGS.DISC_GRU_LAYERS):
            cell = LightLSTM(num_neurons, num_neurons, scope=scope)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=DROPOUT)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # we cannot perform matmul on  three dimensional input so reshape to two
        flat_inputs_r = tf.reshape(inputs[0], [-1, wordmap_len])
        flat_inputs_c = tf.reshape(inputs[1], [-1, wordmap_len])

        # perfrom the embedding
        inputs_r = tf.reshape(tf.matmul(flat_inputs_r, weight_r), [-1, seq_len, num_neurons])
        inputs_c = tf.reshape(tf.matmul(flat_inputs_c, weight_c), [-1, seq_len, num_neurons])
        #inputs_r = tf.reshape(tf.matmul(flat_inputs_r, weight_r), [-1, embed_dim])
        #inputs_c = tf.reshape(tf.matmul(flat_inputs_c, weight_c), [-1, embed_dim])

        # this is if the embed_dim is not equal to num_neurons
        #    however, we have ensured embed_dim == num_neurons
        #inputs_r = tf.reshape(tf.matmul(inputs_r, w1), [-1, seq_len, num_neurons])
        #inputs_c = tf.reshape(tf.matmul(inputs_c, w2), [-1, seq_len, num_neurons])

        # the LightLSTM assumes the row is stacked on top of column
        inputs = tf.concat([inputs_r, inputs_c], axis=0)

        # unstack based on time step and run through rnn
        inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
        output, state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)

        # concat the state and the output from the last layer of the rnn
        state = tf.reshape(state[-1], [-1, num_neurons])
        last = tf.reshape(output[-1], [-1, num_neurons])
        last = tf.concat([last, state], axis=1)

        # perform fully connected layer to 1 output (ie is this batch real or fake)
        shape = last.get_shape().as_list()[1]
        weight = tf.get_variable('W', shape=[shape, 1], initializer=init)
        bias = tf.get_variable('b', shape=[1], initializer=init)
        prediction = tf.matmul(last, weight) + bias

        return prediction

def generator(n_samples, wordmap_len, naw_r, naw_c, seq_len=None, gt=None):
    '''
    This function implements the generator of the gan

    n_samples = the number of fake samples to generate
    wordmap_len = the ceil of the sqrt of the vocab size
    naw_r = the row index of the <naw> token in the map
    naw_c = the col index of the <naw> token in the map
    seq_len = the length of the sequence to generate
    gt = the ground truth inputs (ie the subtrings for us to predict the next word of)
    '''
    with tf.variable_scope('Generator'):
        # noise for the initial states
        noise, _ = get_noise()
        num_neurons = FLAGS.GEN_STATE_SIZE

        # get the LightLSTM cells and wrap them with dropout
        cells = []
        for scope in range(FLAGS.GEN_GRU_LAYERS):
            cell = LightLSTM(num_neurons, num_neurons, scope=scope)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=DROPOUT)
            cells.append(cell)

        # create the initial states for the training and inference predictions
        train_i_states, infer_i_states = create_initial_states(noise), create_initial_states(noise)

        # softax parameters for the row (1) and col (2)
        sm_weight1 = tf.Variable(tf.random_uniform([num_neurons, wordmap_len], \
                                                  minval=-0.1, maxval=.1), name='sftmax-r-w')
        sm_bias1 = tf.Variable(tf.random_uniform([wordmap_len], minval=-0.1, \
                                                maxval=.1), name='sftmax-r-b')
        sm_weight2 = tf.Variable(tf.random_uniform([num_neurons, wordmap_len], \
                                                  minval=-0.1, maxval=.1), name='sftmax-c-w')
        sm_bias2 = tf.Variable(tf.random_uniform([wordmap_len], minval=-0.1, \
                                                maxval=.1), name='sftmax-c-b')

        # the emedding matrices for the row (1) and col (2)
        embedding1 = tf.Variable(tf.random_uniform([wordmap_len, embed_dim], \
                                                  minval=-0.1, maxval=.1), name='embed-r')
        embedding2 = tf.Variable(tf.random_uniform([wordmap_len, embed_dim], \
                                                  minval=-0.1, maxval=.1), name='embed-c')

        # group the row and col parameters together for ease of passing
        sm_weight = [sm_weight1, sm_weight2]
        sm_bias = [sm_bias1, sm_bias2]
        embedding = [embedding1, embedding2]

        # this is noise that we will put at the beginning of the sequence to start the rnn off
        word_input1 = tf.Variable(tf.random_uniform([num_neurons], \
                                                   minval=-0.1, maxval=.1), name='word-input-r')
        word_input2 = tf.Variable(tf.random_uniform([num_neurons], \
                                                   minval=-0.1, maxval=.1), name='word-input-c')
        word_input1 = tf.reshape(tf.tile(word_input1, [n_samples]), [n_samples, 1, num_neurons])
        word_input2 = tf.reshape(tf.tile(word_input2, [n_samples]), [n_samples, 1, num_neurons])

        # group
        word_input = [word_input1, word_input2]

        # get the op for training (ie using the gt_inputs) and for reallocing (ie probabilities of each word from softmax)
        train_pred, realloc_pred = get_train_op(cells, word_input, wordmap_len, \
                                                embedding, gt, n_samples, num_neurons, seq_len, \
                                                sm_bias, sm_weight, train_i_states, naw_r, naw_c)
        # get the op for inference (ie no gt) note that reuse must be true to share the params from training
        inference_op = get_inference_op(cells, word_input, embedding, seq_len, \
                                        sm_bias, sm_weight, infer_i_states, \
                                        num_neurons, wordmap_len, reuse=True)

        return train_pred, inference_op, realloc_pred


def create_initial_states(noise):
    '''
    Creates the initial states for all the generator layers according to noise
    '''
    states = []
    for _ in range(FLAGS.GEN_GRU_LAYERS):
        states.append(noise)
    return states

def get_train_op(cells, word_input, wordmap_len, embedding, gt, n_samples, \
                 num_neurons, seq_len, sm_bias, sm_weight, states, naw_r, naw_c):
    '''
    Uses the network define to predict the next work in the sequence provided in gt

    cells = list containing the LightLSTM cells
    work_input = noise for the begining of the sequence
    wordmap_len = ciel of sqrt |V|
    embedding = the matricies for the embeding layer
    gt = the ground truth sequence we are predicting the next word for
    n_samples = the number of fake sequences to produce
    num_neurons = the number of neurons in each LSTM
    seq_len = the length of the sequence
    sm_bias = the bias terms for softmax
    sm_weight = the weights for the softmax
    states = the initial states of the rnn(noise)
    naw_r = the row index of the <naw> token in the map
    naw_c = the col index of the <naw> token in the map
    '''
    # to us  matmul, we need 2D shape, also performs col and row embeding
    # note that index 0 refers to col and so does 1
    gt_embedding1 = tf.reshape(gt[0], [n_samples * seq_len, wordmap_len])
    gt_input1 = tf.matmul(gt_embedding1, embedding[0])
    gt_embedding2 = tf.reshape(gt[1], [n_samples * seq_len, wordmap_len])
    gt_input2 = tf.matmul(gt_embedding2, embedding[1])


    # get the sequence all the way up to the one that we are going to predict (the last one)
    gt_input1 = tf.reshape(gt_input1, [n_samples, seq_len, num_neurons])[:, :-1]
    gt_input2 = tf.reshape(gt_input2, [n_samples, seq_len, num_neurons])[:, :-1]

    gt_input1 = tf.nn.dropout(gt_input1, DROPOUT)
    gt_input2 = tf.nn.dropout(gt_input2, DROPOUT)

    # concat the row and col together for the LightLSTM
    gt_input, w_input = tf.concat([gt_input1, gt_input2], axis=0), \
                        tf.concat([word_input[0], word_input[1]], axis=0)

    # concat the gt together with the noise to preserve the sequence length
    gt_sentence_input = tf.concat([w_input, gt_input], axis=1)

    # perform the step prediction
    row, col, _ = rnn_step_prediction(cells, wordmap_len, gt_sentence_input,\
                                      num_neurons, seq_len, sm_bias, sm_weight, states)

    # create training substrings.
    # row and col hold predictions for all timesteps so we take them and create all substrings
    # that start at the first word and go through each sequence length
    # in this way we create sequences of length 1,2,....seq_len
    # note that we pad the strings at the front with <naw> tokens
    train_pred1, train_pred2 = [], []
    for i in range(seq_len):
        tmp1 = tf.concat([tf.zeros([BATCH_SIZE, seq_len-i-1, naw_r]), \
                          tf.ones([BATCH_SIZE, seq_len-i-1, 1]), \
                          tf.zeros([BATCH_SIZE, seq_len-i-1, wordmap_len - naw_r-1])], axis=2)
        tmp2 = tf.concat([tf.zeros([BATCH_SIZE, seq_len-i-1, naw_c]), \
                          tf.ones([BATCH_SIZE, seq_len-i-1, 1]), \
                          tf.zeros([BATCH_SIZE, seq_len-i-1, wordmap_len - naw_c-1])], axis=2)

        train_pred1.append(tf.concat([tmp1, gt[0][:, :i], row[:, i:i + 1, :]], axis=1))
        train_pred2.append(tf.concat([tmp2, gt[1][:, :i], col[:, i:i + 1, :]], axis=1))

    # realloc just takes the raw logits
    realloc_1 = tf.reshape(row, [BATCH_SIZE, seq_len, wordmap_len])
    realloc_2 = tf.reshape(col, [BATCH_SIZE, seq_len, wordmap_len])

    # reshape the lists to tensors and randomly select BATCH_SIZE strings
    train_pred1 = tf.reshape(train_pred1, [BATCH_SIZE*seq_len, seq_len, wordmap_len])
    train_pred2 = tf.reshape(train_pred2, [BATCH_SIZE*seq_len, seq_len, wordmap_len])
    #indices = tf.random_uniform([BATCH_SIZE], 0, BATCH_SIZE*seq_len, dtype=tf.int32)
    #train_pred1 = tf.gather(train_pred1, indices)
    #train_pred2 = tf.gather(train_pred2, indices)

    # realloc is supposed to the log softmax
    return [train_pred1, train_pred2], [tf.log(realloc_1), tf.log(realloc_2)]

def rnn_step_prediction(cells, wordmap_len, gt_sentence_input, num_neurons, \
                        seq_len, sm_bias, sm_weight, states, reuse=False):
    '''
    This function performs the RNN step for each timestep

    cells = the LightLSTM cells
    wordmap_len = ciel sqrt |V|
    gt_sentence_input = the sequence to predict off of
    num_neurons = number of neurons in the LSTM
    sm_bias = bias for softmax
    sm_weight = weights for softmax
    states = the initial states of the LSTMs
    reuse = if tf should reuse variables
    '''
    # ensure resue
    with tf.variable_scope("rnn", reuse=reuse):
        output = gt_sentence_input
        for l in range(FLAGS.GEN_GRU_LAYERS):
            # run the update for each layer of the rnn and update its state
            output, states[l] = tf.nn.dynamic_rnn(cells[l], output, \
                                                  dtype=tf.float32,
                                                  initial_state=states[l], \
                                                  scope="layer_%d" % (l + 1))

    # output is row and col stacked on top of each other
    row, col = tf.split(output, 2, 0)


    row = tf.nn.dropout(row, DROPOUT)
    col = tf.nn.dropout(col, DROPOUT)

    # reshape and perform softmax prediction on the outputs of the RNN
    row = tf.reshape(row, [-1, num_neurons])
    row = tf.nn.softmax(tf.matmul(row, sm_weight[0]) + sm_bias[0])
    row = tf.reshape(row, [BATCH_SIZE, -1, wordmap_len])
    col = tf.reshape(col, [-1, num_neurons])
    col = tf.nn.softmax(tf.matmul(col, sm_weight[1]) + sm_bias[1])
    col = tf.reshape(col, [BATCH_SIZE, -1, wordmap_len])

    return row, col, states


def get_inference_op(cells, word_input, embedding, seq_len, \
                     sm_bias, sm_weight, states, num_neurons, \
                     wordmap_len, reuse=False):
    '''
    This function gets the operation for inference ie testing

    cells = the LightLSTM cells
    word_input = the noise
    embedding = the matricies for the embeding layer
    seq_len = the length of sequences to produce
    sm_bias = bias to use for softmax
    sm-weight = weight to use for softmax
    states = initial states of the LSTM layers
    num_neurons = the size of the LSTMs
    wordmap_len = ciel sqrt |V|
    reuse = if tf should reuse vars
    '''

    inference_pred_r = []
    inference_pred_c = []

    # group for easy access
    embed_pred = [[word_input[0]], [word_input[1]]]

    for i in range(seq_len):
        # concat along the sequence dimension and then concat the row and col
        e_pred = tf.concat([tf.concat(embed_pred[0], 1), tf.concat(embed_pred[1], 1)], 0)

        # perform the rnn step prediction
        step_pred_r, step_pred_c, states = rnn_step_prediction(cells, wordmap_len, \
                                                                e_pred, num_neurons, seq_len,\
                                                                sm_bias, sm_weight, states, \
                                                                reuse=reuse)

        # get the best word, one-hot it add it to the inference list and add it embedded to the input to the rnn
        best_words_tensor_r = tf.argmax(step_pred_r, axis=2)
        best_words_one_hot_tensor_r = tf.one_hot(best_words_tensor_r, wordmap_len)
        best_word_r = best_words_one_hot_tensor_r[:, -1, :]
        inference_pred_r.append(tf.expand_dims(best_word_r, 1))
        embed_pred[0].append(tf.expand_dims(tf.matmul(best_word_r, embedding[0]), 1))

        best_words_tensor_c = tf.argmax(step_pred_c, axis=2)
        best_words_one_hot_tensor_c = tf.one_hot(best_words_tensor_c, wordmap_len)
        best_word_c = best_words_one_hot_tensor_c[:, -1, :]
        inference_pred_c.append(tf.expand_dims(best_word_c, 1))
        embed_pred[1].append(tf.expand_dims(tf.matmul(best_word_c, embedding[1]), 1))

        reuse = True

    # return the best words
    return [tf.concat(inference_pred_r, axis=1), tf.concat(inference_pred_c, axis=1)]

def get_noise():
    '''
    Create random noise according to a normal distribution
    '''
    noise_shape = [BATCH_SIZE*2, FLAGS.GEN_STATE_SIZE]
    return tf.random_normal(noise_shape, 0.0, FLAGS.NOISE_STDEV), noise_shape


def params_with_name(name):
    '''
    Used when creating the optimizer, this gets the parameters specific to the scope provided (name)
    '''
    return [p for p in tf.trainable_variables() if name in p.name]
