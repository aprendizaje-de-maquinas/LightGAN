'''
This module contains the definition of the optimization ops, and the creation of the
optimizers
'''

#pylint: disable-msg=too-many-arguments
#pylint: disable-msg=line-too-long
#pylint: disable-msg=bad-continuation
#pylint: disable-msg=missing-docstring
#pylint: disable-msg=too-many-locals
#pylint: disable-msg=invalid-name

from math import ceil, sqrt

import tensorflow as tf
from config import BATCH_SIZE, LAMBDA
from model import generator, discriminator, params_with_name


def get_optimization_ops(disc_cost, gen_cost, global_step):
    '''
    Creates the optimization ops for the discriminator and generator

    disc_cost = the cost for the discriminator
    gen_cost = the cost for the generator
    global_step = the current step
    '''

    # get the correct paramaeters for G and D to ensure that the updates are applied to the correct params
    gen_params = params_with_name('Generator')
    disc_params = params_with_name('Discriminator')

    # a paper suggests using the square of the global norm to regularize the generator
    # this gives decreased perfromance so we have it noted in the comments
    scale = 0.0
    reg = 0
    #reg = tf.square(tf.global_norm(tf.gradients(disc_cost, disc_params)))
    #scale = 0.5

    # use Adam for the gen and disc training make sure to apply to the correct var_list
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, \
                                          beta1=0.9, \
                                          beta2=0.990).minimize(gen_cost+scale*reg, \
                                                              var_list=gen_params, \
                                                              global_step=global_step)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, \
                                           beta1=0.9, \
                                           beta2=0.999).minimize(disc_cost, \
                                                               var_list=disc_params)
    return disc_train_op, gen_train_op


def get_substrings_from_gt(real_inputs, seq_length, wordmap_len, naw_r, naw_c ):
    '''
    This function takes the gt and first creates BATCH_SIZE*seq_len substrings from the
    inputs and then randomly selects BATCH_SIZE versions from that.
    This allows to train on various different length strings at any point in the curriculum training

    real_inputs = the ground truth inputs
    seq_length = the maximum length of the substrings
    wordmap_len = ceil sqrt |V|
    naw_r = the row index of the <naw> token in the map
    naw_c = the col index of the <naw> token in the map
    '''
    train_pred_r = []
    train_pred_c = []

    # extract the substrings from the inputs and pad the beginings with the <naw> token
    for i in range(seq_length):

        tmp1 = tf.concat( [ tf.zeros( [BATCH_SIZE ,  seq_length-i-1 , naw_r ] ) ,\
                            tf.ones( [BATCH_SIZE , seq_length-i-1 , 1 ])  , \
                            tf.zeros( [BATCH_SIZE , seq_length-i-1 , wordmap_len - naw_r-1 ] ) ] , axis=2)
        tmp2 = tf.concat( [ tf.zeros( [BATCH_SIZE ,  seq_length-i-1 , naw_c ] ) ,\
                            tf.ones( [BATCH_SIZE , seq_length-i-1 , 1 ])  , \
                            tf.zeros( [BATCH_SIZE , seq_length-i-1 , wordmap_len - naw_c-1 ] ) ] , axis=2)
        train_pred_r.append( tf.concat([tmp1, real_inputs[0][:, :i + 1]], axis=1))
        train_pred_c.append( tf.concat([tmp2, real_inputs[1][:, :i + 1]], axis=1))

    # reshape the lists into tensors and randomly select BATCH_SIZE inputs from it
    all_sub_strings_r = tf.reshape(train_pred_r, [BATCH_SIZE * seq_length, seq_length, wordmap_len])
    all_sub_strings_c = tf.reshape(train_pred_c, [BATCH_SIZE * seq_length, seq_length, wordmap_len])
    #indices = tf.random_uniform([BATCH_SIZE], 1, all_sub_strings_r.get_shape()[0], dtype=tf.int32)
    #all_sub_strings = [ tf.gather(all_sub_strings_r, indices)[:BATCH_SIZE] , tf.gather(all_sub_strings_c, indices)[:BATCH_SIZE] ]
    all_sub_strings = [ all_sub_strings_r, all_sub_strings_c]
    return all_sub_strings


def define_objective(wordmap, real_inputs_discrete, seq_length , naw_r , naw_c ):
    '''
    This function creates the network and returns the ops for training, inference, and summaries

    wordmap = the map from words to lorations
    real_inputs_discrete = the gt input placeholders
    seq_length = the sequence lengtht we are training on
    naw_r = the row index of the <naw> token in the wordmap
    naw_c = the col index of the <naw> token in the wordmap
    '''

    # get the correct dim for the softmax
    bound = ceil( sqrt( len(wordmap) ) )

    # one hot the inputs
    real_inputs_r = tf.one_hot(real_inputs_discrete[0], bound )
    real_inputs_c = tf.one_hot(real_inputs_discrete[1], bound )
    real_inputs = [ real_inputs_r , real_inputs_c ]

    # rename the gen and dsic
    Generator = generator
    Discriminator = discriminator

    # make the gnerator
    train_pred, inference_op, realloc_op = Generator(BATCH_SIZE, bound , naw_r , naw_c , seq_len=seq_length, gt=real_inputs)

    # extract the gt substrings for training the discriminator on real data
    real_inputs_substrings = get_substrings_from_gt(real_inputs, seq_length, bound , naw_r , naw_c)

    # make the discriminator for the real batch
    disc_real = Discriminator(real_inputs_substrings, bound, seq_length, reuse=False)

    # make the discriminator for the fake batch
    # note that reuse must be true
    disc_fake = Discriminator(train_pred, bound , seq_length, reuse=True)

    # this can be used to create another discriminator for the inference op
    # ie if you wanted to score the inferenced ops
    disc_on_inference = None

    # create ops for the accuracy of the discriminator on the real and fake data
    d_cost_fake = tf.reduce_mean(tf.cast(tf.less(disc_fake, 0.0), tf.float32))
    d_cost_real = tf.reduce_mean(tf.cast(tf.greater(disc_real, 0.0), tf.float32))

    # create the costs to be optimized
    disc_cost, gen_cost = loss_d_g(disc_fake, disc_real, train_pred, inference_op, real_inputs_substrings, wordmap, seq_length, Discriminator)


    return disc_cost, gen_cost, train_pred, disc_fake, disc_real, disc_on_inference, inference_op, realloc_op, d_cost_fake, d_cost_real


def loss_d_g(disc_fake, disc_real, fake_inputs, inf_outputs, real_inputs, wordmap, seq_length, Discriminator):
    '''
    This function creates the costs for the discriminator and generator according to the WGAN-GP model
    '''

    # this is the WGAN cost
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # this is the creation of the GP for the discriminator
    alpha1 = tf.random_uniform(shape=[tf.shape(real_inputs[0])[0], 1, 1], minval=0., maxval=1.)
    alpha2 = tf.random_uniform(shape=[tf.shape(real_inputs[1])[0], 1, 1], minval=0., maxval=1.)

    differences1 = fake_inputs[0] - real_inputs[0]
    differences2 = fake_inputs[1] - real_inputs[1]

    bound = ceil(sqrt(len(wordmap)))

    interpolates = [real_inputs[0] + alpha1*differences1, real_inputs[1] + alpha2*differences2]

    gradients1 = tf.gradients(Discriminator(interpolates, bound, seq_length, reuse=True), [interpolates[0]])[0]
    gradients2 = tf.gradients(Discriminator(interpolates, bound, seq_length, reuse=True), [interpolates[1]])[0]

    slopes1 = tf.sqrt(tf.reduce_sum(tf.square(gradients1), reduction_indices=[1, 2]))
    slopes2 = tf.sqrt(tf.reduce_sum(tf.square(gradients2), reduction_indices=[1, 2]))

    gradient_penalty1 = tf.reduce_mean((slopes1 - 1.) ** 2)
    gradient_penalty2 = tf.reduce_mean((slopes2 - 1.) ** 2)

    # average the gradient penalties
    disc_cost += LAMBDA/2 * gradient_penalty1 + LAMBDA/2 * gradient_penalty2

    return disc_cost, gen_cost



if __name__ == '__main__':


    b_size = 2
    seq_len = 2
    vocab = 5

    gt_r = 3
    gt_c = 2
    tmp  = tf.concat([tf.zeros([b_size, seq_len, gt_r]), \
                        tf.ones([b_size, seq_len, 1]), tf.zeros([b_size, seq_len, vocab - gt_r-1])], axis=2)


    with tf.Session() as sess:

        t = sess.run(tmp)
        print(t)
        print(t.shape)
