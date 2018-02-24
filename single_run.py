'''
Runs a single round  of curriculum training
'''
#pylint: disable-msg=too-many-arguments
#pylint: disable-msg=line-too-long
#pylint: disable-msg=bad-continuation
#pylint: disable-msg=missing-docstring
#pylint: disable-msg=too-many-locals

import sys
from shutil import copyfile
import os

from tqdm import tqdm
import tensorflow as tf

from config import BATCH_SIZE
from model_and_data_serialization import generate_argmax_samples_and_gt_samples, inf_train_gen, \
    perf_reallocate, load_dataset, get_internal_checkpoint_dir, optimistic_restore
from objective import get_optimization_ops, define_objective
from summaries import define_summaries, log_samples
from model import FLAGS, restore_config, CRITIC_ITERS, GEN_ITERS
import numpy as np

sys.path.append(os.getcwd())


def run(iterations, seq_length, is_first, wordmap, inv_wordmap, prev_seq_length, meta_iter, max_meta_iter):
    '''
    Performs a single run of a single meta iter of curriculum training
    also performs reallocate at the end

    iterations = the number of minibatches
    seq_length = the length of the sequences to create
    is_first = if we need to load from ckpt
    wordmap = for getting the naw from
    inv_wordmap = for decoding the samples
    prev_seq_len = for chosing the ckpt to load from
    meta_iter = the round of the seq_length training
    max_meta_iter = used to make sure the tensorboard graphs log using the correct global step
    '''

    # first one so copy from the initial location file
    if seq_length == 1 and meta_iter == 0:
        copyfile('locations/word-0.locations', 'locations/word-1.locations')


    # load the lines from the dataset along with the current wordmap and inv_wordmap
    lines, wordmap, inv_wordmap = load_dataset(seq_length=seq_length, n_examples=FLAGS.MAX_N_EXAMPLES)

    # placeholders for the input from the datset
    real_inputs_discrete = [tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length]), \
                              tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])]

    # global step
    global_step = tf.Variable((seq_length-1)*iterations, trainable=False, name='global-step')

    # indec of <naw> in the map
    naw_r, naw_c = wordmap['<naw>'][0], wordmap['<naw>'][1]

    # define the network
    disc_cost, gen_cost, fake_inputs, disc_fake, disc_real, disc_on_inference, inference_op, realloc_op, d_cost_fake, d_cost_real = define_objective(wordmap,
                                                                                                            real_inputs_discrete,
                                                                                                            seq_length, \
                                                                                                            naw_r, naw_c)
    # get the summaries, optimizers, and saver
    merged, train_writer = define_summaries(d_cost_real, d_cost_fake, gen_cost, disc_cost)
    disc_train_op, gen_train_op = get_optimization_ops(disc_cost, gen_cost, global_step)
    saver = tf.train.Saver(tf.trainable_variables())

    # make sure tf does not take up all the GPU mem (model size is not that large so there is unlikely to be fragmentation problems)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        # write the graph and init vars
        train_writer.add_graph(session.graph)
        session.run(tf.global_variables_initializer())

        # if not is_first, we need to load from ckpt
        if not is_first:
            internal_checkpoint_dir = get_internal_checkpoint_dir(prev_seq_length)
            optimistic_restore(session, tf.train.latest_checkpoint(internal_checkpoint_dir, "checkpoint"))
            restore_config.set_restore_dir(load_from_curr_session=True)

        # start the generator to get minibatches from the dataset
        gen = inf_train_gen(lines, wordmap, seq_length)

        # cool progress bar
        with tqdm(total=iterations, ncols=150) as pbar:

            # train loop
            for iteration in range(iterations):
                # Train critic first
                for _ in range(CRITIC_ITERS):
                    _data = next(gen)
                    _disc_cost, _, _ = session.run([disc_cost, disc_train_op, disc_real], \
                                                   feed_dict={real_inputs_discrete[0]: _data[0], \
                                                              real_inputs_discrete[1]: _data[1]})

                # Train generator
                for _ in range(GEN_ITERS):
                    _data = next(gen)
                    _, _g_cost = session.run([gen_train_op, gen_cost], feed_dict={real_inputs_discrete[0]:_data[0], \
                                                                                  real_inputs_discrete[1]: _data[1]})


                # update progress bat with costs and inc its counter by 1
                pbar.set_description("disc cost %f \t gen cost %f \t"% (_disc_cost, _g_cost))
                pbar.update(1)

                # write sumamries
                if iteration % 100 == 99:
                    _data = next(gen)
                    summary_str = session.run(merged, feed_dict={real_inputs_discrete[0]:_data[0], \
                                                                 real_inputs_discrete[1]:_data[1]})

                    train_writer.add_summary(summary_str, \
                                             global_step=(seq_length-1)*iterations*max_meta_iter + \
                                             meta_iter*iterations + iteration)

                    # generate and log ouput from training
                    fake_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_wordmap,
                                                                                          fake_inputs,
                                                                                          disc_fake,
                                                                                          gen,
                                                                                          real_inputs_discrete,
                                                                                          feed_gt=True, method='argmax')
                    log_samples(fake_samples, fake_scores, iteration, seq_length, "gen-w-gt")

                    # generate and log output from inference
                    test_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_wordmap,
                                                                                        inference_op,
                                                                                          disc_on_inference,
                                                                                          gen,
                                                                                          real_inputs_discrete,
                                                                                          feed_gt=False, method='argmax')
                    log_samples(test_samples, fake_scores, iteration, seq_length, "gen-no-gt")


        #*********************************************************************************
        # copy current location file to next sequence_length as we are going to gen sequences of length seq_len +1 for realloc
        copyfile('locations/word-%d.locations' % (seq_length), 'locations/word-%d.locations' % (seq_length+1))

        # get the lines, note that wordmap and inv_wormap stay the same
        lines, _, _ = load_dataset(seq_length=seq_length+1, n_examples=FLAGS.MAX_N_EXAMPLES,\
                                   no_write=True)

        # start generator and perform the reallocation
        gen = inf_train_gen(lines, wordmap, seq_length+1)
        perf_reallocate(iterations*10, session, inv_wordmap, realloc_op, \
                        gen, seq_length, real_inputs_discrete, naw_r, naw_c)

        #  realloc creates the new location file in seq_length +1 so we move it back if we are not at the last meta_iter
        if meta_iter != max_meta_iter -1:
            copyfile('locations/word-%d.locations' % (seq_length+1), 'locations/word-%d.locations' % (seq_length))
            #os.remove('locations/word-%d.locations' % (seq_length+1))
            os.remove('locations/word-%d.locations.string' % (seq_length+1))


        # save the ckpt and close the session because we need to reset the graph
        saver.save(session, get_internal_checkpoint_dir(seq_length) + "/ckp")
        session.close()
