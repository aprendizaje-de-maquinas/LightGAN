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

from config import BATCH_SIZE, PRETRAIN, FLAGS
from model_and_data_serialization import generate_argmax_samples_and_gt_samples, inf_train_gen, inf_realloc_gen, \
    perf_reallocate, load_dataset, get_internal_checkpoint_dir, optimistic_restore, decode_indices_to_string
from objective import get_optimization_ops, define_objective
from summaries import define_summaries, log_samples
from model import FLAGS, restore_config, CRITIC_ITERS, GEN_ITERS
import numpy as np
import itertools

sys.path.append(os.getcwd())

def test(beam_length, seq_len_to_test, batch_size, n):

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    seq_length = seq_len_to_test

    _, wordmap, inv_wordmap = load_dataset(seq_length=0, n_examples=FLAGS.MAX_N_EXAMPLES)

    real_inputs_discrete = [tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length]), \
                              tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])]

    # global step
    global_step = tf.Variable(0, trainable=False, name='global-step')

    # indec of <naw> in the map
    naw_r, naw_c = wordmap['<naw>'][0], wordmap['<naw>'][1]

    session = tf.Session(config=config)

    _, _, ops, _, _ = define_objective(session, wordmap, real_inputs_discrete, seq_length, naw_r, naw_c, None)

    fake, inference_op, _ = ops

    with session.as_default():

        optimistic_restore(session, tf.train.latest_checkpoint('pretrain/seq-%d' % n, 'checkpoint'))
        restore_config.set_restore_dir(load_from_curr_session=True)

        #inference_op[0] = tf.reshape(inference_op[0], [-1, len(inv_wordmap)])
        #inference_op[1] = tf.reshape(inference_op[1], [-1, len(inv_wordmap)])

        logits = []
        for b in range(BATCH_SIZE):
            buff = []
            for t in range(seq_len_to_test):

                #tmp_col = tf.reshape(tf.tile( tf.reshape(fake[1][b][t], [-1]), [len(inv_wordmap)]), \
                #                    [len(inv_wordmap), len(inv_wordmap)])
                tmp_col = tf.reshape(tf.tile( tf.reshape(inference_op[1][b][t], [-1]), [len(inv_wordmap)]), \
                                     [len(inv_wordmap), len(inv_wordmap)])
                tmp_col = tf.nn.softmax(tmp_col)

                #tmp_row = tf.reshape(tf.exp(fake[0][b][t]), [-1,1])
                tmp_row = tf.reshape(tf.nn.softmax(inference_op[0][b][t]), [-1,1])
                #tmp_row = tf.reshape(inference_op[0][b][t], [-1,1])

                tmp = tmp_col + tmp_row
                #tmp = tf.matmul(tf.reshape(inference_op[0][b][t], [-1,1]), tf.reshape(inference_op[1][b][t], [1,-1]))
                tmp = tf.reshape(tmp, [1,-1])
                #tmp = tf.concat([tmp, tf.zeros([1,1], dtype=tf.float32)], -1)

                buff.append(tmp)
            logits.append(tf.reshape(buff, [-1, len(inv_wordmap)**2]))

        logits = tf.reshape(logits, [BATCH_SIZE, seq_len_to_test, len(inv_wordmap)**2])
        logits = tf.transpose(logits, [1,0,2])
        #logits = tf.nn.softmax(logits)
        #_logits = tf.exp(logits)
        #_logits = tf.nn.softmax(logits)
        _logits = logits
        #_logits = tf.log(logits)
        #print(logits)

        length = tf.multiply(tf.ones([BATCH_SIZE], dtype=tf.int32),tf.constant(seq_len_to_test,dtype=tf.int32))

        #length = tf.multiply(tf.ones([BATCH_SIZE], dtype=tf.int32),tf.constant(10,dtype=tf.int32))
        print(session.run(length))

        #res = tf.nn.ctc_beam_search_decoder(_logits, length, beam_width=10, merge_repeated=False)
        res = tf.nn.ctc_greedy_decoder(_logits, length, merge_repeated=False)

        paths = tf.sparse_tensor_to_dense(res[0][0], default_value=-1)   # Shape: [batch_size, max_sequence_len]
        for batch in range(BATCH_SIZE):


            infer, logs, logit, i_op = session.run([paths, res[1], _logits, inference_op[0]])

            #for x in range(1):
            #   for y in range(20):
            #       for z in range(62501):
            #           assert (logit[x][y][z] != 0)

            print(logit)
            for i in range(len(infer)):
                for j in range(len(infer[0])):
                    ind = infer[i][j]
                    if infer[i][j] == -1:
                        break

                    row = ind // len(inv_wordmap)
                    col = ind % len(inv_wordmap)
                    print( inv_wordmap[row][col] , end=' ')

                print('')

            #print(infer)
            #infer_r, infer_c = infer



    session.close()



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
    # make sure tf does not take up all the GPU mem (model size is not that large so there is unlikely to be fragmentation problems)
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    # first one so copy from the initial location file
    if is_first and os.path.isfile('locations/word-0.locations') or FLAGS.WORDVECS is not None:
        copyfile('locations/word-0.locations', 'locations/word-%d.locations' % seq_length)


    # load the lines from the dataset along with the current wordmap and inv_wordmap
    lines, wordmap, inv_wordmap = load_dataset(seq_length=seq_length, n_examples=FLAGS.MAX_N_EXAMPLES)

    if not os.path.isfile('locations/word-0.locations'):
        copyfile('locations/word-%d.locations' % seq_length, 'locations/word-0.locations')


    # placeholders for the input from the datset
    real_inputs_discrete = [tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length]), \
                              tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])]

    # global step
    global_step = tf.Variable(iterations, trainable=False, name='global-step')

    # indec of <naw> in the map
    naw_r, naw_c = wordmap['<naw>'][0], wordmap['<naw>'][1]

    session = tf.Session(config=config)

    # start the generator to get minibatches from the dataset
    gen = inf_train_gen(lines, wordmap, seq_length)

    # define the network
    #disc_cost, gen_cost, fake_inputs, disc_fake, disc_real, disc_on_inference, inference_op, realloc_op, d_cost_fake, d_cost_real = define_objective(session, wordmap, real_inputs_discrete, seq_length, naw_r, naw_c, gen)
    # get the summaries, optimizers, and saver

    optim_costs, discs, ops, log_costs, embeddings = define_objective(session, wordmap, real_inputs_discrete, seq_length, naw_r, naw_c, gen)

    #embed_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    #r_embedding = embed_config.embeddings.add()
    #r_embedding.tensor_name = embeddings.name
    #r_embedding.metadata_path = '../../loations/metadata.tsv'

    disc_cost, gen_cost = optim_costs
    disc_fake, disc_real, disc_on_inference = discs
    fake_inputs, inference_op, realloc_op = ops
    d_cost_fake, d_cost_real = log_costs

    merged, train_writer = define_summaries(d_cost_real, d_cost_fake, gen_cost, disc_cost)
    disc_train_op, gen_train_op = get_optimization_ops(disc_cost, gen_cost, global_step)
    saver = tf.train.Saver(tf.trainable_variables())


    with session.as_default():
        # write the graph and init vars
        train_writer.add_graph(session.graph)
        session.run(tf.global_variables_initializer())

        # if not is_first, we need to load from ckpt
        if is_first and PRETRAIN:
            optimistic_restore(session, tf.train.latest_checkpoint('pretrain', 'checkpoint'))
            restore_config.set_restore_dir(load_from_curr_session=True)
        elif not is_first:
            internal_checkpoint_dir = get_internal_checkpoint_dir(prev_seq_length)
            optimistic_restore(session, tf.train.latest_checkpoint(internal_checkpoint_dir, "checkpoint"))
            restore_config.set_restore_dir(load_from_curr_session=True)



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

                    iii = meta_iter*iterations + iteration + ((seq_length-1)*iterations*max_meta_iter)

                    train_writer.add_summary(summary_str, \
                                             global_step=iii)

                    # generate and log ouput from training
                    fake_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_wordmap,
                                                                                          fake_inputs,
                                                                                          disc_fake,
                                                                                          gen,
                                                                                          real_inputs_discrete,
                                                                                          feed_gt=True, method='argmax')
                    log_samples(fake_samples, fake_scores, iii, seq_length, "gen-w-gt")

                    # generate and log output from inference
                    test_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_wordmap,
                                                                                        inference_op,
                                                                                          disc_on_inference,
                                                                                          gen,
                                                                                          real_inputs_discrete,
                                                                                          feed_gt=False, method='argmax')
                    log_samples(test_samples, fake_scores, iii, seq_length, "gen-no-gt")


        #*********************************************************************************
        #if seq_length >= 40:
        # copy current location file to next sequence_length as we are going to gen sequences of length seq_len +1 for realloc
        if FLAGS.WORDVECS is None:

            copyfile('locations/word-%d.locations' % (seq_length), 'locations/word-%d.locations' % (seq_length+1))
            if seq_length >= 0:
                # get the lines, note that wordmap and inv_wormap stay the same
                lines, _, _ = load_dataset(seq_length=48, n_examples=FLAGS.MAX_N_EXAMPLES,\
                                           no_write=True)

                # start generator and perform the reallocation
                gen = inf_realloc_gen(lines, wordmap, seq_length+1)
                perf_reallocate(int(1000000), session, inv_wordmap, realloc_op, \
                                gen, seq_length, real_inputs_discrete, naw_r, naw_c)

                #  realloc creates the new location file in seq_length +1 so we move it back if we are not at the last meta_iter
                if meta_iter != max_meta_iter -1:
                    copyfile('locations/word-%d.locations' % (seq_length+1), 'locations/word-%d.locations' % (seq_length))
                    #os.remove('locations/word-%d.locations' % (seq_length+1))
                    os.remove('locations/word-%d.locations.string' % (seq_length+1))


        # save the ckpt and close the session because we need to reset the graph
        saver.save(session, get_internal_checkpoint_dir(seq_length) + "/ckp")
        session.close()
