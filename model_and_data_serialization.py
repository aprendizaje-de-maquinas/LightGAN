'''
This module implements the data io functions
'''



import os
import tensorflow as tf

from config import DATA_DIR, BATCH_SIZE, restore_config

import numpy as np
import collections
from math import ceil , sqrt

from ctypes import c_double, create_string_buffer
import ctypes
from random import shuffle
from tqdm import tqdm

def _load_dataset(max_length, max_n_examples, max_vocab_size=1000000, data_dir='data/', no_write=False):
    '''
    Internal function for loading dataset returns the lines of the training file, the map from words to loactions
    and the map from locations to words

    max_length = the length of the substrings
    nax_n_examples = limit to the number of lines from the training file
    max_voacb_size = maximum size of the vocab
    data_dir = the directory the training data is in
    no_write = if we should write files
    '''
    lines = []
    path = data_dir+'train.txt'

    with open(path, 'r') as f:
        for line in f:
            line = line.split()[:max_length]
            lines.append(line)
            if len(lines) > max_n_examples:     break
    np.random.shuffle(lines)

    wordmap = {}
    inv_wordmap = []

    i = 0
    buff = []

    # if vocab file does not exist, then make it
    if not os.path.isfile('locations/vocab.txt'):
        counts = collections.Counter(word for line in lines for word in line)
        counts.update( { '<naw>' : 0 } )
        counts = [word for word,count in counts.most_common(max_vocab_size)]
        shuffle(counts)
        print ('Writing Vocab File')
        with open('locations/vocab.txt','w') as outfile:
            for word in counts:
                outfile.write(word + '\n')
    counts = []
    with open( 'locations/vocab.txt' , 'r' ) as infile:
        counts =  [word[:-1] for word in infile]

    # if locations file does not exist, make it and read from it
    # else, just read the location file
    if os.path.isfile('locations/word-%d.locations' %(max_length)):
        with open('locations/word-%d.locations' %(max_length), 'r') as infile:
            for r,line in enumerate(infile):
                tmp = []
                line = line.split()
                for c,word in enumerate(line):
                    if word == '-1':
                        word = '[%d %d]' % (r, c)
                        counts.append(word)
                        word = len(counts) -1

                    tmp.append(counts[int(word)])
                    wordmap[counts[int(word)]] = [r, c]
                inv_wordmap.append(tmp)
    else:
        bound = ceil(sqrt(len(counts)))

        for word in counts:
            if word not in wordmap:
                wordmap[word] = [i // bound, i % bound]
                buff.append(word)

                if len(buff) == bound:
                    inv_wordmap.append(buff)
                    buff = []
                i+= 1

        s = ('[ %d %d ]' % (i // bound, i % bound))
        while(len(inv_wordmap)) != bound:
            while len(buff) < bound:
                wordmap[s] = [i // bound, i % bound]
                buff.append(s)
                i+=1
                s = ('[ %d %d ]' % (i // bound, i % bound))

            inv_wordmap.append(buff)
            buff = []
        with open('locations/word-%d.locations' % (max_length), 'w') as outfile:
            for r in range(len(inv_wordmap)):
                for c in range( len( inv_wordmap ) ):
                    if inv_wordmap[r][c] == ('[ %d %d ]' % (r, c)):
                        outfile.write('-1\t')
                    else:
                        outfile.write(str(counts.index(inv_wordmap[r][c])) + '\t')
                outfile.write('\n')

    # filter the lines if vocab_size exceeds the max_vocab_size
    filtered_lines = []
    for line in lines:
        filtered_line = []
        for word in line:
            if word in wordmap:
                filtered_line.append(word)
            else:
                filtered_line.append('unk')
        filtered_lines.append(filtered_line)

    return filtered_lines, wordmap, inv_wordmap


def load_dataset(seq_length=32, n_examples=10000000, no_write=False):
    '''
    Wrapper for the internal loca dataset function

    seq_length = the seqeunce length to load
    n_examples = the max mumber of examples
    no_write = if we write files
    '''
    lines, word_map, inv_wordmap = _load_dataset(max_length=seq_length, \
                                                 max_n_examples=n_examples, \
                                                 data_dir=DATA_DIR, \
                                                 no_write=no_write)

    return lines, word_map, inv_wordmap

def get_internal_checkpoint_dir(seq_length):
    '''
    Get the location of the checkpoint directory for the given sequence length
    '''
    internal_checkpoint_dir = os.path.join(restore_config.get_restore_dir(), "seq-%d" % seq_length)

    # if it does not exist, make it
    if not os.path.isdir(internal_checkpoint_dir):
        os.makedirs(internal_checkpoint_dir)
    return internal_checkpoint_dir


def generate_argmax_samples_and_gt_samples(session, inv_wordmap, fake_inputs, disc_fake, \
                                           gen, real_inputs_discrete, feed_gt=True, method='argmax'):

    '''
    Generate samples for logging

    session = tf session so that we can run things
    inv_wordmap = map from locations to words
    fake_inputs = op to create the samples
    disc_fake = the dicriminator
    gen = the gt generator
    real_inputs_discrete = placeholders for the gt inputs
    feed_gt = flag for if this is for train or inference
    method = method for chosing the shamples
    '''
    scores = []
    samples = []
    samples_probs = []

    # generate 10 batches and add them to lists
    for _ in range( 10 ):
        argmax_samples, real_samples, samples_scores = generate_samples(session, inv_wordmap, \
                                                                         fake_inputs, \
                                                                         disc_fake, gen, \
                                                                         real_inputs_discrete, \
                                                                         feed_gt=feed_gt, method=method)

        #if samples_scores != None:
        scores.extend(samples_scores)

        samples.extend(argmax_samples)
        samples_probs.extend(real_samples)

    return samples, samples_probs, scores

def generate_samples(session, inv_wordmap, fake_inputs, disc_fake, gen,\
                     real_inputs_discrete, feed_gt=True, method='argmax'):
    '''
    This function is what runs the session to generate samples

    session = tf session so that we can evaluate tensors
    inv_wordmap = map from row col to word
    fake_inputs = the op for creating the samples
    disc_fake = the discriminator to evaluate the samples
    gen = the data generator
    real_inputs_discrete = the palceholders for the real inputs
    feed_gt = falg for if this is inference or not
    method = way to create samples
    '''

    # setup the feed
    if feed_gt:
        _data = next(gen)
        f_dict = {real_inputs_discrete[0]:_data[0], real_inputs_discrete[1]:_data[1]}
    else:
        f_dict = {}

    # make the correct runs
    if disc_fake != None:
        fake_samples, fake_scores = session.run([fake_inputs, disc_fake], feed_dict=f_dict)
    else:
        fake_samples = session.run(fake_inputs, feed_dict=f_dict)
        fake_scores = []

    # make sure the dim is only 1D
    fake_scores = np.squeeze(fake_scores)

    batch, seq, dim = fake_samples[0].shape

    # argmax or sampled
    if method == 'argmax':
        fake_samples_1 = np.argmax(fake_samples[0], axis=2)
        fake_samples_2 = np.argmax(fake_samples[1], axis=2)
    else:
        fake_samples_1, fake_samples_2 = [], []
        for x in range(batch):
            tmp1 = []
            tmp2 = []
            for y in range(seq):
                tmp1.append( np.random.choice(dim, p=fake_samples[0][x, y, :]))
                tmp2.append( np.random.choice(dim, p=fake_samples[1][x, y, :]))
            fake_samples_1.append(tmp1)
            fake_samples_2.append(tmp2)

    fake_samples = [fake_samples_1, fake_samples_2]

    # decode the samples
    decoded_samples = decode_indices_to_string(fake_samples, inv_wordmap)
    return decoded_samples, fake_samples, fake_scores

def decode_indices_to_string(samples, inv_wordmap):
    '''
    Converts the row and col indices into words

    samples = the rows and cols of the samples
    inv_wordmpa = the map from indices to words
    '''
    decoded_samples = []
    sample_r, sample_c = samples[0], samples[1]


    # iterate over the samples and convert them to strings
    for i in range(len(sample_r)):
        decoded =[]
        for j in range(len(sample_r[i])):
            decoded.append(inv_wordmap[sample_r[i][j]][sample_c[i][j]])
        decoded_samples.append(decoded)

    return decoded_samples

def inf_train_gen(lines, wordmap, seq_len):
    '''
    A data generator that just does not stop

    lines = the lines of the dataset file
    wordmap = the map from words to roc,col
    seq_len = the length of the sequences to generate
    '''
    while True:
        # once we run out of lines, shuffle the lines so that we train in a different order
        np.random.shuffle(lines)

        # iterate over all the lines
        for i in range(0, len(lines) - BATCH_SIZE+1, BATCH_SIZE):

            row = []
            col = []

            # iterate over the batch and append the encoded words to the lists
            for l in lines[i : i+BATCH_SIZE]:
                tmp_r, tmp_c =[],[]
                while len(l) != seq_len:
                    l = lines[np.random.randint(0, len(lines), dtype=np.int32)]
                for c in l:
                    tmp_r.append(wordmap[c][0])
                    tmp_c.append(wordmap[c][1])

                row.append(tmp_r)
                col.append(tmp_c)

            # convert to array for tf
            row, col = np.array(row, dtype='int32'), np.array(col, dtype='int32')
            yield [row, col]

    return


def optimistic_restore(session, save_file):
    '''
    This function restores the session and ensures that none of the dimensions have changed
    Can be seen as a safe restore to session from save_file
    '''
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()

    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print("Not loading: %s." % saved_var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def _allocate_table(row, col, vocab_size, vocab_sqrt, seq_len):
    '''
    This function is the interface with compiled C++ program that actually performs the reallocation

    The compiled C++ code is used from Microsoft Research under a Microsoft liscense
    see: https://github.com/Microsoft/CNTK/blob/master/LICENSE.md
    '''
    dll_name = 'libpyreallocate.so'
    path_dir = os.path.split(os.path.realpath(__file__))[0]
    dll_path = os.path.join(path_dir, dll_name)
    lib = ctypes.cdll.LoadLibrary(dll_path)
    row = np.concatenate(row)
    col = np.concatenate(col)

    row_size = len(row)
    row = (c_double * row_size)(*row)
    col_size = len(col)
    col = (c_double * col_size)(*col)
    word_path = create_string_buffer('locations/vocab.txt'.encode('utf-8'))

    tmp = 'locations/word-%d.locations' % (seq_len+1)

    save_location_path = create_string_buffer(tmp.encode('utf-8'))
    lib.allocate_table(row, col, vocab_size, vocab_sqrt, save_location_path , word_path)

    return

def perf_reallocate(iterations, session, inv_wordmap, realloc_op, gen, \
                    seq_len, real_inputs_discrete, naw_r, naw_c):
    '''
    Sets up the arrays for use with _allocate_table

    iterations = the number of iterations to run to approximate the reallocation
    session = used to eval tensors
    iniv_wordmap = the map from index to word
    realloc_op = the log of the softmax logits at each step
    gen = the data generator
    seq_len = the length of the sequences to create
    real_inputs_discrete = the placeholders for the real inputs
    naw_r = the row index of the <naw> token
    naw_c = the col index of the <naw> token
    '''

    counts = {}
    with open('locations/vocab.txt' , 'r') as infile:
        tmp = [word[:-1] for word in infile]
        counts = dict(zip(tmp, range(len(tmp))))

    # allocate the space for the loss vectors
    row_loss_vector = np.zeros((len(counts), len(inv_wordmap)))
    col_loss_vector = np.zeros((len(counts), len(inv_wordmap)))

    print('\t Calculate Loss Vector')
    with tqdm(total=iterations, ncols=150) as pbar:

        for mb_iter in range(iterations):

            # set up the feed
            _data = next(gen)
            f_dict= {real_inputs_discrete[0]:_data[0][:, :seq_len], real_inputs_discrete[1]:_data[1][:, :seq_len]}

            # get the log softmax logits
            row_prob, col_prob = session.run(realloc_op, feed_dict= f_dict)

            label1, label2 = _data[0][:, :], _data[1][:, :]

            for i in range(BATCH_SIZE):
                for j in range(seq_len):
                    row_word = counts[inv_wordmap[label1[i][j]][label2[i][j]]]
                    col_word = counts[inv_wordmap[label1[i][j+1]][label2[i][j+1]]]
                    row_loss_vector[row_word] -= row_prob[i][j]
                    col_loss_vector[col_word] -= col_prob[i][j]


            pbar.update( 1 )

    # run the allocation
    _allocate_table(row_loss_vector, col_loss_vector, len(counts), len(inv_wordmap ), seq_len)
    return
