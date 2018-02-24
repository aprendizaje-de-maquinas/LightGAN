'''
This module implements the train loop for curriculum training
'''
import model_and_data_serialization
from config import *
from single_run import run
from summaries import log_run_settings

import os
from shutil import copyfile

# silence tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


create_logs_dir()
log_run_settings()

# setup to make sure that we know the size of the wordmap
#_, wordmap, inv_wordmap = model_and_data_serialization.load_dataset(seq_length=67)
wordmap = None
inv_wordmap = None
#os.remove( 'locations/word-32.locations' )

# curriculum training
stages = range(FLAGS.START_SEQ, FLAGS.END_SEQ)

# controls the number of meta iterations (ie the number of reallocs to perform per sequence_length)
num = 2

for i in range(len(stages)):
    # get the correct stage and prev_seq_len (for loading correct file)
    if FLAGS.TRAIN_FROM_CKPT:
        prev_seq_length = stages[i]
    else:
        prev_seq_length = stages[i-1] if i > 0 else 0
    seq_length = stages[i]


    # loop permeta iter
    for j in range(num):
        print("********************Training on Seq Len = %d, BATCH SIZE: %d, META ITER: %d********************"\
              % (seq_length, BATCH_SIZE, j))

        tf.reset_default_graph()
        iterations = FLAGS.ITERATIONS_PER_SEQ_LENGTH

        param = prev_seq_length if j == 0 else seq_length
        first = seq_length == stages[0] and not (FLAGS.TRAIN_FROM_CKPT) and j == 0

        # run one single meta iter
        run(iterations, seq_length, first, wordmap, inv_wordmap, param, j, num)

        # make sure to remove unnecessary files created
        if j != num -1:
            copyfile('locations/word-%d.locations' % (seq_length+1), \
                     'locations/word-%d.locations' % (seq_length))
            os.remove('locations/word-%d.locations' % (seq_length+1))
