'''
Config for the net. Sets up all the parameters
'''
import os
import time

import tensorflow as tf

flags = tf.app.flags


flags.DEFINE_string('LOGS_DIR', './logs/', '')
flags.DEFINE_string('DATA_DIR', './data/', "")
flags.DEFINE_string('CKPT_PATH', "./ckpt/", "")
flags.DEFINE_integer('BATCH_SIZE', 20, '')
flags.DEFINE_integer('CRITIC_ITERS', 5, '')
flags.DEFINE_integer('LAMBDA', 10, '')
flags.DEFINE_integer('MAX_N_EXAMPLES', 10000000, '')
flags.DEFINE_integer('GEN_ITERS', 1, '')
flags.DEFINE_integer('ITERATIONS_PER_SEQ_LENGTH', 15000, '')
flags.DEFINE_float('NOISE_STDEV', 10.0, '')
flags.DEFINE_integer('DISC_STATE_SIZE', 512, '')
flags.DEFINE_integer('GEN_STATE_SIZE', 512, '')
flags.DEFINE_integer('GEN_GRU_LAYERS', 2, '')
flags.DEFINE_integer('DISC_GRU_LAYERS', 2, '')
flags.DEFINE_integer('START_SEQ', 1 , '')
flags.DEFINE_integer('END_SEQ', 48, '')
flags.DEFINE_integer('SAVE_CHECKPOINTS_EVERY', 20000, '')
flags.DEFINE_boolean('TRAIN_FROM_CKPT', False, '')
flags.DEFINE_boolean('USE_PRETRAIN', False, '')
flags.DEFINE_boolean('TESTING', True, '')
flags.DEFINE_string('WORDVECS', 'pretrain/word_vectors', '')


FLAGS = flags.FLAGS

LOGS_DIR = os.path.join(FLAGS.LOGS_DIR, "%s-" % (time.time()))


class RestoreConfig():
    '''
    Info for restoring
    '''
    def __init__(self):
        if FLAGS.TRAIN_FROM_CKPT:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=False)
        else:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=True)

    def set_restore_dir(self, load_from_curr_session=True):
        if load_from_curr_session:
            restore_dir = os.path.join(LOGS_DIR, 'checkpoint')
        else:
            restore_dir = FLAGS.CKPT_PATH
        return restore_dir

    def get_restore_dir(self):
        return self.restore_dir

def create_logs_dir():
    os.makedirs(LOGS_DIR)

restore_config = RestoreConfig()

DATA_DIR = FLAGS.DATA_DIR
BATCH_SIZE = FLAGS.BATCH_SIZE
CRITIC_ITERS = FLAGS.CRITIC_ITERS
LAMBDA = FLAGS.LAMBDA
MAX_N_EXAMPLES = FLAGS.MAX_N_EXAMPLES
CKPT_PATH = FLAGS.CKPT_PATH
GEN_ITERS = FLAGS.GEN_ITERS
PRETRAIN = FLAGS.USE_PRETRAIN
