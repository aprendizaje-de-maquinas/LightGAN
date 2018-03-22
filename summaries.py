'''
Methods for logging
'''
import os
import tensorflow as tf

import model_and_data_serialization
from config import LOGS_DIR

FLAGS = tf.app.flags.FLAGS


def define_summaries(disc_cost_real, disc_cost_fake, gen_cost, disc_cost):
    '''
    Defines summaries for various differnt tensorboard graphs
    '''
    train_writer = tf.summary.FileWriter(LOGS_DIR)
    tf.summary.scalar("d_acc_real", disc_cost_real)
    tf.summary.scalar("d_acc_fake", disc_cost_fake)
    tf.summary.scalar("g_loss", gen_cost)
    tf.summary.scalar('d_loss', disc_cost)
    merged = tf.summary.merge_all()
    return merged, train_writer


def log_samples(samples, scores, iteration, seq_length, prefix):
    '''
    Logs the samples from the generator network.

    if scores, then we also output what the discrinator scored the sample
    else then we just ouptut the inference
    '''
    if scores:
        sample_scores = zip(samples, scores)
        sample_scores = sorted(sample_scores, key=lambda sample: sample[1])
    

        with open(model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + '/{}_{}.txt'.format(
                prefix, iteration),
                  'a') as f:
            for s, score in sample_scores:
                # dont print <naw>
                a = [ w for w in s if w != "<naw>" ]
                if s[-1] == '<naw>':
                    a.append(s[-1])
                s = " ".join(a)
                f.write("%s \t %f\n" % (s, score))
            f.close()
    else:
        with open(model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + '/{}_{}.txt'.format(
                prefix, iteration),
                  'a') as f:
            for s in samples:
                # dont print <naw>
                a = [ w for w in s if w != "<naw>"]
                if s[-1] == '<naw>':
                    a.append(s[-1])
                s = " ".join(a)
                f.write("%s \n" % (s))
            f.close()


def log_run_settings():
    '''
    Simply logs all of the FLAGS
    '''
    with open(os.path.join(LOGS_DIR, 'run_settings.txt'), 'w') as f:
        for key in tf.flags.FLAGS.__flags.keys():
            entry = "%s: %s" % (key, tf.flags.FLAGS.__flags[key])
            f.write(entry + '\n')
    f.close()
