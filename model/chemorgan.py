from __future__ import absolute_import, division, print_function
from gpu_utils import pick_gpus_lowest_memory
from builtins import range
from collections import OrderedDict
import os
from generator import Generator, Rollout
import numpy as np
import tensorflow as tf
import random
import time
import mol_methods as mm
import json
from data_loaders import Gen_Dataloader, Dis_Dataloader
from discriminator import Discriminator
from custom_metrics import get_metrics, metrics_loading
from rdkit import rdBase
import pandas as pd
from tqdm import tqdm
__version__ = '0.2.1'

__logo__ = """
//////////////////////////////////////////////////////////////////
     ___ _                      ___                       
    / __\ |__   ___ _ __ ___   /___\_ __ __ _  __ _ _ __  
   / /  | '_ \ / _ \ '_ ` _ \ //  // '__/ _` |/ _` | '_ \ 
  / /___| | | |  __/ | | | | / \_//| | | (_| | (_| | | | |
  \____/|_| |_|\___|_| |_| |_\___/ |_|  \__, |\__,_|_| |_|
                                        |___/             
                                             version {}            
////////////////////////////////////////////////////////////////"""


class ChemORGAN(object):

    def __init__(self, name, params=None, read_file=False, params_file='exp.json'):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        rdBase.DisableLog('rdApp.error')
        print(__logo__.format(__version__))
        self.detect_gpu()
        self.set_default()
        self.PREFIX = name

        if read_file == True:
            self.params = json.loads(
                open(params_file).read(), object_pairs_hook=OrderedDict)
            self.set_user_parameters()
            print('Parameters loaded from {}'.format(params_file))
        elif params is not None:
            self.params = params
            self.set_user_parameters()
            print('Parameters loaded from user-specified dictionary.')
        else:
            print('No parameters were specified. ChemORGAN will use default values.')

        self.pretrain_is_loaded = False
        self.sess_is_loaded = False

    def detect_gpu(self):

        self.config = tf.ConfigProto()

        try:
            gpu_free_number = str(pick_gpus_lowest_memory()[0, 0])
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)
            print('GPUs {} detected and selected'.format(gpu_free_number))
            self.config.gpu_options.allow_growth = True

        except Exception:
            print('No GPU detected')
            pass

    def set_default(self):

        # Training hyperparameters
        self.PRETRAIN_GEN_EPOCHS = 240
        self.PRETRAIN_DIS_EPOCHS = 50
        self.GEN_ITERATIONS = 2  # generator
        self.GEN_BATCH_SIZE = 64
        self.SEED = None
        self.DIS_BATCH_SIZE = 64
        self.DIS_EPOCHS = 3

        # Generator hyperparameters
        self.GEN_EMB_DIM = 32
        self.GEN_HIDDEN_DIM = 32
        self.START_TOKEN = 0
        self.SAMPLE_NUM = 6400
        self.BIG_SAMPLE_NUM = self.SAMPLE_NUM * 5
        self.LAMBDA = 0.5
        self.D = max(int(5 * self.LAMBDA), 1)

        # Discriminator hyperparameters
        self.DIS_EMB_DIM = 64
        self.DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.DIS_NUM_FILTERS = [100, 200, 200, 200,
                                200, 100, 100, 100, 100, 100, 160, 160]
        self.DIS_DROPOUT = 0.75
        self.DIS_L2REG = 0.2

    def set_user_parameters(self):

        # Training hyperparameters
        if 'PRETRAIN_GEN_EPOCHS' in self.params:
            self.PRETRAIN_GEN_EPOCHS = self.params['PRETRAIN_GEN_EPOCHS']
        if 'PRETRAIN_DIS_EPOCHS' in self.params:
            self.PRETRAIN_DIS_EPOCHS = self.params['PRETRAIN_DIS_EPOCHS']
        if 'GEN_ITERATIONS' in self.params:
            self.GEN_ITERATIONS = self.params['GEN_ITERATIONS']  # generator
        if 'GEN_BATCH_SIZE' in self.params:
            self.GEN_BATCH_SIZE = self.params['GEN_BATCH_SIZE']
        if 'SEED' in self.params:
            self.SEED = self.params['SEED']
        if 'DIS_BATCH_SIZE' in self.params:
            self.DIS_BATCH_SIZE = self.params['DIS_BATCH_SIZE']
        if 'DIS_EPOCHS' in self.params:
            self.DIS_EPOCHS = self.params['DIS_EPOCHS']

        # Generator hyperparameters
        if 'GEN_EMB_DIM' in self.params:
            self.GEN_EMB_DIM = self.params['GEN_EMB_DIM']
        if 'GEN_HIDDEN_DIM' in self.params:
            self.GEN_HIDDEN_DIM = self.params['GEN_HIDDEN_DIM']
        if 'START_TOKEN' in self.params:
            self.START_TOKEN = self.params['START_TOKEN']
        if 'SAMPLE_NUM' in self.params:
            self.SAMPLE_NUM = self.params['SAMPLE_NUM']
        if 'BIG_SAMPLE_NUM' in self.params:
            self.BIG_SAMPLE_NUM = self.params['BIG_SAMPLE_NUM']
        elif 'SAMPLE_NUM' in self.params:
            self.BIG_SAMPLE_NUM = self.params['SAMPLE_NUM'] * 5
        if 'LAMBDA' in self.params:
            self.LAMBDA = self.params['LAMBDA']
            self.D = max(int(5 * self.LAMBDA), 1)

        # Discriminator hyperparameters
        if 'DIS_EMB_DIM' in self.params:
            self.DIS_EMB_DIM = self.params['DIS_EMB_DIM']
        if 'DIS_FILTER_SIZES' in self.params:
            self.DIS_FILTER_SIZES = self.params['DIS_FILTER_SIZES']
        if 'DIS_NUM_FILTERS' in self.params:
            self.DIS_NUM_FILTERS = self.params['DIS_FILTER_SIZES']
        if 'DIS_DROPOUT' in self.params:
            self.DIS_DROPOUT = self.params['DIS_DROPOUT']
        if 'DIS_L2REG' in self.params:
            self.DIS_L2REG = self.params['DIS_L2REG']

    def load_training_set(self, file):
        
        self.train_samples = mm.load_train_data(file)
        self.char_dict, self.ord_dict = mm.build_vocab(self.train_samples)
        self.NUM_EMB = len(self.char_dict)
        self.PAD_CHAR = self.ord_dict[self.NUM_EMB-1]
        self.PAD_NUM = self.char_dict[self.PAD_CHAR]
        self.DATA_LENGTH = max(map(len, self.train_samples))
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.train_samples, key=len)) * 1.5)
        to_use = [sample for sample in self.train_samples if mm.verified_and_below(
            sample, self.MAX_LENGTH)]
        self.positive_samples = [mm.encode(sample, self.MAX_LENGTH, self.char_dict)
                                 for sample in to_use]
        self.POSITIVE_NUM = len(self.positive_samples)
        print('Starting ObjectiveGAN for {:7s}'.format(self.PREFIX))
        print('Data points in train_file {:7d}'.format(
            len(self.train_samples)))
        print('Max data length is        {:7d}'.format(self.DATA_LENGTH))
        print('Max length to use is      {:7d}'.format(self.MAX_LENGTH))
        print('Avg length to use is      {:7f}'.format(
            np.mean([len(s) for s in to_use])))
        print('Num valid data points is  {:7d}'.format(self.POSITIVE_NUM))
        print('Size of alphabet is       {:7d}'.format(self.NUM_EMB))
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        #mm.print_params(self.params)

        self.gen_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.dis_loader = Dis_Dataloader()
        self.generator = Generator(self.NUM_EMB, self.GEN_BATCH_SIZE, self.GEN_EMB_DIM,
                                   self.GEN_HIDDEN_DIM, self.MAX_LENGTH, self.START_TOKEN)
        self.set_discriminator()
        self.sess = tf.Session(config=self.config)

    def set_training_program(self, metrics=None, steps=None):

        self.TOTAL_BATCH = np.sum(np.asarray(steps))
        self.EDUCATION = {}
        self.METRICS = metrics
        i = 0
        for j, stage in enumerate(steps):
            for _ in range(stage):
                self.EDUCATION[i] = metrics[j]
                i += 1

    def set_discriminator(self):

        with tf.variable_scope('discriminator'):
            self.discriminator = Discriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG)

        self.dis_params = [param for param in tf.trainable_variables()
                           if 'discriminator' in param.name]
        # Define Discriminator Training procedure
        self.dis_global_step = tf.Variable(
            0, name="global_step", trainable=False)
        self.dis_optimizer = tf.train.AdamOptimizer(1e-4)
        self.dis_grads_and_vars = self.dis_optimizer.compute_gradients(
            self.discriminator.loss, self.dis_params, aggregation_method=2)
        self.dis_train_op = self.dis_optimizer.apply_gradients(
            self.dis_grads_and_vars, global_step=self.dis_global_step)

    def make_reward(self, train_samples, nbatch):

        metric = self.EDUCATION[nbatch] 
        reward_func = self.load_reward(metric)

        def batch_reward(samples):
            decoded = [mm.decode(sample, self.ord_dict)
                       for sample in samples]
            pct_unique = len(list(set(decoded))) / float(len(decoded))
            rewards = reward_func(decoded, train_samples, **self.kwargs[metric])
            weights = np.array([pct_unique / float(decoded.count(sample))
                                for sample in decoded])

            return rewards * weights

        return batch_reward

    def load_reward(self, objective):

        metrics = get_metrics()

        if objective in metrics.keys():
            return metrics[objective]
        else:
            raise ValueError('objective {} not found!'.format(objective))
        return

    def load_metrics(self):

        loadings = metrics_loading()
        met = list(set(self.METRICS))
        self.kwargs = {}

        for m in met:

            load_fun = loadings[m]
            args = load_fun()
            if args is not None:
                fun_args = {}
                for arg in args:
                    fun_args[arg[0]] = arg[1]
                self.kwargs[m] = fun_args
            else: 
                self.kwargs[m] = None

        return 

    def load_prev_pretraining(self, ckpt_dir=None):

        # Loading previous checkpoints
        saver = tf.train.Saver()
        if ckpt_dir is None:
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_file = os.path.join(ckpt_dir, 'pretrain_ckpt')
        if os.path.isfile(ckpt_file + '.meta'):
            saver.restore(self.sess, ckpt_file)
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt_file))
            self.pretrain_is_loaded = True
        else:
            print('\t* No pre-training data found as {:s}.'.format(ckpt_file))

    def load_prev_training(self, prev_ckpt):

        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        saver = tf.train.Saver()
        ckpt_dir = 'checkpoints/{}'.format(self.PREFIX)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_file = os.path.join(ckpt_dir, prev_ckpt)
        if os.path.isfile(ckpt_file + '.meta'):
            saver.restore(self.sess, ckpt_file)
            print('Training loaded from previous checkpoint {}'.format(ckpt_file))
            self.sess_is_loaded = True
        else:
            print('\t* No training checkpoint found as {:s}.'.format(ckpt_file))
 
    def pre_train_epoch(self, sess, trainable_model, data_loader):
        supervised_g_losses = []
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            batch = data_loader.next_batch()
            _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)

    def pretrain(self, sess, generator, train_discriminator):
        self.gen_loader.create_batches(self.positive_samples)
        results = OrderedDict({'exp_name': self.PREFIX})

        #  pre-train generator
        print('Start pre-training...')
        start = time.time()
        for epoch in tqdm(range(self.PRETRAIN_GEN_EPOCHS)):
            print(' gen pre-train')
            loss = self.pre_train_epoch(sess, generator, self.gen_loader)
            if epoch == 10 or epoch % 40 == 0:
                samples = self.generate_samples(
                    sess, generator, self.GEN_BATCH_SIZE, self.SAMPLE_NUM)
                self.gen_loader.create_batches(samples)
                print('\t train_loss {}'.format(loss))
                mm.compute_results(
                    samples, self.train_samples, self.ord_dict, results)

        samples = self.generate_samples(
            sess, generator, self.GEN_BATCH_SIZE, self.SAMPLE_NUM)
        self.gen_loader.create_batches(samples)

        samples = self.generate_samples(
            sess, generator, self.GEN_BATCH_SIZE, self.SAMPLE_NUM)
        self.gen_loader.create_batches(samples)

        print('Start training discriminator...')
        for i in tqdm(range(self.PRETRAIN_DIS_EPOCHS)):
            print(' discriminator pre-train')
            d_loss, acc = train_discriminator()
        end = time.time()
        print('Total time was {:.4f}s'.format(end - start))
        return

    def print_rewards(self, rewards):
        print('Rewards be like...')
        np.set_printoptions(precision=3, suppress=True)
        print(rewards)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        min_r, max_r = np.min(rewards), np.max(rewards)
        print('Mean: {:.3f} , Std:  {:.3f}'.format(mean_r, std_r), end='')
        print(', Min: {:.3f} , Max:  {:.3f}\n'.format(min_r, max_r))
        np.set_printoptions(precision=8, suppress=False)
        return

    def save_results(self, sess, folder, name, results_rows=None, nbatch=None):
        if results_rows is not None:
            df = pd.DataFrame(results_rows)
            df.to_csv('{}_results.csv'.format(folder), index=False)
        if nbatch is None:
            label = 'final'
        else:
            label = str(nbatch)

        # save models
        model_saver = tf.train.Saver()
        ckpt_dir = os.path.join(self.params['CHK_PATH'], folder)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_file = os.path.join(
            ckpt_dir, '{}_{}.ckpt'.format(name, label))
        path = model_saver.save(sess, ckpt_file)
        print('Model saved at {}'.format(path))
        return

    def generate_samples(self, sess, trainable_model, batch_size, generated_num, verbose=False):
        #  Generated Samples
        generated_samples = []
        start = time.time()
        for _ in range(int(generated_num / batch_size)):
            generated_samples.extend(trainable_model.generate(sess))
        end = time.time()
        if verbose:
            print('Sample generation time: %f' % (end - start))
        return generated_samples

    def train_discriminator(self):
        if self.LAMBDA == 0:
            return 0, 0

        negative_samples = self.generate_samples(
            self.sess, self.generator, self.GEN_BATCH_SIZE, self.POSITIVE_NUM)

        #  train discriminator
        dis_x_train, dis_y_train = self.dis_loader.load_train_data(
            self.positive_samples, negative_samples)
        dis_batches = self.dis_loader.batch_iter(
            zip(dis_x_train, dis_y_train), self.DIS_BATCH_SIZE, self.DIS_EPOCHS
        )

        for batch in dis_batches:
            x_batch, y_batch = zip(*batch)
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
                self.discriminator.dropout_keep_prob: self.DIS_DROPOUT
            }
            _, step, loss, accuracy = self.sess.run(
                [self.dis_train_op, self.dis_global_step, self.discriminator.loss, self.discriminator.accuracy], feed)
        print('\tD loss  :   {}'.format(loss))
        print('\tAccuracy: {}'.format(accuracy))
        return loss, accuracy

    def train(self):

        if not self.pretrain_is_loaded and not self.sess_is_loaded:
            self.sess.run(tf.global_variables_initializer())
            self.pretrain(self.sess, self.generator, self.train_discriminator)
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file = os.path.join(ckpt_dir, 'pretrain_ckpt')
            saver = tf.train.Saver()
            path = saver.save(self.sess, ckpt_file)
            print('Pretrain finished and saved at {}'.format(path))

        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        print('#########################################################################')
        print('Start Reinforcement Training Generator...')
        results_rows = []
        for nbatch in tqdm(range(self.TOTAL_BATCH)):

            results = OrderedDict({'exp_name': self.PREFIX})
            batch_reward = self.make_reward(self.train_samples, nbatch)

            print('* Making samples')
            if nbatch % 10 == 0:
                gen_samples = self.generate_samples(
                    self.sess, self.generator, self.GEN_BATCH_SIZE, self.BIG_SAMPLE_NUM)
            else:
                gen_samples = self.generate_samples(
                    self.sess, self.generator, self.GEN_BATCH_SIZE, self.SAMPLE_NUM)
            self.gen_loader.create_batches(gen_samples)
            print('batch_num: {}'.format(nbatch))
            results['Batch'] = nbatch

            # results
            mm.compute_results(
                gen_samples, self.train_samples, self.ord_dict, results)

            print(
                '#########################################################################')
            print('-> Training generator with RL.')
            print('G Epoch {}'.format(nbatch))

            for it in range(self.GEN_ITERATIONS):
                samples = self.generator.generate(self.sess)
                rewards = self.rollout.get_reward(
                    self.sess, samples, 16, self.discriminator, batch_reward, self.LAMBDA)
                nll = self.generator.generator_step(
                    self.sess, samples, rewards)
                # results
                self.print_rewards(rewards)
                print('neg-loglike: {}'.format(nll))
                results['neg-loglike'] = nll
            self.rollout.update_params()

            # generate for discriminator
            print('-> Training Discriminator')
            for i in range(self.D):
                print('D_Epoch {}'.format(i))
                d_loss, accuracy = self.train_discriminator()
                results['D_loss_{}'.format(i)] = d_loss
                results['Accuracy_{}'.format(i)] = accuracy
            print('results')
            results_rows.append(results)
            if nbatch % self.params["EPOCH_SAVES"] == 0:
                self.save_results(self.sess, self.PREFIX,
                                  self.PREFIX + '_model', results_rows, nbatch)

        # write results
        self.save_results(self.sess, self.PREFIX,
                          self.PREFIX + '_model', results_rows)

        print('\n:*** FINISHED ***')

if __name__ == '__main__':

    model = ChemORGAN('load_test')
    model.load_training_set('../data/toy.csv')
    model.load_prev_pretraining()
    model.set_training_program(['validity'], [1])
    model.load_metrics()
    model.train()
