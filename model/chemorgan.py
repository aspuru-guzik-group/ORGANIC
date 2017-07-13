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
import pandas as pd
from tqdm import tqdm
__version__ = '0.2.1'

__logo__="""
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

    def __init__(self, params=None, read_file=True, params_file='exp.json'):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        print(__logo__.format(__version__))
        #    '#########################################################################\n\n')
        #print("   ___  _                         ___    __     ___    _        __  ")
        #print("  / __\| |__    ___  _ __ ___    /___\  /__\   / _ \  /_\    /\ \ \ ")
        #print(" / /   | '_ \\  / _ \| '_ ` _ \  //  // / \//  / /_\/ //_\\  /  \/ / ")
        #print("/ /___ | | | ||  __/| | | | | |/ \_// / _  \ / /_\\ /  _  \/ /\  /  ")
        ##print("\____/ |_| |_| \___||_| |_| |_|\___/  \/ \_/ \____/ \_/ \_/\_\ \/   ")
        #print("\n                                           version {}            ".format(__version__))
        #print(
        #    '#########################################################################\n\n')

        print('Setting up GPU...')
        self.detect_gpu()

        if read_file == True:
            self.params=json.loads(
                open(params_file).read(), object_pairs_hook=OrderedDict)
            print('Parameters loaded from {}'.format(params_file))
        elif params is not None:
            self.params=params
            print('Parameters loaded from user-specified dictionary.')
        else:
            raise ValueError('No parameters were specified.')

        self.set_hyperparameters()
        self.set_parameters()

        self.gen_loader=Gen_Dataloader(self.BATCH_SIZE)
        self.dis_loader=Dis_Dataloader()

        self.generator=Generator(self.NUM_EMB, self.BATCH_SIZE, self.EMB_DIM,
                                   self.HIDDEN_DIM, self.MAX_LENGTH, self.START_TOKEN)
        self.set_discriminator()

        self.sess=tf.Session(config=self.config)

    def setTrainingProgram(self, batches, objectives):

        if (type(batches) is list) or (type(objectives) is list):

            self.TRAINING_PROGRAM=True
            if type(objectives) is not list or type(batches) is not list:
                raise ValueError("Unmatching training program parameters")
            if len(objectives) != len(batches):
                raise ValueError("Unmatching training program parameters")
            self.TOTAL_BATCH=np.sum(np.asarray(batches))

            i=0
            self.EDUCATION={}
            for j, stage in enumerate(batches):
                for _ in range(stage):
                    self.EDUCATION[i]=objectives[j]
                    i += 1
        else:
            self.TRAINING_PROGRAM=False
            self.TOTAL_BATCH=batches
            self.OBJECTIVE=objectives

    def pretrain(self, sess, generator, train_discriminator):
        # samples = generate_samples(sess, BATCH_SIZE, generated_num)
        self.gen_loader.create_batches(self.positive_samples)
        results=OrderedDict({'exp_name': self.PREFIX})

        #  pre-train generator
        print('Start pre-training...')
        start=time.time()
        for epoch in tqdm(range(self.PRE_EPOCH_NUM)):
            print(' gen pre-train')
            loss=self.pre_train_epoch(sess, generator, self.gen_loader)
            if epoch == 10 or epoch % 40 == 0:
                samples=self.generate_samples(
                    sess, generator, self.BATCH_SIZE, self.SAMPLE_NUM)
                self.gen_loader.create_batches(samples)
                print('\t train_loss {}'.format(loss))
                mm.compute_results(
                    samples, self.train_samples, self.ord_dict, results)

        samples=self.generate_samples(
            sess, generator, self.BATCH_SIZE, self.SAMPLE_NUM)
        self.gen_loader.create_batches(samples)

        samples=self.generate_samples(
            sess, generator, self.BATCH_SIZE, self.SAMPLE_NUM)
        self.gen_loader.create_batches(samples)

        print('Start training discriminator...')
        for i in tqdm(range(self.dis_alter_epoch)):
            print(' discriminator pre-train')
            d_loss, acc=train_discriminator()
        end=time.time()
        print('Total time was {:.4f}s'.format(end - start))
        return

    def generate_samples(self, sess, trainable_model, batch_size, generated_num, verbose=False):
        #  Generated Samples
        generated_samples=[]
        start=time.time()
        for _ in range(int(generated_num / batch_size)):
            generated_samples.extend(trainable_model.generate(sess))
        end=time.time()
        if verbose:
            print('Sample generation time: %f' % (end - start))
        return generated_samples

    def pre_train_epoch(self, sess, trainable_model, data_loader):
        supervised_g_losses=[]
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            batch=data_loader.next_batch()
            _, g_loss, g_pred=trainable_model.pretrain_step(sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)

    def set_discriminator(self):

        with tf.variable_scope('discriminator'):
            self.discriminator=Discriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.dis_embedding_dim,
                filter_sizes=self.dis_filter_sizes,
                num_filters=self.dis_num_filters,
                l2_reg_lambda=self.dis_l2_reg_lambda)

        self.dis_params=[param for param in tf.trainable_variables()
                           if 'discriminator' in param.name]
        # Define Discriminator Training procedure
        self.dis_global_step=tf.Variable(
            0, name="global_step", trainable=False)
        self.dis_optimizer=tf.train.AdamOptimizer(1e-4)
        self.dis_grads_and_vars=self.dis_optimizer.compute_gradients(
            self.discriminator.loss, self.dis_params, aggregation_method=2)
        self.dis_train_op=self.dis_optimizer.apply_gradients(
            self.dis_grads_and_vars, global_step=self.dis_global_step)

    def train_discriminator(self):
        if self.D_WEIGHT == 0:
            return 0, 0

        negative_samples=self.generate_samples(
            self.sess, self.generator, self.BATCH_SIZE, self.POSITIVE_NUM)

        #  train discriminator
        dis_x_train, dis_y_train=self.dis_loader.load_train_data(
            self.positive_samples, negative_samples)
        dis_batches=self.dis_loader.batch_iter(
            zip(dis_x_train, dis_y_train), self.dis_batch_size, self.dis_num_epochs
        )

        for batch in dis_batches:
            x_batch, y_batch=zip(*batch)
            feed={
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
                self.discriminator.dropout_keep_prob: self.dis_dropout_keep_prob
            }
            _, step, loss, accuracy=self.sess.run(
                [self.dis_train_op, self.dis_global_step, self.discriminator.loss, self.discriminator.accuracy], feed)
        print('\tD loss  :   {}'.format(loss))
        print('\tAccuracy: {}'.format(accuracy))
        return loss, accuracy

    def detect_gpu(self):

        self.config=tf.ConfigProto()

        try:
            gpu_free_number=str(pick_gpus_lowest_memory()[0, 0])
            os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(gpu_free_number)
            print('GPUs {} detected and selected'.format(gpu_free_number))
            self.config.gpu_options.allow_growth=True

        except Exception:
            print('No GPU detected')
            pass

    def set_hyperparameters(self):

        # Training hyperparameters
        self.PREFIX=self.params['EXP_NAME']
        self.PRE_EPOCH_NUM=self.params['G_PRETRAIN_STEPS']
        self.TRAIN_ITER=self.params['G_STEPS']  # generator
        self.BATCH_SIZE=self.params["BATCH_SIZE"]
        self.SEED=self.params['SEED']
        self.dis_batch_size=64
        self.dis_num_epochs=3
        self.dis_alter_epoch=self.params['D_PRETRAIN_STEPS']

        self.BATCHES=self.params['TOTAL_BATCH']
        self.OBJECTIVE=self.params['OBJECTIVE']

        # Generator hyperparameters
        self.EMB_DIM=32
        self.HIDDEN_DIM=32
        self.START_TOKEN=0
        self.SAMPLE_NUM=6400
        self.BIG_SAMPLE_NUM=self.SAMPLE_NUM * 5
        self.D_WEIGHT=self.params['LAMBDA']

        self.D=max(int(5 * self.D_WEIGHT), 1)

        # Discriminator hyperparameters
        self.dis_embedding_dim=64
        self.dis_filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.dis_num_filters=[100, 200, 200, 200,
                                200, 100, 100, 100, 100, 100, 160, 160]
        self.dis_dropout_keep_prob=0.75
        self.dis_l2_reg_lambda=0.2

    def set_parameters(self):

        self.train_samples=mm.load_train_data(self.params['TRAIN_FILE'])
        self.char_dict, self.ord_dict=mm.build_vocab(self.train_samples)
        self.NUM_EMB=len(self.char_dict)
        self.DATA_LENGTH=max(map(len, self.train_samples))
        self.MAX_LENGTH=self.params["MAX_LENGTH"]
        to_use=[sample for sample in self.train_samples if mm.verified_and_below(
            sample, self.MAX_LENGTH)]
        self.positive_samples=[mm.encode(sample, self.MAX_LENGTH, self.char_dict)
                                 for sample in to_use]
        self.POSITIVE_NUM=len(self.positive_samples)
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

        mm.print_params(self.params)

    def make_reward(self, train_samples, nbatch):

        if self.TRAINING_PROGRAM == False:

            reward_func=mm.load_reward(self.OBJECTIVE)

            def batch_reward(samples):
                decoded=[mm.decode(sample, self.ord_dict)
                           for sample in samples]
                pct_unique=len(list(set(decoded))) / float(len(decoded))
                rewards=reward_func(decoded, train_samples)
                weights=np.array([pct_unique / float(decoded.count(sample))
                                    for sample in decoded])

                return rewards * weights

            return batch_reward

        else:

            reward_func=mm.load_reward(self.education[nbatch])

            def batch_reward(samples):
                decoded=[mm.decode(sample, self.ord_dict)
                           for sample in samples]
                pct_unique=len(list(set(decoded))) / float(len(decoded))
                rewards=reward_func(decoded, train_samples)
                weights=np.array([pct_unique / float(decoded.count(sample))
                                    for sample in decoded])

                return rewards * weights

            return batch_reward

    def print_rewards(self, rewards):
        print('Rewards be like...')
        np.set_printoptions(precision=3, suppress=True)
        print(rewards)
        mean_r, std_r=np.mean(rewards), np.std(rewards)
        min_r, max_r=np.min(rewards), np.max(rewards)
        print('Mean: {:.3f} , Std:  {:.3f}'.format(mean_r, std_r), end='')
        print(', Min: {:.3f} , Max:  {:.3f}\n'.format(min_r, max_r))
        np.set_printoptions(precision=8, suppress=False)
        return

    def save_results(self, sess, folder, name, results_rows=None, nbatch=None):
        if results_rows is not None:
            df=pd.DataFrame(results_rows)
            df.to_csv('{}_results.csv'.format(folder), index=False)
        if nbatch is None:
            label='final'
        else:
            label=str(nbatch)

        # save models
        model_saver=tf.train.Saver()
        ckpt_dir=os.path.join(self.params['CHK_PATH'], folder)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_file=os.path.join(
            ckpt_dir, '{}_{}.ckpt'.format(name, label))
        path=model_saver.save(sess, ckpt_file)
        print('Model saved at {}'.format(path))
        return

    def load_prev(self):

        # Loading previous checkpoints
        saver=tf.train.Saver()
        pretrain_is_loaded=False
        sess_is_loaded=False

        ckpt_dir='checkpoints/{}_pretrain'.format(self.PREFIX)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_file=os.path.join(ckpt_dir, 'pretrain_ckpt')
        if os.path.isfile(ckpt_file + '.meta') and self.params["LOAD_PRETRAIN"]:
            saver.restore(self.sess, ckpt_file)
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt_file))
            pretrain_is_loaded=True
        else:
            if self.params["LOAD_PRETRAIN"]:
                print(
                    '\t* No pre-training data found as {:s}.'.format(ckpt_file))
            else:
                print('\t* LOAD_PRETRAIN was set to false.')

        self.rollout=Rollout(self.generator, 0.8)

        if self.params['LOAD_PREV_SESS']:
            saver=tf.train.Saver()
            ckpt_dir='checkpoints/{}'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file=os.path.join(ckpt_dir, self.params['PREV_CKPT'])
            if os.path.isfile(ckpt_file + '.meta'):
                saver.restore(self.sess, ckpt_file)
                print('Training loaded from previous checkpoint {}'.format(ckpt_file))
                sess_is_loaded=True
            else:
                print('\t* No training data found as {:s}.'.format(ckpt_file))
        else:
            print('\t* LOAD_PREV_SESS was set to false.')

        if not pretrain_is_loaded and not sess_is_loaded:
            self.sess.run(tf.global_variables_initializer())
            self.pretrain(self.sess, self.generator, self.train_discriminator)
            path=saver.save(self.sess, ckpt_file)
            print('Pretrain finished and saved at {}'.format(path))

    def train(self):

        print('#########################################################################')
        print('Start Reinforcement Training Generator...')
        results_rows=[]
        for nbatch in tqdm(range(self.TOTAL_BATCH)):
            results=OrderedDict({'exp_name': self.PREFIX})
            batch_reward=self.make_reward(self.train_samples, nbatch)
            if nbatch % 1 == 0 or nbatch == self.TOTAL_BATCH - 1:
                print('* Making samples')
                if nbatch % 10 == 0:
                    gen_samples=self.generate_samples(
                        self.sess, self.generator, self.BATCH_SIZE, self.BIG_SAMPLE_NUM)
                else:
                    gen_samples=self.generate_samples(
                        self.sess, self.generator, self.BATCH_SIZE, self.SAMPLE_NUM)
                self.gen_loader.create_batches(gen_samples)
                print('batch_num: {}'.format(nbatch))
                results['Batch']=nbatch

                # results
                mm.compute_results(
                    gen_samples, self.train_samples, self.ord_dict, results)

            print(
                '#########################################################################')
            print('-> Training generator with RL.')
            print('G Epoch {}'.format(nbatch))

            for it in range(self.TRAIN_ITER):
                samples=self.generator.generate(self.sess)
                rewards=self.rollout.get_reward(
                    self.sess, samples, 16, self.discriminator, batch_reward, self.D_WEIGHT)
                nll=self.generator.generator_step(
                    self.sess, samples, rewards)
                # results
                self.print_rewards(rewards)
                print('neg-loglike: {}'.format(nll))
                results['neg-loglike']=nll
            self.rollout.update_params()

            # generate for discriminator
            print('-> Training Discriminator')
            for i in range(self.D):
                print('D_Epoch {}'.format(i))
                d_loss, accuracy=self.train_discriminator()
                results['D_loss_{}'.format(i)]=d_loss
                results['Accuracy_{}'.format(i)]=accuracy
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

    model=ChemORGAN()
    #model.load_prev()
    #model.train()
