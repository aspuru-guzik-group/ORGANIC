"""

GENERATOR
====================

Generator and Rollout models.

This module is slightly modified from Gabriel Guimaraes and
Benjamin Sanchez-Lengeling original implementation (which is
also a slight modification from SeqGAN's model).

Carlos Outeiral has handled some issues with the Rollout efficiency,
removed duplicates, corrected bugs and added some documentation.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
    """
    Class for the generative model.
    """

    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.001, reward_gamma=0.95,
                 temperature=1.0, grad_clip=5.0):
        """Sets parameters and defines the model architecture."""

        """
        Set specified parameters
        """
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.reward_gamma = reward_gamma
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.start_token = tf.constant(
            [start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)

        """
        Set important internal variables
        """
        self.g_params = []  # This list will be updated with LSTM's parameters
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))
        self.x = tf.placeholder(  # true data, not including start token
            tf.int32, shape=[self.batch_size, self.sequence_length])
        self.rewards = tf.placeholder(  # rom rollout policy and discriminator
            tf.float32, shape=[self.batch_size, self.sequence_length])

        """
        Define generative model
        """
        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(
                self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(
                self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(
                self.g_params)  # maps h_t to o_t (output token logits)

        """
        Process the batches
        """
        with tf.device("/cpu:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length,
                              value=tf.nn.embedding_lookup(self.g_embeddings,
                                                           self.x))
            self.processed_x = tf.stack(  # seq_length x batch_size x emb_dim
                [tf.squeeze(input_, [1]) for input_ in inputs])
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        """
        Generative process
        """
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32,
                                             size=self.sequence_length,
                                             dynamic_size=False,
                                             infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32,
                                             size=self.sequence_length,
                                             dynamic_size=False,
                                             infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(
                log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(
                self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0]) # batch_size x seq_length

        """
        Pretraining
        """
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        g_logits = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions, g_logits):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(
                i, tf.nn.softmax(o_t))  # batch x vocab_size
            g_logits = g_logits.write(i, o_t)  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions, g_logits

        _, _, _, self.g_predictions, self.g_logits = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(
                           self.g_embeddings, self.start_token),
                       self.h0, g_predictions, g_logits))

        self.g_predictions = tf.transpose(
            self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.g_logits = tf.transpose(
            self.g_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions,
                                            [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        pretrain_opt = self.g_optimizer(self.learning_rate)  # training updates
        self.pretrain_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(
            zip(self.pretrain_grad, self.g_params))

        """
        Unsupervised Training
        """
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(
                        self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )
        g_opt = self.g_optimizer(self.learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

    def generate(self, session):
        """Generates a batch of samples."""
        outputs = session.run([self.gen_x])
        return outputs[0]

    def pretrain_step(self, session, x):
        """Performs a pretraining step on the generator."""
        outputs = session.run([self.pretrain_updates, self.pretrain_loss,
                               self.g_predictions], feed_dict={self.x: x})
        return outputs

    def generator_step(self, sess, samples, rewards):
        """Performs a training step on the generator."""
        feed = {self.x: samples, self.rewards: rewards}
        _, g_loss = sess.run([self.g_updates, self.g_loss], feed_dict=feed)
        return g_loss

    def init_matrix(self, shape):
        """Returns a normally initialized matrix of a given shape."""
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        """Returns a vector of zeros of a given shape."""
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        """Defines the recurrent process in the LSTM."""

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        """Defines the output part of the LSTM."""

        self.Wo = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        """Sets the optimizer."""
        return tf.train.AdamOptimizer(*args, **kwargs)


class Rollout(object):
    """
    Class for the rollout policy model.
    """

    def __init__(self, lstm, update_rate, pad_num):
        """Sets parameters and defines the model architecture."""

        self.lstm = lstm
        self.update_rate = update_rate
        self.pad_num = pad_num
        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        # maps h_tm1 to h_t for generator
        self.g_recurrent_unit = self.create_recurrent_unit()
        # maps h_t to o_t (output token logits)
        self.g_output_unit = self.create_output_unit()

        #######################################################################
        # placeholder definition
        self.x = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)
        # sequence of indices of generated data generated by generator, not
        # including start token

        # processed for batch
        with tf.device("/cpu:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length,
                              value=tf.nn.embedding_lookup(self.g_embeddings, self.x))
            self.processed_x = tf.stack(
                [tf.squeeze(input_, [1]) for input_ in inputs])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))
        #######################################################################

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(
                log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(
                self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        # batch_size x seq_length
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])

    def get_reward(self, sess, input_x, rollout_num, cnn, reward_fn=None, D_weight=1):
        """Calculates the rewards for a list of SMILES strings."""

        reward_weight = 1 - D_weight
        rewards = []
        for i in range(rollout_num):

            already = []
            for given_num in range(1, self.sequence_length):
                feed = {self.x: input_x, self.given_num: given_num}
                outputs = sess.run([self.gen_x], feed)
                generated_seqs = outputs[0]  # batch_size x seq_length
                gind = np.array(range(len(generated_seqs)))

                feed = {cnn.input_x: generated_seqs,
                        cnn.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(cnn.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])

                if reward_fn:

                    ypred = D_weight * ypred
                    # Delete sequences that are already finished,
                    # and add their rewards
                    for k, r in reversed(already):
                        generated_seqs = np.delete(generated_seqs, k, 0)
                        gind = np.delete(gind, k, 0)
                        ypred[k] += reward_weight * r

                    # If there are still seqs, calculate rewards
                    if generated_seqs.size:
                        rew = reward_fn(generated_seqs)

                    # Add the just calculated rewards
                    for k, r in zip(gind, rew):
                        ypred[k] += reward_weight * r

                    # Choose the seqs finished in the last iteration
                    for j, k in enumerate(gind):
                        if input_x[k][given_num] == self.pad_num and input_x[k][given_num-1] == self.pad_num:
                            already.append((k, rew[j]))
                    already = sorted(already, key=lambda el: el[0])

                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred       

            # Last char reward
            feed = {cnn.input_x: input_x, cnn.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(cnn.ypred_for_auc, feed)
            if reward_fn:
                ypred = D_weight * np.array([item[1]
                                             for item in ypred_for_auc])
                ypred += reward_weight * reward_fn(input_x)
            else:
                ypred = np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[-1] += ypred

        rewards = np.transpose(np.array(rewards)) / \
            (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def create_recurrent_unit(self):
        """Defines the recurrent process in the LSTM."""

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        """Updates the weights and biases of the rollout's LSTM
        recurrent unit following the results of the training."""

        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + \
            (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + \
            (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + \
            (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + \
            (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + \
            (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        """Defines the output process in the LSTM."""

        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        """Updates the weights and biases of the rollout's LSTM
        output unit following the results of the training."""

        self.Wo = self.update_rate * self.Wo + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + \
            (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        """Updates all parameters in the rollout's LSTM."""
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()
