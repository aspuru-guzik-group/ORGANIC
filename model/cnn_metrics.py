import sys
import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem.AllChem as Chem
from rdkit import DataStructs
from rdkit.Chem import PandasTools
from tqdm import tqdm
import os
import math
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn

class CNN(object):

    def __init__(self, label, nBits=4096):

        self.nBits = nBits
        if self.nBits != 4096 and self.nBits != 1024:
            raise ValueError('{} is not a valid value for nBits'.format(self.nBits))
        self.lBits = math.sqrt(self.nBits)

        self.label = label
        self.model_folder = os.path.join(os.getcwd(), 'neuralnets/{}'.format(self.label))
        if not os.path.exists(self.model_folder):
            raise ValueError('Could not find trained models for {}'.format(self.label))
        self.model_folder = os.path.join(self.model_folder, '{:02.0f}bit'.format(self.lBits))
        print(self.model_folder)
        if not os.path.exists(self.model_folder):
            raise ValueError('Could not find {:02.0f} bit models for {}'.format(self.lBits, self.label))

        self.model = self.model_cnn64 if self.lBits == 64 else self.model_cnn32
        self.nn = learn.Estimator(model_fn=self.model, model_dir=self.model_folder)

    def model_cnn64(self, features, targets, mode):

        features = tf.cast(features, tf.float32)
        input_layer = tf.reshape(features, [-1, 64, 64, 1])

        in_conv = tf.layers.conv2d(       
            inputs=input_layer,
            filters=68,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
        in_pool = tf.layers.max_pooling2d(
            inputs=in_conv,
            pool_size = [2,2],
            strides=4)

        conv2 = tf.layers.conv2d(
            inputs=in_pool,
            filters=68,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2,2],
            strides=2)

        end_conv = tf.layers.conv2d(
            inputs=pool2,
            filters=68,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
            
        end_pool = tf.layers.max_pooling2d(
            inputs=end_conv,
            pool_size=[2,2],
            strides=2)

        in_flat = tf.reshape(end_pool, [-1, 8*8*17])
        dense = tf.layers.dense(
            inputs=in_flat,
            units=1024,
            activation=tf.nn.relu)

        output = tf.layers.dense(
            inputs=dense,
            units=1,
            activation=tf.nn.relu)

        loss = None
        train_op = None
        prediction_values = tf.reshape(output, [-1])
        predictions = {"output": prediction_values}

        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.mean_squared_error(
            targets,
            prediction_values)

        if mode == learn.ModeKeys.TRAIN:
            targets = tf.cast(targets, tf.float32)
            train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")
            
        model = model_fn.ModelFnOps(
            mode = mode,
            predictions = predictions,
            loss = loss,
            train_op = train_op)

        return model

    def model_cnn32(self, features, targets, mode):

        features = tf.cast(features, tf.float32)
        input_layer = tf.reshape(features, [-1, 32, 32, 1])

        in_conv = tf.layers.conv2d(       
            inputs=input_layer,
            filters=36,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
        in_pool = tf.layers.max_pooling2d(
            inputs=in_conv,
            pool_size = [2,2],
            strides=4)

        end_conv = tf.layers.conv2d(
            inputs=in_pool,
            filters=36,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
        end_pool = tf.layers.max_pooling2d(
            inputs=end_conv,
            pool_size=[2,2],
            strides=2)

        in_flat = tf.reshape(end_pool, [-1, 4*4*36])
        dense = tf.layers.dense(
            inputs=in_flat,
            units=1024,
            activation=tf.nn.relu)
        output = tf.layers.dense(
            inputs=dense,
            units=1,
            activation=tf.nn.relu)

        loss = None
        train_op = None
        prediction_values = tf.reshape(output, [-1])
        predictions = {"output": prediction_values}

        if mode != learn.ModeKeys.INFER:
            loss = tf.losses.mean_squared_error(
            targets,
            prediction_values)

        if mode == learn.ModeKeys.TRAIN:
            targets = tf.cast(targets, tf.float32)
            train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

        model = model_fn.ModelFnOps(
            mode = mode,
            predictions = predictions,
            loss = loss,
            train_op = train_op)

        return model

    def predict(self, smiles):

        data_x = self.computeFingerprints(smiles)

        def input_predict(features):
            def _input_fn():
                all_x = tf.constant(features, shape=features.shape, dtype=tf.float32)
                return all_x
            return _input_fn

        predictions = self.nn.predict(input_fn=input_predict(data_x))
        return self.itToList(predictions)

    def computeFingerprints(self, smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fps = [np.asarray(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=self.nBits)) for mol in mols]
        return np.asarray(fps)

    def itToList(self, predictions):
        return [p['output'] for p in predictions]



import time
data = pd.read_csv('../data/opv.csv')
smiles_text = data['smiles'][10000:]
pce = data['PCE_calib'][10000:]
print('Data loaded. Starting to predict...')
cnn = CNN('pce', nBits=4096)
nbatch = 1000

error = np.zeros(int(len(pce)/nbatch))

for i in range(len(error)):
    smiles = np.asarray(smiles_text[nbatch*i:nbatch*(i+1)])
    real = pce[nbatch*i:nbatch*(i+1)]

    start = time.time()
    pr = cnn.predict(smiles)
    end = time.time() - start

    error[i] = np.mean(np.abs(np.asarray(pr) - np.asarray(real)))
    print('Batch {} predicted in {} s with mean error {}'.format(i+1, end, error[i]))

print('Mean error: {} points'.format(np.mean(error)))

#cnn = cnn_pce(64)
#data = pd.read_csv('../data/opv.csv')
#smiles_text = data['smiles'][10000:10010]
#pce = np.asarray(data['PCE_calib'][10000:10010])
#
#error = np.zeros(len(smiles_text))
#for i, smile in enumerate(smiles_text):
#   p = cnn.predict([smile])
#   error[i] = abs(pce[i] - p)
#   print('Real: {:.4}, Predicted: {:.4}, Error: {:.4}'.format(pce[i], p, error[i]))
#print(np.mean(error))
