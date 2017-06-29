
#
# DEPRECATED FILE
#

import sys
import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem.AllChem as Chem
from rdkit import DataStructs
from rdkit.Chem import PandasTools
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn

def model_cnn(features, targets, mode):

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

class cnn_pce(object):

    def __init__(self):
        self.cnn = learn.Estimator(model_fn=model_cnn, model_dir='../data/cnn_pce/')

    def predict(self, smile):

        mol = Chem.MolFromSmiles(smile)
        fp = np.asarray(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024))
        data_x = np.array([fp])

        def input_predict(features):

            def _input_fn():
                all_x = tf.constant(features, shape=features.shape, dtype=tf.float32)
                return all_x

            return _input_fn

        predictions = self.cnn.predict(input_fn=input_predict(data_x))
        data_y = self.itToList(predictions)
        return data_y

    def itToList(self, predictions):
        return sum([p['pce'] for p in predictions])

class cnn_homolumo(object):

    def __init__(self):
        self.cnn = learn.Estimator(model_fn=model_cnn, model_dir='./check32')

    def train(self):

        data = pd.read_csv('opv.csv')
        smiles_text = data['smiles'][:10]
        homo = data['homo_calib'][:10]
        lumo = data['lumo_calib'][:10]
        hl = lumo - homo 

        smiles = [Chem.MolFromSmiles(smile) for smile in tqdm(smiles_text)]
        fps = np.asarray([Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024) for mol in tqdm(smiles)])
        train_x = np.asarray(fps)
        train_y = np.asarray(hl)

        def batched_input_fn(features, labels, batch_size):
            def _input_fn():
                all_x = tf.constant(features, shape=features.shape, dtype=tf.float32)
                all_y = tf.constant(labels, shape=labels.shape, dtype=tf.float32)
                sliced_input = tf.train.slice_input_producer([all_x, all_y])
                return tf.train.batch(sliced_input, batch_size=batch_size)
            return _input_fn

        self.cnn.fit(input_fn = batched_input_fn(train_x, train_y, 1000), steps=10000)

    def predict(self, smile):

        mol = Chem.MolFromSmiles(smile)
        fp = np.asarray(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024))
        data_x = np.array([fp])

        def input_predict(features):

            def _input_fn():
                all_x = tf.constant(features, shape=features.shape, dtype=tf.float32)
                return all_x

            return _input_fn

        predictions = self.cnn.predict(input_fn=input_predict(data_x))
        data_y = self.itToList(predictions)
        return data_y

    def itToList(self, predictions):
        return sum([p['pce'] for p in predictions])


cnn = cnn_homolumo()
cnn.train()
