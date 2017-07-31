"""

NN METRICS
====================

Support for metrics based in neural networks. To the
date of release, support for Keras models is included.

This module has been entirely created by Carlos Outeiral.
"""

import numpy as np
import rdkit.Chem.AllChem as Chem
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras_tqdm import TQDMCallback


class KerasNN(object):
    """
    Class for handling Keras neural network models.
    """

    def __init__(self, label, nBits=4096):
        """Initializes the model.

        Arguments
        -----------

            - label. Identifies the property predicted
               by the neural network.

            - nBits. Refers to the number of bits in which
               the Morgan fingerprints are encoded. By
               default, 4096.

        """

        self.label = label
        self.graph = tf.Graph()
        self.nBits = nBits

    def predict(self, smiles, batch_size=100):
        """
        Computes the predictions for a batch of molecules.

        Arguments
        -----------

            - smiles. Array or list containing the
            SMILES representation of the molecules.

            - batch_size. Optional. Size of the batch
            used for computing the properties.

        Returns
        -----------

            A list containing the predictions.

        """

        with self.graph.as_default():
            input_x = self.computeFingerprints(smiles)
            return self.model.predict(input_x, batch_size=batch_size)

    def evaluate(self, train_x, train_y):
        """
        Evaluates the accuracy of the method.

        Arguments
        -----------

            - train_x. Array or list containing the
            SMILES representation of the molecules.
            - train_y. The real values of the desired
            properties.

        Returns
        -----------

            Test loss.

        """

        with self.graph.as_default():
            input_x = self.computeFingerprints(train_x)
            return self.model.evaluate(input_x, train_y, verbose=0)

    def train(self, train_x, train_y, batch_size, nepochs,
              earlystopping=True, min_delta=0.001):
        """
        Trains the model

        Arguments
        -----------

            - train_x. Array or list containing the
               SMILES representation of the molecules.

            - train_y. The real values of the desired
               properties.

            - batch_size. The size of the batch.

            - nepochs. The maximum number of epochs.

            - earlystopping. Boolean specifying whether early
            stopping will be used or not. True by default.

            - min_delta. If earlystopping is True, the variation
            on the validation set's value which will trigger
            the stopping.

        """

        with self.graph.as_default():

            self.model = Sequential()
            self.model.add(Dropout(0.2, input_shape=(self.nBits,)))
            self.model.add(BatchNormalization())
            self.model.add(Dense(300, activation='relu',
                                 kernel_initializer='normal'))
            self.model.add(Dense(300, activation='relu',
                                 kernel_initializer='normal'))
            self.model.add(Dense(1, activation='linear',
                                 kernel_initializer='normal'))
            self.model.compile(optimizer='adam', loss='mse')

            input_x = self.computeFingerprints(train_x)

            if earlystopping is True:
                callbacks = [EarlyStopping(monitor='val_loss',
                                           min_delta=min_delta,
                                           patience=10,
                                           verbose=0,
                                           mode='auto'),
                             TQDMCallback()]
            else:
                callbacks = [TQDMCallback()]

            self.model.fit(input_x, train_y,
                           shuffle=True,
                           epochs=nepochs,
                           batch_size=batch_size,
                           validation_split=0.1,
                           verbose=2,
                           callbacks=callbacks)

            self.model.save('../data/nns/{}.h5'.format(self.label))

    def load(self, file):
        """
        Loads a previously trained model.

        Arguments
        -----------

            - file. A string pointing to the .h5 file.

        """

        with self.graph.as_default():
            self.model = load_model(file)

    def computeFingerprints(self, smiles):
        """
        Computes Morgan fingerprints

        Arguments
        -----------

            - smiles. An array or list of molecules in
               the SMILES codification.

        Returns
        -----------

            A numpy array containing Morgan fingerprints
            bitvectors.

        """

        if isinstance(smiles, str):
            smiles = [smiles]

        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fps = [Chem.GetMorganFingerprintAsBitVect(
            mol, 12, nBits=self.nBits) for mol in mols]
        bitvectors = [self.fingerprintToBitVect(fp) for fp in fps]
        return np.asarray(bitvectors)

    def fingerprintToBitVect(self, fp):
        """
        Transforms a Morgan fingerprint to a bit vector.

        Arguments
        -----------

            - fp. Morgan fingerprint

        Returns
        -----------

            A bit vector.

        """
        return np.asarray([float(i) for i in fp])
