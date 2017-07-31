"""

GP METRICS
====================

This module allows the use of Gaussian Processes for property
prediction. It is basically a wrapper of GPmol within the
ChemORGAN architecture. It has been entirely programmed by
Carlos Outeiral.

The author whishes to thank Rodolfo Ferro for sharing his
library in the alpha phase.
"""

import GPmol
import tensorflow as tf
import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem


class GaussianProcess(object):
    """
    Class for handling gaussian processes.
    """

    def __init__(self, label, nBits=4096):
        """Initializes the model.

        Arguments
        -----------

            - label. Identifies the property predicted
               by the gaussian process.

            - nBits. Refers to the number of bits in which
               the Morgan fingerprints are encoded. By
               default, 4096.

        Note
        -----------

            Using the same label for different Machine Learning
            models in the same run might lead to crashes.

        """

        self.name = label
        self.nBits = nBits
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = GPmol.model.GPMol()

    def predict(self, smiles):
        """
        Computes the predictions for a batch of molecules.

        Arguments
        -----------

            - smiles. Array or list containing the
            SMILES representation of the molecules.

        Returns
        -----------

            A list containing the predictions.

        """

        with self.graph.as_default():
            input_x = self.computeFingerprints(smiles)
            input_x = np.reshape(input_x, (-1, self.nBits))
            return self.model.predict(input_x)[0]

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

            Mean squared error.

        """

        with self.graph.as_default():
            input_x = self.computeFingerprints(train_x)
            predicted = self.model.predict(input_x)[0]
            error = (predicted - train_y)**2
            return np.mean(error)

    def train(self, train_x, train_y):
        """
        Trains the model

        Arguments
        -----------

            - train_x. Array or list containing the
               SMILES representation of the molecules.

            - train_y. The real values of the desired
               properties.

        Arguments
        -----------

            A .json file will be created in the main dir,
            named according to the model's label.

        """

        with self.graph.as_default():

            input_x = pd.DataFrame({'smiles': pd.Series(train_x),
                                    'property': pd.Series(train_y)})

            preproc = GPmol.preprocessor.Preprocessor(input_x)
            preproc.addFp(duplicates=True, args={'nBits': self.nBits,
                                                 'radius': 12})
            preproc.addTarget(target='property')
            kernel = GPmol.kernels.Tanimoto(input_dim=self.nBits)

            self.model = GPmol.model.GPMol(kernel=kernel, preprocessor=preproc)
            self.model.train()

            self.model.save('../data/gps/{}.json'.format(self.name))

    def load(self, file):
        """
        Loads a previously trained model.

        Arguments
        -----------

            - file. A string pointing to the .json file.

        """

        with self.graph.as_default():
            self.model.loadJSON(file)

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
