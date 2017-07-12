import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import objectives
import pickle
from sklearn.cross_validation import train_test_split

# Import rdkit tools
# this is all setup for the notebook
from IPython.display import HTML
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
# headsup: this import change the behavior of dataframes with mols in them
from rdkit.Chem import PandasTools
# some global configuration of the pandastools


def get_bit_vector_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=4096)
    return [float(i) for i in fingerprint]


if __name__ == '__main__':

    print('Program start:')
    df = pd.read_csv('opv.csv')
    target = 'PCE_calib'
    y_label = 'PCE_test'
    # Normalize data
    pce_mean = df[target].mean()
    pce_std = df[target].std()
    print('Mean and standard deviation are')
    print(pce_mean, pce_std)
    df[y_label] = (df[target] - pce_mean) / pce_std
    y = np.vstack(df[y_label].values).astype('float32')
    # get fingerprints
    df['fps'] = df['smiles'].apply(get_bit_vector_fingerprint)
    X = np.vstack(df['fps'].values).astype('float32')

    # Shuffle dataframe and take a fraction
    #test_df = df.sample(
    #    frac=0.75).reset_index(drop=True)
    
    input_dim = len(X[0])
    print('Writing shapes of data')
    print(X[0],X.shape)
    print(y[:10],y.shape)


    def model(input_dim=input_dim):
        model = Sequential()
        model.add(Dropout(0.2, input_shape=(input_dim,)))
        model.add(Dense(300, activation='relu', init='normal'))
        model.add(Dense(300, activation='relu', init='normal'))
        model.add(Dense(1, activation='linear', init='normal'))
        model.compile(optimizer='adam',
                      loss='mse')
        return model

    callbacks = [ EarlyStopping(
        monitor='val_loss', min_delta=0.05, patience=10, verbose=0, mode='auto')]
    model = model()
    history = model.fit(X, y,
                        shuffle=True,
                        nb_epoch=100,
                        batch_size=10,
                        validation_split=0.33,
                        verbose=1,
                        callbacks=callbacks)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('training.png', dpi=300)

    from keras.models import load_model
    model.save('model_fingerprint.h5')
