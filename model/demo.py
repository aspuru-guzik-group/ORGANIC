import mol_methods as mm
import numpy as np
from mol_methods import unpad
from custom_metrics import batch_logP

from chemorgan import ChemORGAN

params = {
    'MAX_LENGTH': 16,
    'GEN_ITERATIONS': 1,
    'DIS_EPOCHS': 1,
    'DIS_BATCH_SIZE': 30,
    'GEN_BATCH_SIZE': 30,
    'GEN_EMB_DIM': 32,
    'DIS_EMB_DIM': 32,
    'DIS_FILTER_SIZES': [5, 10, 15],
    'DIS_NUM_FILTERS': [100, 100, 100]
}

model = ChemORGAN('demo', params=params, verbose=False)
model.load_training_set('../data/trainingsets/toy.csv')
model.load_prev_pretraining('checkpoints/demo_pretrain/pretrain_ckpt')
model.set_training_program(['logP'], [50])
model.load_metrics()
model.train()

def evaluate_logP(model):
    samples = [list(x) for x in model.generate_samples(640)]
    smiles = [unpad(''.join(model.ord_dict[x] for x in sample)) for sample in samples]
    scores = batch_logP(smiles)
    print('Maximum value:  {:3f}'.format(np.max(scores)))
    print('Mean value:     {:3f} +/- {:3f}'.format(np.mean(scores), np.std(scores)))
