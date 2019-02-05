import sys
import os
from pathlib import Path

# Set environment variables appropriately
ORGANIC_dir = Path(os.getcwd()).parent
sys.path.append(str(ORGANIC_dir))

# ORGANIC imports
import model 
from model.organic import ORGANIC
import mol_methods as mm

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

model = ORGANIC('tutorial1', params=params)                            
model.load_training_set('../data/trainingsets/toy.csv') 
model.set_training_program(['logP'], [5])               
model.load_metrics()                         
#model.load_prev_pretraining(ckpt='<ABSOLUTE PATH TO ORGANIC>/model/checkpoints/tutorial1_pretrain_ckpt') ## Uncomment this line if you have archived pretraining data
#model.load_prev_training(ckpt='<ABSOLUTE PATH TO ORGANIC>/model/checkpoints/tutorial1/checkpoints/tutorial1/tutorial1_4.ckpt') ## Uncomment this line if you have archived training data
model.train()  
for i in model.generate_samples(30):
    print(mm.decode(i, model.ord_dict))
