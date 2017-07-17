# ChemORGAN

ChemORGAN is a molecular discovery tool able to generate new molecules with some previously specified properties. It is based on the Objective-Reinforced Generative Adversarial Networks (ORGAN) architecture, developed in our laboratory in the Department of Chemistry and Chemical Biology, Harvard University. You are invited to read our article about ChemORGAN, and to contact our developers for any issue or interest to collaborate.

This implementation of ChemORGAN is authored by Carlos Outeiral (couteiral@gmail.com), Benjamin Sanchez-Lengeling (beangoben@gmail.com) and Alan Aspuru-Guzik (alan@aspuru.com), 

## Requirements
> tensorflow==1.2
> keras
> future==0.16.0
> numpy
> scipy
> pandas
> matplotlib
> seaborn
> rdkit
> tqdm
> pymatgen

## How to install

First, clone our repo:

```
git clone https://github.com/couteiral/ChemORGAN.git
```

And, it is done!
## How to use

ChemORGAN has been carefully designed to be tremendously simple, while still allowing intense customization of every parameter. This is the minimal functional code:

```python
model = ChemORGAN('OPVs')                   # Loads a ChemORGAN with name 'OPVs'
model.load_training_set('opv.smi')          # Loads the a training set (molecules encoded as SMILES)
model.set_training_program(['PCE'], [50])   # Sets the training program as 50 epochs with the PCE metric
model.load_metrics()                        # Loads all the metrics
model.train()                               # Proceeds with the training
```
