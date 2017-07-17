# ChemORGAN
------------------

**ChemORGAN** is an efficient molecular discovery tool, able to create molecules with desired properties. We invite you to check our article about ChemORGAN, and/or contact the developers if you have any issue or are interested in collaborations.

This implementation of ChemORGAN is authored by Carlos Outeiral (couteiral@gmail.com), Benjamin Sanchez-Lengeling (beangoben@gmail.com) and Alan Aspuru-Guzik (alan@aspuru.com), affiliated to Harvard University, Department of Chemistry and Chemical Biology, at the time of release.

## Requirements

- tensorflow==1.2
- future==0.16.0
- rdkit
- keras
- numpy
- scipy
- pandas
- tqdm
- pymatgen

## How to install

To use, just clone our repo:

```
git clone https://github.com/couteiral/ChemORGAN.git
```

And, it is done!
## How to use

ChemORGAN has been carefully designed to be simple, while still allowing full customization of every parameter in the models. Have a look at a minimal example of our code:

```python
model = ChemORGAN('OPVs')                   # Loads a ChemORGAN with name 'OPVs'
model.load_training_set('opv.smi')          # Loads the a training set (molecules encoded as SMILES)
model.set_training_program(['PCE'], [50])   # Sets the training program as 50 epochs with the PCE metric
model.load_metrics()                        # Loads all the metrics
model.train()                               # Proceeds with the training
```


