# ORGANIC

**ORGANIC** ([Objective-Reinforced Generative Adversarial Network for Inverse-design Chemistry](https://chemrxiv.org/articles/ORGANIC_1_pdf/5309668)) is an efficient molecular generation tool, able to create molecules with desired properties. It has a user-oriented interface, and doesn't require a HPC cluster. Feel free to check our article about ORGANIC, and/or contact the developers if you have any issue or are interested in collaborations.

The concepts behind this algorithm is in the preprint <https://chemrxiv.org/articles/ORGANIC_1_pdf/5309668>.
This implementation is authored by **Carlos Outeiral** (carlos@outeiral.net), **Benjamin Sanchez-Lengeling** (beangoben@gmail.com), **Gabriel Guimaraes** (gabrielguimaraes@college.harvard.edu) and **Alan Aspuru-Guzik** (alan@aspuru.com), affiliated to Harvard University, Department of Chemistry and Chemical Biology, at the time of release.

## Installation
### How-to
To install, just clone our repo:

```
git clone https://github.com/aspuru-guzik-group/ORGANIC.git
```

And, it is done!

### Requirements

- tensorflow==1.2
- future==0.16.0
- rdkit
- keras
- numpy
- scipy
- pandas
- tqdm
- pymatgen

## How to use

ORGANIC has been carefully designed to be simple, while still allowing full customization of every parameter in the models. Have a look at a minimal example of our code:

```python
model = ORGANIC('OPVs')                     # Loads a ORGANIC with name 'OPVs'
model.load_training_set('opv.smi')          # Loads the training set (molecules encoded as SMILES)
model.set_training_program(['PCE'], [50])   # Sets the training program as 50 epochs with the PCE metric
model.load_metrics()                        # Loads all the metrics
model.train()                               # Proceeds with the training
```

If you are interested, we encourage you to read our '20 minutes to ORGANIC' tutorial, where you can learn all (literally, all) the functionalities of ORGANIC.
