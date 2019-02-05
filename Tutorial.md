# 20 minutes to ORGANIC

<sup>This tutorial is authored by __Carlos Outeiral__. If you have any doubt, feel free to ask your questions to *carlos@outeiral.net* at any time.</sup>

During the whole development process, our team had a clear idea: a code, no matter its power, is useless if nobody can run it. That's the reasonwhy we invested so much time in making **ORGANIC** so simple. In this tutorial, I am going to teach you how to use every functionality of the code and unleash the power of machine learning in chemical space.

If it takes more than 20 minutes, please ask for a refund.

## Installation

It is recommended to use Anaconda 3 to install ORGANIC. Create a new empty Python 3.6 environment called `ORGANIC` and update the environment using the command `conda env update -f=requirements.yml`.

## Working on your machine

One great thing of ORGANIC is that it does not require a super-powerful machine; it can be run on a personal computer if you're ready to wait about a week (or even less if you've got a good graphic card). Nonetheless, we want our tutorial examples to run very quickly. So I will ask you to open your Jupyter Notebook, write the following lines and wait until we are ready to teach you what they mean.

```python
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
```

This will set the hyperparameters of the internal backend so it can run swiftly on your computer. On the other hand, it will also limit the scope of this tutorial to very small molecules, but it will be more than enough to learn.

## Our very first ORGANIC optimization

The time has come! Let's run the following snippet:

```python
model = ORGANIC('tutorial1', params=params)                            
model.load_training_set('../data/trainingsets/toy.csv') 
model.set_training_program(['logP'], [5])               
model.load_metrics()                         
model.train()                               
```

It will take a few minutes to optimize, depending on the power of your machine. Meanwhile, let's have a look at what every line.

First, we have to define a ORGANIC model. This can be done with just the first line:

```python
model = ORGANIC('tutorial1', params=params)
```

Here, we generate an ORGANIC model with name *tutorial1*, and setting some parameters through the *params* dictionary we just defined. Please forget about the latter for the moment; we'll come back to that shortly.

Once that we've got the object, we load a training set. This step is crucial; even if you won't actually train the model, this step will define the vocabulary and set some parameters, such as the maximum length. So please bear in mind that omiting might (and *will*) result in errors.

```python
model.load_training_set('../data/trainingsets/toy.csv')
```

Then, we define the training program. In this particular case we have selected a training program with 5 epochs of the *log P* (water-octanol partition coefficient) metric. Then, it is important to execute the command *load_metrics()*, which will actually import the metrics to the model.

```python
model.set_training_program(['logP'], [5])
model.load_metrics()
```

And, finally, we indicate to the model that we want to train it:

```python
model.train()
```

If we want, we can genreate and print one batch of samples (which we defined earlier as 30 samples):

```python
model.train()
for i in model.generate_samples(30):
    print(mm.decode(i, model.ord_dict))
```

Seems simple, right? So now, let us try to get a little bit deeper.

## The elements of optimization

There are three main elements that every property optimization requires:

* An optimization model (in this case, the ORGANIC model)
* A training set
* A metric

The ORGANIC model is what we're learning to use, and the training set (just a list of molecules, in the SMILES encoding, that have properties similar to those that we'd like to find) can be easily loaded with the *load_training_set()* command, as we just explained. So let's talk about the metric.

The metric is some kind of procedure that, given a molecule (as always, encoded in a SMILES string), is able to assign a score in the interval [0, 1] that tells the model how well it is performing. As we shown in our research paper, given a good metric, ORGANIC is able to optimize the molecular distribution to the desired parameters. The problem is, however, how to specify an appropriate metric.

### Built-in metrics



### Metric remapping

### User-defined metrics

## Control over hyperparameters
