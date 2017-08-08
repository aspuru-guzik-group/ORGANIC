# 20 minutes to ORGANIC

During the whole development process, our team had a clear idea: a code, no matter its power, is useless if nobody can run it. That's the reasonwhy we invested so much time in making **ORGANIC** so simple. In this tutorial, I am going to teach you how to use every functionality of the code and unleash the power of machine learning in chemical space.

If it takes more than 20 minutes, please ask for a refund.

## Working on your machine

One great thing of ORGANIC is that it does not require a super-powerful machine; it can be run on a personal computer if you're ready to wait about a week (or even less if you've got a good graphic card). Nonetheless, we want our tutorial examples to run very quick. So I will ask you to open your Jupyter Notebook, write the following lines and wait until we are ready to teach you what they mean.

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
model = ORGANIC('tutorial1')                            
model.load_training_set('../data/trainingsets/toy.csv') 
model.set_training_program(['logP'], [5])               
model.load_metrics()                         
model.train()                               
```


