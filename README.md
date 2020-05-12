# Multiple Sequence Labeling with Linear-Chain Conditional Random Fields
## Research Paper

[Multiple Sequence Labeling with Linear-Chain Conditional Random Fields](https://www.researchgate.net/publication/341322345_Multiple_Sequence_Labeling_with_Linear-Chain_Conditional_Random_Fields)
_____________________________________________________________________
## Overview
This project is based on a novel work by Andrew McCallum, Khashayar Rohanimanesh,
and Charles Sutton, Department of Computer Science University of Massachusetts.
The goal of this work is exploiting complex graphical models to perform multipel task (Part of speech
tagging and Noun-phrase segmentation) simultaneously, instead of using traditional Pipelines.

This work is the extension of our previous work, which we trained a discriminative Hidden Markov Model with averaged percepteron algorithm to gain batter result for part-of-speech tagging task. For this work we have chosen Conditional Random Fields (CRFs), because unlike Hidden Markov Models, CRFs are discriminative in nature, they are robust type of Probabilistic Graphical Models which can be replicated through a sequence to capturing dependencies over a sequence.

The problem of labeling and segmenting sequences of observations arises in many different areas, specially in Natural Language Processing, typically sequence labeling task are done through pipelines. For example, information retrieval task; is often chained into: part-of-speech tagging, shallow parsing and then the main extraction task.

## Required libraries:
[Requirments](./requirements.txt)

## Run
Simply run [experiments](./expriments.py)

There are multiple fusion types defined, feel free to play with parameters

Also both constant and optimized hyperparameters can be used


## Data
Used dataset resides in [dataset](./dataset). CoNLL-2000 shared task data-set, has been chosen for this research; dividing text into syntactically related non-overlapping groups of words, so-called text chunking. The data-set is specifically designed for chunking task, and each token has been tagged as chunk tags and part-of-speech tags. This data consists of the same partitions of the Wall Street Journal corpus (WSJ) as the widely used data for noun phrase chunking. Table 2, shows some information about the dataset that we have used.