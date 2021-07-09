---
author: Rick Stevens
title: CANDLE Project KPP-1 Verification
---

# Science Challenge Problem Description

The CANDLE challenge problem is to solve large-scale machine learning (ML)
problems for two cancer-related pilot applications: (1) predicting drug
interactions and (2) predicting cancer phenotypes and treatment trajectories
from patient documents.  The CANDLE pilot application that involves predicting
the state of molecular dynamics simulations is treated as a stretch goal.  The
CANDLE project has specific strategies to address these challenges. For the
drug response problem, unsupervised ML methods are used to capture the
complex, nonlinear relationships between the properties of drugs and the
properties of the tumors to predict response to treatment and therefore
develop a model that can provide treatment recommendations for a given tumor.
For the treatment strategy problem, semisupervised ML is used to automatically
read and encode millions of clinical reports into a form that can be computed
upon. Each problem requires a different approach to the embedded learning
problem, all of which are supported with the same scalable deep learning code
in CANDLE.

The challenge for exascale manifests in the need to train large numbers of
models. One need inherent to each of the pilot applications is producing
high-resolution models that cover the space of specific predictions
individualized in the precision medicine sense. For example, consider training
a model that is specific to a certain drug and individual cancer. Starting
with 1000 different cancer cell lines and 1000 different drugs, a
leave-one-out strategy to create a high-resolution model for all
drug-by-cancers requires approximately 1 million models. Yet, these models are
similar enough that using a transfer learning strategy in which weights are
shared during training in a way that avoids information leakage can
significantly reduce the time needed to train a large set of models.

In practice, speedup related to weight sharing can be discussed in the context
of the challenge problem in terms of work actually done and the naive work
done.  Consider work actually done $W_{D}$ as being the number of jobs $J$
multiplied by the actual number of epochs $E$ trained for all stages $s$:
$$
  W_{D} = \sum_{1}^{s}JE\:.
$$
A stage is a discrete and naive work done $W_{N}$ being the number of jobs in
the last stage multiplied by the number of epochs needed for a model to
converge $E_{c}$,:
$$
  W_{n} = J_{s}E_{c}\:.
$$
The team can then talk about speedup as being the ratio $W_{N}/W_{D}$.

Several parameters exist when considering accelerated model training via the
transfer of weights. These include how many transfer events, how to partition
the input data, and how many epochs before a transfer occurs.  Additional
considerations include what weights to transfer and whether to allow those
weights to be updated in subsequent models.

Table: Challenge problem details

------------------------------------------------------------------------------------
Functional requirement     Minimum criteria
-------------------------  ---------------------------------------------------------
Physical phenomena and     Deep learning neural networks for cancer: feed-forward,
associated models          auto-encoder, recurrent neural networks.


Numerical approach,        Gradient descent of model parameters, optimization
and associated models      of loss
                           function, network activation function;
                           regularization, and learning rate scaling methods.

Simulation details         Large-scale ML solutions will be computed
                           for the three cancer pilots:

$\phantom{continue}$       *Pilot1*: Leave-one-out cross validation of roughly
                           1000 drugs by 1000 cell lines. This
                           involves roughly 1 million models.  Partition the
                           drugs and cell lines into $n$ sets and train with
                           those for $e$ epochs, then transfer the weights $w$
                           to the next set of models, expanding the number of
                           models in each iteration. Each of the models at
                           iteration $i$ can be safely (i.e., avoiding
                           information leakage) used to seed models for
                           iteration $i+1$, where the set of drugs and cell
                           lines in the $i+1$ validation set was not in the
                           training set of the model at iteration $i$.

$\phantom{continue}$       *Pilot2*: State the identification and classification of
                           one or more RAS proteins binding to a lipid
                           membrane; prediction over time of clustering
                           behavior of key lipid populations that leads to RAS
                           protein binding. RAS proteins are represented in
                           sufficient resolution to model all pairwise
                           interactions within and between proteins. Lipid
                           membranes are represented as continuous density
                           fields of tens of species of lipid concentration.
                           Predictions are trained on the cross product of
                           thousands of simulations, each of which is
                           thousands of time steps, over multiple protein
                           configurations, and performed for a large range of
                           different concentrations.

$\phantom{continue}$       *Pilot3*: Predicting cancer phenotypes and patient
                           treatment trajectories from millions of cancer
                           registry documents. Thousands of multitask
                           phenotype classification models will be built from
                           defined combinations of descriptive terms extracted
                           from 10,000 curated text training sets. To
                           accelerate model training, the team will use a
                           transfer learning scheme with weight sharing during
                           training.

Demonstration calculation  The computation performed at scale will be standard
requirements               neural network computations, matrix multiplies, 2D
                           convolutions, pooling, and so on. These will be
                           specifically defined by the models chosen to
                           demonstrate transfer learning. The computations
                           performed at scale will require weight sharing.
------------------------------------------------------------------------------------

# Demonstration Calculation

## Facility Requirements

*List significant I/O, workflow, and/or third party library requirements that need facility support (Note: this is outside of standard MPI and ECP-supported software technologies)*

## Input description and Execution

*Describe your problem inputs, setup, estimate of resource allocation, and runtime settings*

## Problem Artifacts

*Describe the artifacts produced during the simulation, e.g., output files, and the mechanisms that are used to process the output to verify that capabilities have been demonstrated.*

## Verification of KPP-1 Threshold

*Give evidence that*

1. *The FOM measurement met threshold ($>50$)*
2. *The executed problem met challenge problem minimum criteria*
