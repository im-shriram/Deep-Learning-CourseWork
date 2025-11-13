# Deep Learning Notes and Projects

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook%2FLab-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

A curated collection of Jupyter notebooks exploring core deep learning concepts, practical training techniques, and applied projects. The repository emphasizes PyTorch, with occasional Keras/TensorFlow demonstrations, covering topics from perceptrons to CNNs/RNNs, regularization, transfer learning, and NLP with Transformers.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Topics Covered](#topics-covered)
- [How to Use the Notebooks](#how-to-use-the-notebooks)
- [Dependencies](#dependencies)
- [Recommended .gitignore](#recommended-gitignore)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This repository organizes learning materials and mini-projects into topic-focused folders. Each notebook includes hands-on code, visualizations, and explanatory notes. Use the file structure below to jump directly to the content you need.

## Quick Start
- Clone: `git clone <your-repo-url>`
- Create environment: `python -m venv .venv && .venv\\Scripts\\Activate.ps1` (Windows PowerShell)
- Install basics: `pip install torch torchvision torchaudio numpy matplotlib scikit-learn jupyter transformers datasets`
- Launch Jupyter: `jupyter lab` or `jupyter notebook`
- Optional GPU: Install the CUDA build of PyTorch from the official site if you have a compatible GPU (see `https://pytorch.org/get-started/locally/`).

## Repository Structure
(Names and paths preserved exactly as in the filesystem. Environment and checkpoint folders are intentionally omitted for clarity.)

- Root
  - [Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow.pdf](Hands-On%20Machine%20Learning%20with%20Scikit-Learn,%20Keras,%20and%20Tensorflow.pdf) — Reference book.

- ANN
  - [ANN in production.ipynb](ANN/ANN%20in%20production.ipynb) — Deploying and tracking ANN models in production-like setups.

- Activation Functions
  - [Activation Functions.ipynb](Activation%20Functions/Activation%20Functions.ipynb) — Overview and comparison of common activation functions.
  - [GELU, Swish and Mish activation functions.ipynb](Activation%20Functions/GELU,%20Swish%20and%20Mish%20activation%20functions.ipynb) — Modern smooth activations and their behavior.
  - [PyTorch.ipynb](Activation%20Functions/PyTorch.ipynb) — Activation demos and implementations in PyTorch.

- Backpropagation
  - [Classification.ipynb](Backpropagation/Classification.ipynb) — Backpropagation on classification tasks.
  - [Regression.ipynb](Backpropagation/Regression.ipynb) — Backpropagation on regression tasks.

- Batch Normalization
  - [Batch Normalization.ipynb](Batch%20Normalization/Batch%20Normalization.ipynb) — Concept, benefits, and caveats of BatchNorm.
  - [PyTorch Implemantation.ipynb](Batch%20Normalization/PyTorch%20Implemantation.ipynb) — Applying BatchNorm in PyTorch models.

- CNN
  - [CNN in production.ipynb](CNN/CNN%20in%20production.ipynb) — Practical deployment aspects of CNNs.
  - [Image Classification using PyTorch.ipynb](CNN/Image%20Classification%20using%20PyTorch.ipynb) — End-to-end PyTorch image classification.
  - [LENET-5 Architecture.ipynb](CNN/LENET-5%20Architecture.ipynb) — Classic LeNet-5 architecture walkthrough.
  - [Padding & Strids.ipynb](CNN/Padding%20&%20Strids.ipynb) — Effects of padding and strides on feature maps.
  - [Pooling Layer.ipynb](CNN/Pooling%20Layer.ipynb) — Max/avg pooling behavior and impact.

- Data Augumentation
  - [Data Augumentation.ipynb](Data%20Augumentation/Data%20Augumentation.ipynb) — Augmentation techniques and their impact.

- Data Scaling
  - [Feature Scaling.ipynb](Data%20Scaling/Feature%20Scaling.ipynb) — Normalization/standardization strategies and when to use them.

- Early Stopping
  - [Early Stopping.ipynb](Early%20Stopping/Early%20Stopping.ipynb) — Preventing overfitting with early stopping.

- Encoder Decoder (seq2seq)
  - [Machine Translation.ipynb](Encoder%20Decoder%20(seq2seq)/Machine%20Translation.ipynb) — Seq2seq architecture applied to machine translation.

- Functional Models
  - [Multi-Input ANN.ipynb](Functional%20Models/Multi-Input%20ANN.ipynb) — Architectures with multiple input streams.
  - [Multi-Output ANN.ipynb](Functional%20Models/Multi-Output%20ANN.ipynb) — Models producing multiple outputs.
  - [Skip Connections.ipynb](Functional%20Models/Skip%20Connections.ipynb) — Residual/skip connections in practice.

- Hyperparameter Tuning
  - [Hyperparameter Tuning using Keras Tuner.ipynb](Hyperparameter%20Tuning/Hyperparameter%20Tuning%20using%20Keras%20Tuner.ipynb) — Automated hyperparameter search demo.

- LSTM
  - [Next Word Prediction using LSTM.ipynb](LSTM/Next%20Word%20Prediction%20using%20LSTM.ipynb) — Language modeling with LSTM.
  - [PyTorch LSTM Implementation.ipynb](LSTM/PyTorch%20LSTM%20Implementation.ipynb) — Building LSTM modules in PyTorch.

- Loss Functions
  - [Loss_Functions.ipynb](Loss%20Functions/Loss_Functions.ipynb) — Overview of common losses and their gradients.

- Natural Language Processing with Transformers and HuggingFace
  - [01 - Introduction and HuggingFace Pipelines.ipynb](Natural%20Language%20Processing%20with%20Transformers%20and%20HuggingFace/01%20-%20Introduction%20and%20HuggingFace%20Pipelines.ipynb) — Quickstart with HuggingFace pipelines.
  - [02 - Sentement Analysis with DistilBERT.ipynb](Natural%20Language%20Processing%20with%20Transformers%20and%20HuggingFace/02%20-%20Sentement%20Analysis%20with%20DistilBERT.ipynb) — Sentiment classification using DistilBERT (name preserved).

- Optimizers
  - [Exponential Weighted Moving Average.ipynb](Optimizers/Exponential%20Weighted%20Moving%20Average.ipynb) — EWMA for smoothing metrics.
  - [Optimizers.ipynb](Optimizers/Optimizers.ipynb) — SGD, Momentum, Adam, and popular variants.

- Perceptron
  - [Multi Layer Perceptron.ipynb](Perceptron/Multi%20Layer%20Perceptron.ipynb) — MLP basics and training.
  - [Perceptron.ipynb](Perceptron/Perceptron.ipynb) — Single-layer perceptron fundamentals.

- PyTorch Basics
  - [.gitignore](PyTorch%20Basics/.gitignore) — Ignores environment and checkpoint files within this folder.
  - [AutoGrad.ipynb](PyTorch%20Basics/AutoGrad.ipynb) — Autograd mechanics and custom gradients.
  - [Dataset and DataLoader - The What.ipynb](PyTorch%20Basics/Dataset%20and%20DataLoader%20-%20The%20What.ipynb) — Concepts for datasets/dataloaders.
  - [Dataset and DataLoader - The Why.ipynb](PyTorch%20Basics/Dataset%20and%20DataLoader%20-%20The%20Why.ipynb) — Motivation and best practices.
  - [Model Training Pipeline.ipynb](PyTorch%20Basics/Model%20Training%20Pipeline.ipynb) — Structuring training loops.
  - [Tensors.ipynb](PyTorch%20Basics/Tensors.ipynb) — Tensor operations, shapes, and dtypes.
  - [nn Module.ipynb](PyTorch%20Basics/nn%20Module.ipynb) — Building with `torch.nn`.

- RNN
  - Deep (Stacked) RNN
    - [Deep RNN's.ipynb](RNN/Deep%20(Stacked)%20RNN/Deep%20RNN's.ipynb) — Stacked RNN architectures and training.
  - Embeddings
    - [Tokenization, Padding and Embeddings.ipynb](RNN/Embeddings/Tokenization,%20Padding%20and%20Embeddings.ipynb) — Text preprocessing and embeddings.
  - Question Answering System - Mini Project
    - [PyTorch Question Answering System using SimpleRNN.ipynb](RNN/Question%20Answering%20System%20-%20Mini%20Project/PyTorch%20Question%20Answering%20System%20using%20SimpleRNN.ipynb) — QA pipeline with SimpleRNN.
    - [PyTorch SimpleRNN - Question Answering System.ipynb](RNN/Question%20Answering%20System%20-%20Mini%20Project/PyTorch%20SimpleRNN%20-%20Question%20Answering%20System.ipynb) — Alternate QA implementation.
  - Sentement Classification Mini Project
    - [Sentement Analysis - Embeddings.ipynb](RNN/Sentement%20Classification%20Mini%20Project/Sentement%20Analysis%20-%20Embeddings.ipynb) — Sentiment classification with embeddings (name preserved).
    - [Sentement Analysis - Integer Encoding.ipynb](RNN/Sentement%20Classification%20Mini%20Project/Sentement%20Analysis%20-%20Integer%20Encoding.ipynb) — Sentiment via integer encoding (name preserved).

- Regularization
  - [Dropout Layer Analysis.ipynb](Regularization/Dropout%20Layer%20Analysis.ipynb) — Effect of dropout on generalization.
  - [Max-Norm Regularization.ipynb](Regularization/Max-Norm%20Regularization.ipynb) — Constraining weights to reduce overfitting.
  - [Regularization.ipynb](Regularization/Regularization.ipynb) — L1/L2 and other techniques overview.

- Transfer Learning
  - [Feature Extraction with Data Augumentation.ipynb](Transfer%20Learning/Feature%20Extraction%20with%20Data%20Augumentation.ipynb) — Frozen base + augmented features.
  - [Feature Extraction without Data Augumentation.ipynb](Transfer%20Learning/Feature%20Extraction%20without%20Data%20Augumentation.ipynb) — Feature extraction baseline.
  - [Finetuning.ipynb](Transfer%20Learning/Finetuning.ipynb) — Unfreeze and fine-tune strategy.
  - [Transfer Learning of ANN model.ipynb](Transfer%20Learning/Transfer%20Learning%20of%20ANN%20model.ipynb) — Transfer learning applied to ANN.
  - [Transfer Learning.ipynb](Transfer%20Learning/Transfer%20Learning.ipynb) — General transfer learning primer.

- Vanishing Gradient Problem
  - [Vanishing Gradient Problem in ANN.ipynb](Vanishing%20Gradient%20Problem/Vanishing%20Gradient%20Problem%20in%20ANN.ipynb) — Causes and mitigations in training.

- Weight Initialization Techniques
  - [Disadvantages - Zero and Constant value initialization.ipynb](Weight%20Initialization%20Techniques/Disadvantages%20-%20Zero%20and%20Constant%20value%20initialization.ipynb) — Why naive initializations fail.
  - [PyTorch.ipynb](Weight%20Initialization%20Techniques/PyTorch.ipynb) — Weight initialization in PyTorch.
  - [Weight Initialization Techniques.ipynb](Weight%20Initialization%20Techniques/Weight%20Initialization%20Techniques.ipynb) — Xavier/He and other strategies.

## Topics Covered
- Perceptron and ANN fundamentals
- CNN architecture and training
- RNN/LSTM for sequence modeling
- NLP with HuggingFace Transformers
- Training mechanics: optimizers, loss functions, regularization, early stopping
- Practical techniques: data scaling, augmentation, transfer learning, weight initialization

## How to Use the Notebooks
- Open any `.ipynb` directly in Jupyter Lab or Notebook.
- Run cells sequentially; most notebooks are self-contained.
- If a notebook expects a dataset, the first cells typically include download or path instructions.

## Dependencies
- Core requirements:

```
torch
torchvision
torchaudio
numpy
matplotlib
scikit-learn
jupyter
transformers
datasets
```

- Optional requirements (for specific notebooks):

```
tensorflow
keras
keras-tuner
seaborn
pandas
```

Install only what you need to keep environments lightweight.

## Recommended .gitignore
- Environment folders: `.venv/`, `venv/`, `env/`, `myenv/`
- Jupyter checkpoints: `.ipynb_checkpoints/`
- OS files: `.DS_Store`, `Thumbs.db`
- Experiment artifacts: `runs/`, `wandb/`, `logs/`, `checkpoints/`
- Model weights: `*.pt`, `*.pth`

## Contributing
- PRs welcome: keep notebook names descriptive and consistent with topic folders.
- Prefer small, focused notebooks; include brief context and dependencies.

## License
- MIT (or your preferred license). Add a `LICENSE` file if you want this formalized.

## Acknowledgements
- PyTorch, HuggingFace, TensorFlow/Keras, scikit-learn, and the open-source community.
