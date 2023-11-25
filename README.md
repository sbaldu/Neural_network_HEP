# Neural_network_tracking

## Introduction
Nowadays neural networks are widely used in many branches of physics, in particular in particle 
physics.  
The Higgs ML challenge asked to classify in signal and background a set of events simulated based on the 
ATLAS detector. This is exactly the kind of problem that could be effectively tackled using a neural 
network.  
Nowadays there are a lot libraries already written to work with neural networks. The first two that
come to mind, and also the largest ones, are Tensorflow and PyTorch, which are available in Python, C++ 
and other languages.  
These libraries are very thoroughly written and efficient, but from an accademic point of view it is
instructive, in order to really understand the functioning of neural networks, to learn how to write 
them from scratch.  
The goal of this project is to write from scratch in C++ a framework for building neural networks, test 
it with the MNIST dataset and finally use it to tackle the Higgs ML challenge.

## The library
This repository contains a self contained library called `nnhep`, which defines basic neural networks
with dense layers and the forward/back propagation methods for their training.  
The library is header only, so to make it easier to import it and use it.  
The library defines the `Matrix` data format, for the matrices containing the weights of the neural
network. Then the activation functions are defined as functors, so that they can be passed to the
neural network as template parameters, thus being chosen at compile time. The activation functors
are templated on the type of data contained in the neurons of the network.  
The loss functions are defined as functors as well, templated on the type of data contained in the
neurons and on the activation function used.

## Application to the Higgs ML challenge
As said above, the initial goal of this project was to apply the neural network framework built from
scratch to the Higgs ML challenge, which is a simple binary classification task coming from particle
physics with simulated data generated based on the ATLAS detector.  
To make it easier to reproduce the preprocessing and training/validation steps used for tackling this
challenge, a docker image can be built containing all the necessary dependencies, as well as the
data files and used scripts.  
To build the docker image go to the base directory of this repository and use the command:
```
sudo docker build -t . higgs
```
where `higgs` can be substituted with any name (of course).  
Then run the docker image with:
```
sudo docker run -it higgs
```
