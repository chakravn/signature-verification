# Signature verification using Siamese Networks and Amazon Sagemaker

### Few short learning with Siamese Networks
This notebook tries to classify images using Siamese Networks proposed by **Gregory et. al**, in his paper [Siamese Neural Networks for One-shot Image Recognition](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf), to solve the **One shot learning** problem.

This notebook uses a deep convolutional neural network (CNN) to extract features from input images. Keras is used for implementing the CNN.

### Few shot learning

One of the main requisites of highly accurate deep learning models is large amount of data. The set of hyperparameters a Deep Model need to be tuned are very large, and the amount of data needed to get the right set of value for these hyperparameters is also large.

But what if we need an automated system, which can successfully classify images to various classes given the data for each image class is quite less.

**Few shot learning** is such a problem. We can **Few shot learning** as a problem to classify data into K classes where each class has only few examples. The paper written by [Gregory et. al](http://www.cs.utoronto.ca/~gkoch/fil

## Siamese networks 

Siamese networkis a Deep Nueral Network architecture proposed by **Gregory et. al** in his paper [Siamese Neural Networks for One-shot Image Recognition](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf), the paper proposes an architecture where using Convolutional Nueral Networks one can tackle the problem of One Shot Learning.

The model aims to solve the basic problem of image verification, given that we have very few samples of image of each class or category

The models aims to learn the embeddings of 2 separate images fed into the Nueral Network, the two embeddings are used to calculate the L1 distance between the 2 embeddings. Once the distance embedding metric is calculated, the embedding is fed into a sigmoid unit which by the magic of back propogation, learns the correct set of hyperparameters to carry out the image verification.

## Follow the steps as below.
1. Create a Sagemaker notebook instance with the instance type as 'ml.t2.medium'
2. Once the Notebook instance is "In Service", clone this git repo in the Jupyter environment
3. Run "Data_prep_v2.ipynb" notebook to prepare the training/validation dataset
4. Run "Signature_Verification_Sagemaker.ipynb" notebook to train and deploy the model with Amazon Sagemaker followed by Inference
5. Refer to "train.py" script used for training the model


