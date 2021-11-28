[<img src="https://user-images.githubusercontent.com/43852207/140623269-642a3d0f-ee76-4194-aa56-b7305617de9d.jpeg" width="170" height="170">](https://www.brizzy.eu) [<img src="https://user-images.githubusercontent.com/43852207/140623238-20bbde3c-7dfe-4965-b773-331e6e906069.jpg" width="170" height="170">](https://www.maastrichtuniversity.nl/education/bachelor/data-science-and-artificial-intelligence)

# Multinomial Classification Of Human Sleep Physiological Signal Data
### Maastricht University - BSc Thesis in Collaboration with Nomics Sleep & Care
> September 2021 -> December 2021


> Author: Clément Detry

## Introduction

### Company

Nomics Sleep & Care is a Belgian company known for developing electronic devices used for sleep apnea diagnosis. The great advantage of using their technologies is the way patients are treated. Indeed, they have the possibility to test themselves at home, in their own environment, with only 2 facial sensors. This is a big difference considering the costs and the difficulty of obtaining significant results in clinical tests. In short, Nomics has more than 25 clinical papers, more than 10 years of specialization in sleep diagnosis, more than 15,000 patients by 2018 and at least 90% true diagnoses.

### Problem Statement

One of the main problems in this testing setup is the lack of direct feedback during the patient examinations. Sometimes, this could lead to invalid analyses. For example, when a sensor is disconnected or moved for too long overnight. In the worst cases, patients must be retested if the total diagnosis time is outside certain conditions. Unfortunately, each uploaded graph must be manually reviewed by a Nomics operator, which slows down the process. Hence, the objective is to have an algorithm that will automatically process the signal beforehand to be able to tell the doctor if his patient has been correctly examined or not. 

First, the task is to deal exclusively with a two-class classification problem, whose objective is to provide a learning model capable of identifying valid and invalid signal regions.
According to the obtained result, a decision must be taken on the overall quality of the signal, following this simplified rule. If there is a continuous 4-hour block or at least two continuous 2-hour blocks of validity, the entire analysis will
be retained for diagnosis. Next, a third class of interest is introduced to detect areas in the signal where the patient is not sleeping. This new type of signal is considered valid. Still, it must be removed from the total amount of validity of
the signal to have a more pertinent statement for the decision making of the entire analysis.

## Methodology

One solution to address this problem would be to use machine learning to replace this redundant human task. Recently, deep learning methods such as CNNs, which use one-dimensional convolutional neural networks or recurrent neural networks such as LSTMs, have been shown to provide state-of-the-art results for difficult time series classification tasks with little or no data feature engineering.

### Dataset Prepocessing

In order to have consistent input data for machine learning, it is advisable to find a way to obtain samples of the same size. Indeed, time series are not of the same length because the recording time can vary. One way to do this is to segment the data into windows of the same size with their corresponding label. The following image summarizes all the data preprocessing before training the model.

<p align="center">
  <img width="472" alt="Screenshot 2021-11-27 at 23 03 45" src="https://user-images.githubusercontent.com/43852207/143721891-3bf19bda-739b-4464-8955-c64444e37a60.png">
</p>

The entire implementation can be found inside the _preprocessing.py_ file. Now that the data is ready to be processed, the deep learning models can come into play.

### Convolutional Neural Network (CNN)

The first approach is to use the well-known CNN classification technique. In the context of time series data, the convolutional neural network process can be divided into 4 main steps.

1. It first creates a kernel of time series width length k that will only move in one direction, from the start to the end of the time series.
2. Then, the convolutional layer is created after performing convolution with the kernel on the input data. Its main goal is to filter the information and to extract the most significant features.
3. Next, the dimensionality space is reduced using gloabel max pooling to fit the data in into a NN.
4. Finally, the data goes into a fully connected layer to perform classification.

<p align="center">
  <img src="https://user-images.githubusercontent.com/43852207/140625099-a3b752d7-290f-4f2c-9abb-baa92fc8834b.png" width="400" height="200">
</p>

The main pros of using the CNN for such a classification is its noise resistance as well as its ability to extract deep features independent from time. The CNN model is implemented using [Tensorflow Keras API](https://www.tensorflow.org/guide/keras/sequential_model) sequential architecture inside the _cnn_model.py_ file. With the objective of achieving the best results, a version of the keras tuner is included to find the best hyperparameters after performing random search.

### Recurrent Neural Network (RNN)

RNN is the second deep learning approach used for this study. And more precisely, the Long-Short Term Memory variant. LSTM is a deep recurrent neural network architecture used for the classification of time series data. Time frequency and space properties of time series are presented here as a robust framework for LSTM treatment of long sequential data. It is distinguished from a regular feedforward network because its architecture includes a feedback loop as it can be seen in the figure below. A particular unit, called a memory cell, retains past information for a more extended period of time to make an accurate prediction.

<p align="center">
  <img src="https://user-images.githubusercontent.com/43852207/143721962-273387a8-fe2a-4ec0-8dfc-2a85dde50642.png" width="250" height="250">
</p>


The advantage of using the LSTM model is its ability to learn and remember the information over long sequences of input data. It also does not require any domain expertise to manually engineer input features. In the same way as for the CNN, the architecture used for the LSTM comes from [Tensorflow Keras API](https://www.tensorflow.org/guide/keras/sequential_model) inside the _lstm_model.py_ file and is complemented with a optimization tuner.

## Experiments, Results, Interpretations 

-> Please check these sections in the full version of the [paper](https://github.com/clem2507/thesis_nomics/files/7612428/clement_detry_paper_v2.pdf)

## How to run the classification tool?

1. Clone the repository
2. Install the libraries using the requirements file by running the following line in your command line:

```
pip install -r requirements.txt
```

3. Three analyses are available to test the classification tool on by running:

```
python3 classification/classify_jawac.py --edf 'training/test_data/patient_data1.edf' --out_graph True --model 'LSTM' --num_class 3
python3 classification/classify_jawac.py --edf 'training/test_data/patient_data2.edf' --out_graph True --model 'LSTM' --num_class 3
python3 classification/classify_jawac.py --edf 'training/test_data/patient_data3.edf' --out_graph True --model 'LSTM' --num_class 3
```
- The model parameter can be changed between CNN and LSTM and the num_class parameter can be switched from 2 to 3 for binary or multinomial classification.

## Links
- [Nomics Sleep & Care](https://www.brizzy.eu)
- [Maastricht University DSAI](https://www.maastrichtuniversity.nl/education/bachelor/data-science-and-artificial-intelligence)
- [Full research paper link](https://github.com/clem2507/thesis_nomics/files/7612428/clement_detry_paper_v2.pdf)
