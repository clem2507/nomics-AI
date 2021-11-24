[<img src="https://user-images.githubusercontent.com/43852207/140623269-642a3d0f-ee76-4194-aa56-b7305617de9d.jpeg" width="170" height="170">](https://www.brizzy.eu) [<img src="https://user-images.githubusercontent.com/43852207/140623238-20bbde3c-7dfe-4965-b773-331e6e906069.jpg" width="170" height="170">](https://www.maastrichtuniversity.nl/education/bachelor/data-science-and-artificial-intelligence)

# Multinomial Classification Ai Of Human Sleep Physiological Signal Data
### Maastricht University - BSc Thesis in Collaboration with Nomics Sleep & Care
> Clément Detry

## Introduction

### Company

Nomics Sleep & Care is a Belgian company known for developping electronical devices used for sleep apnea diagnosis. The great benefit of using their technologies is the way patients are treated. Indeed, they have the possibility to test themselves at home, in their own environment, with only 2 facial captors. A big difference considering the costs and the difficulty to obtain meaningful results in clinical tests. In brief, Nomics is more than 25 clinical papers, over 10 years specialized in sleep diagnosis, more than 15000 patients by 2018 and at least 90% of true diagnosis.

### Context

However, one issue driven by such a system is the lack of direct control over patients during the test. This could somtimes lead to invalid recordings, for example if one of the captor has moved during the night. In some of these cases, if the total duration of continuous valid analysis is not sufficient, patients must test themselves again.

### Problem Statement

Up to now, humans are responsible for checking the validity of each graph. This is not a gratifying task and it takes a little longer for doctors to get direct feedback right after uploading their patients' analysis.

## Methodology

One solution would be to use machine learning to replace this redundant human task. Recently, deep learning methods such as CNN that make use of one-dimensional convolutional neural networks or recurrent neural networks as LSTMs have been shown to provide state-of-the-art results on challenging time series classification tasks with little or no data feature engineering.

### Dataset Prepocessing

In order to have consistent input data for the machine learning training, it is advisable to find a way to get samples of the same size. In fact, time series do not have the same length as the recording time can vary. One approach of doing it is by splitting data into blocks of same size with their corresponding label (cf. image below).

<p align="center">
  <img src="https://user-images.githubusercontent.com/43852207/140624930-3faee870-7a69-4e6a-a14f-d3248fd68a15.png" width="400" height="400">
</p>

The implementation of this technique is inside the [preprocessing.py](https://github.com/clem2507/thesis_nomics/blob/main/data_loader/preprocessing.py) file. Now that the data is ready to be processed, deep learning models can get into play.

### Convolutional Neural Network (CNN)

First approach is to use the well known CNN classification technique. In the scope of time series data, the Convolutional Neural Network process can be divided in 4 main steps.

1. It first creates a kernel of time series width length k that will only move in one direction, from the start to the end of the time series.
2. Then, the convolutional layer is created after performing convolution with the kernel on the input data. Its main goal is to filter the information and to extract the most significant features.
3. Next, the dimensionality space is reduced using gloabel max pooling to fit the data in into a NN.
4. Finally, the data goes into a fully connected layer to perform classification.

<p align="center">
  <img src="https://user-images.githubusercontent.com/43852207/140625099-a3b752d7-290f-4f2c-9abb-baa92fc8834b.png" width="500" height="200">
</p>

The main pros os using CNN for such a classification is its noise resistance as well as its ability to extract deep features independent from time. The CNN model is implemented using [Tensorflow Keras API](https://www.tensorflow.org/guide/keras/sequential_model) sequential architecture inside the [cnn_model.py](https://github.com/clem2507/thesis_nomics/blob/main/deep_learning_models/cnn_model.py) file. With the objective of achieving the best results, a version of the keras tuner is included to find the best hyperparameters after performing random search.

### Long Short Term Memory (LSTM)

Second approach is to try LSTM architecture instead of CNN. Its neuron structure is a bit more complex. It is composed of 3 main gates functions.

1. Forgotten:
2. Input:
3. Output:

<p align="center">
  <img src="https://user-images.githubusercontent.com/43852207/140625736-77ef9fee-65d5-446a-ac8e-0e2fed2bcaf3.png" width="500" height="250">
</p>

The advantage of using LSTM model is its ability to learn and remember the information over long sequences of input data. It also do not require any domain expertise to manually engineer input features. In the same way as for the CNN, the architecture used for the LSTM comes from [Tensorflow Keras API](https://www.tensorflow.org/guide/keras/sequential_model) and is complemented with a optimization tuner. 

## Results

--> Output Graphs

--> Metrics

## Conclusion

...

## Links
- [Nomics Sleep & Care](https://www.brizzy.eu)
- [Maastricht University DSAI](https://www.maastrichtuniversity.nl/education/bachelor/data-science-and-artificial-intelligence)
