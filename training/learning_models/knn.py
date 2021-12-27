import os
import sys
import pickle
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from preprocessing import Preprocessing
from util import plot_confusion_matrix


def evaluate_model(analysis_directory, segmentation_value, downsampling_value, neighbors, data_balancing, log_time, standard_scale):
    """
    Method used to create and evaluate a KNN on data

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -neighbors: number of k-nearest neighbors to train the model
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -standard_scale: true to perform a standardizatin by centering and scaling the data

    Returns:

    -accuracy: model accuracy on the testing set
    """

    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/knn/{log_time}'
    os.mkdir(save_dir)

    X_train, y_train, X_test, y_test = Preprocessing(analysis_directory=analysis_directory, segmentation_value=segmentation_value, downsampling_value=downsampling_value, data_balancing=data_balancing, log_time=log_time, standard_scale=standard_scale).create_dataset()
    X_train, y_train = shuffle(X_train, y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    model = KNeighborsRegressor(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    classes = np.argmax(predictions, axis=1)

    int_y_test = []    # predicted label list
    # loop that runs through the list of model predictions to keep the highest predicted probability values
    for item in y_test:
        int_y_test.append(np.argmax(item))

    cm = confusion_matrix(y_true=int_y_test, y_pred=classes)
    accuracy = accuracy_score(int_y_test, classes)

    # confusion matrix plot
    labels = ['Invalid', 'Valid']
    cm_plt = plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix')

    cm_plt.savefig(f'{save_dir}/cm_plt.png', bbox_inches='tight')
    cm_plt.close()
    model_info_file = open(f'{save_dir}/info.txt', 'w')
    model_info_file.write(f'This file contains information about the knn model \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Segmentation value = {segmentation_value} \n')
    model_info_file.write(f'Downsampling value = {downsampling_value} \n')
    model_info_file.write(f'Signal resolution = {1/downsampling_value} Hz \n')
    model_info_file.write(f'Standard scale = {int(standard_scale)} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Log time = {log_time} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Data balancing = {data_balancing} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Num of neighbors = {neighbors} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Confusion matrix (invalid | valid) = \n {cm} \n')
    model_info_file.write('--- \n')
    model_info_file.close()

    pickle.dump(model, open(f'{save_dir}/saved_model', 'wb'))

    return accuracy


def train_model(analysis_directory, segmentation_value, downsampling_value, neighbors, data_balancing, log_time, standard_scale):
    """
    Callable method to start the training of the model

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -neighbors: number of k-nearest neighbors to train the model
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -standard_scale: true to perform a standardizatin by centering and scaling the data
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)
    score = evaluate_model(analysis_directory, segmentation_value, downsampling_value, neighbors, data_balancing, log_time, standard_scale)
    score = score * 100.0
    print('score:', score, '%')
    print('-----')
