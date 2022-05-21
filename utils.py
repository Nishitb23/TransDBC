import pickle
import os
from sklearn.model_selection import train_test_split
import numpy as np

#read data from pkl file
def load_full_dataset(data_path='drive/MyDrive/data'):

    with open(os.path.join(data_path, 'motorway_dataset.pkl'), 'rb') as f:
        save = pickle.load(f,encoding='bytes')
        motorway_dataset = save[b'dataset']
        motorway_labels = save[b'labels']
        del save

    with open(os.path.join(data_path,'secondary_dataset.pkl'), 'rb') as f:
        save = pickle.load(f,encoding='bytes')
        secondary_dataset = save[b'dataset']
        secondary_labels = save[b'labels']
        del save

    dataset = np.concatenate((motorway_dataset,secondary_dataset), axis=0)
    labels = np.concatenate((motorway_labels,secondary_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test
