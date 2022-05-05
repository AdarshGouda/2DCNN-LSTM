import argparse
import yaml
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from prepare_raw_data import prepare_raw_processed

parser = argparse.ArgumentParser(description='Video_Classifier')
#parser.add_argument('--config', default='./configs/config_default.yaml')
parser.add_argument('--config', default='/content/drive/MyDrive/Colab Notebooks/Project/Final_Project_rev3/configs/config_default.yaml')


def predict_probas(model, X):
    y_probs = model.predict_classes(X, batch_size=int(args.batch_size))
    return y_probs


def predict_classes(model, X):
    y_probs = predict_probas(model, X)
    return np.argmax(y_probs, axis=1)


def test(from_directory):
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    classes_num = args.classes
    model_path = args.model_path
    grab_frame = args.grab_frame

    X_test, y_test = prepare_raw_processed(from_directory, grab_frames=int(grab_frame))

    print(X_test.shape)
    print(y_test.shape)

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_hot = lb.transform(y_test)
    print(y_hot.shape)
    
    model = tf.keras.models.load_model(model_path)

    score = model.evaluate(X_test, y_hot, batch_size=int(args.batch_size))

    y_probs = model.predict(X_test, batch_size=int(args.batch_size))
    y_pred = np.argmax(y_probs, axis=1)+1

    #print(y_pred[1])
    #print(y_test[1])

    accuracy = np.mean(y_pred == y_test)

    print("Score: ", score)
    print("Test Accuracy: ", accuracy)

    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".2f", xticklabels=range(1,int(args.classes)+1), yticklabels=range(1,int(args.classes)+1),
                square=True)
    plt.ylabel("Observed")
    plt.xlabel("Predict")
    plt.savefig("/content/drive/MyDrive/Colab Notebooks/Project/Final_Project_rev3/report/test_heatmap.jpg")
    print("Image Saved")



