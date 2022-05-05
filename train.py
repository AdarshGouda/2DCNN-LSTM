import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, GlobalAveragePooling1D
from tensorflow.keras.layers import TimeDistributed

from prepare_raw_data import prepare_raw_processed

parser = argparse.ArgumentParser(description='Video_Classifier')
#parser.add_argument('--config', default='./configs/config_default.yaml')
parser.add_argument('--config', default='/content/drive/MyDrive/Colab Notebooks/Project/Final_Project_rev3/configs/config_default.yaml')


def predict_probas(model, X, n_samples=10):
    y_probs = [model.predict(X) for sample in range(n_samples)]
    return np.mean(y_probs, axis=0)


def predict_classes(model, X, n_samples=10):
    y_probs = predict_probas(model, X, n_samples)
    return np.argmax(y_probs, axis=1)


def train(from_directory):
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    model_path = args.model_path
    image_shape = args.image_shape
    grab_frame = args.grab_frame
    model_type = args.type
    rnns = args.rnn
    hidden_size = args.hidden_size
    lr = args.learning_rate
    classes_num = args.classes

    X_data, y_data = prepare_raw_processed(from_directory, grab_frames=int(grab_frame))

    print(X_data.shape)

    lb = LabelBinarizer()
    lb.fit(y_data)
    y_hot = lb.transform(y_data)

    print(y_hot.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_hot, test_size=0.30, random_state=42)

    if model_type == "VGG19":
        vg19 = tf.keras.applications.vgg19.VGG19
        base_model = vg19(include_top=False, weights='imagenet', input_shape=(int(image_shape), int(image_shape), 3))

    cnn = Sequential()
    cnn.add(base_model)
    cnn.add(Flatten())
    cnn.trainable = False

    model = Sequential()
    model.add(TimeDistributed(cnn, input_shape=(int(grab_frame), int(image_shape), int(image_shape), 3)))
    model.add(LSTM(int(rnns), return_sequences=True))
    model.add(TimeDistributed(Dense(int(hidden_size), activation='relu')))
    model.add(GlobalAveragePooling1D(name="globale"))
    model.add(Dense(int(classes_num), activation="softmax", name="last"))
    adam = tf.keras.optimizers.Adam(lr=float(lr), beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rms = tf.keras.optimizers.RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    history = model.fit(x=X_train, y=y_train, batch_size=int(args.batch_size),
                        steps_per_epoch=int(args.steps_per_epochs), epochs=int(args.epochs),
                        validation_split=float(args.val_split))

    hist = pd.DataFrame(history.history)
    plot = hist.plot()
    #plot.figure.savefig("D:/Study/OMSCS/Semester4/CS7643 Deep Learning/Project/Final_Project_rev2/report/Train_curves.jpg")
    plot.figure.savefig(
        "/content/drive/MyDrive/Colab Notebooks/Project/Final_Project_rev3/report/Train_curves.jpg")

    model.save(model_path)

    print("Learning Curves saved in Report Folder")
    print("Model saved to: ", model_path)


if __name__ == '__main__':
    gpu = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpu[0], True)

    train(from_directory="./data/split_data/train/")






