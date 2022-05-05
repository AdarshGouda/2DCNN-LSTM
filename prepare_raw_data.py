# https://gist.github.com/metal3d/f671c921440deb93e6e040d30dd8719b
# https://github.com/hthuwal/sign-language-gesture-recognition/blob/master/video-to-frame.py

import argparse
import yaml
import numpy
import numpy as np
import pandas as pd
import os
import os.path
from tqdm import tqdm
import cv2
from mask import masking_operation
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser(description='Video_Classifier')
#parser.add_argument('--config', default='./configs/config_default.yaml')
parser.add_argument('--config', default='/content/drive/MyDrive/Colab Notebooks/Project/Final_Project_rev3/configs/config_default.yaml')

def prepare_raw_processed(source_directory, grab_frames=2):

    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    X_data = []
    y_temp = []

    root_ = os.getcwd()

    source_directory = os.path.abspath(source_directory)

    os.chdir(source_directory)
    sub_folders = os.listdir(os.getcwd())

    for folder in tqdm(sub_folders, unit="actions", ascii=True):
        folder_path = os.path.join(source_directory, folder)
        os.chdir(folder_path)

        files = os.listdir(os.getcwd())

        samples = 0

        for file in tqdm(files, unit="actions", ascii=False):

            class_ = int(file.split('_')[0])

            frames = []
            count = 0

            video = cv2.VideoCapture(os.path.abspath(file))
            #print(video.get(cv2.CAP_PROP_FRAME_COUNT))

            alternate = True

            while count<(grab_frames):

                ret, frame = video.read()
                if not ret:
                    break
                frame = masking_operation(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (args.image_shape,args.image_shape))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                if alternate:
                    frames.append(frame)
                    lastFrame = frame
                    count += 1
                    alternate = not alternate


            while count<(grab_frames):
                frames.append(lastFrame)
                count += 1

            video.release()
            cv2.destroyAllWindows()

            X_data.append(frames)
            y_temp.append(class_)

            #df = pd.DataFrame({"Videos":X_data, "Labels":y_temp})
            #df.to_pickle(os.path.join(source_directory, "raw_data.pkl"))

    return np.asarray(X_data), np.asarray(y_temp)

    '''    
    nTrain = int((1-test_split)*len(X_data))

    shuffle_index = np.random.shuffle(np.arange(len(X_data)))

    train_index = shuffle_index[:nTrain]
    test_index = shuffle_index[nTrain:]

    X_train = X_data[train_index]
    y_train = y_data[train_index]

    X_test = X_data[train_index]
    y_test = y_data[train_index]
    
    '''


if __name__ == '__main__':

    prepare_raw_processed(source_directory="./data/input_folder/", target_shape = 224, grab_frames=2)





