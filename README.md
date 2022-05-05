# 2DCNN-LSTM

##Libraries Used:
1. TensorFlow
2. OpenCv2
3. sklearn
4. argparse
5. yaml
6. seaborn
7. matplotlib
8. tqdm

## Procedure:
1. Download the raw data into ./data folder
2. Run python script sort_data.py --> all video files will be sorted according to class in folders
3. run test_train_split.py --> New folders will be created ./test and ./val
4. Look for config_default.yaml file in ./config folder --> changes hyperparameters as needed
5. Run train.py --> model will be saved in ./saved_model folder and the loss curves will be saved in ./report folder
6. Run test.py --> accuracy will be printed to the terminal and the confusion matrix plot will be saved in ./report folder
