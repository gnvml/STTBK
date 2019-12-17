'''
This function is to merge all embeded faces in 1 file for training data

'''

import pandas as pd
import numpy as np
import os
from keras.utils.np_utils import to_categorical

CSV_PATH = 'CSV_LIST/CSV_List.csv'
EMB = 'Embeded_Face'
FILE_NAME = 'train_data.npy'


def get_y_true(df, NUMBER_OF_CLASSES):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['ID'], num_classes=NUMBER_OF_CLASSES))
    
    return np.array(y_true)

def embededAll(df, newNames):

    '''
    Check if exit file train_data.npy or not and 
    Return (x_train, y_train, NUMBER_OF_CLASSES) 

    '''

    train_new = []
    NUMBER_OF_CLASSES = df['ID'].values.max() + 1
    
    for name in newNames:    
        for embFile in os.listdir(os.path.join(EMB, name)):
            full_file_path = os.path.join(EMB, name, embFile)
            emb = np.load(full_file_path)
            train_new.append(emb)

    train_new = np.array(train_new)

    if not os.path.isfile(FILE_NAME):
        np.save(FILE_NAME, train_new)
        y_train = get_y_true(df, NUMBER_OF_CLASSES)
        
        return train_new, y_train, NUMBER_OF_CLASSES

    else:
        train = np.load(FILE_NAME)
        train = np.concatenate((train,train_new), axis = 0)
        print(train.shape)
        np.save('train_data.npy', train)
        y_train = get_y_true(df, NUMBER_OF_CLASSES)

        return train, y_train, NUMBER_OF_CLASSES