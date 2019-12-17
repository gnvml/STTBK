'''
Architecture Ann model for face recognition

'''
import os
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json

BATCH_SIZE = 32
EPOCHS = 100
INPUT_DIM = 128


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def Training(x_train, y_train, df, NUMBER_OF_CLASSES):
    '''
    Training ANN model and automatically save model in cache foler
    '''

    
    model = Sequential()
    model.add(Dense(2048, input_dim=INPUT_DIM, init='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=NUMBER_OF_CLASSES, init='uniform', activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
    
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=BATCH_SIZE, epoch=EPOCHS, suffix='none'))


class Model():
    '''
    Usage: 
    
    from utils.annModel import Model
    
    Model.predict(list_endcodings)
    
    Return list names prediction

    '''


    def __init__(self):
        
        self.model = model_from_json(open("cache/architecture_32_100_none.json").read())
        self.model.load_weights("cache/model_weights_32_100_none.h5")
        self.df = pd.read_csv('CSV_LIST/CSV_List.csv')
        NUMBER_OF_CLASSES = self.df['ID'].values.max() + 1
        self.data = np.load("train_data.npy")
    
        self.classes = []
        
        for i in range(NUMBER_OF_CLASSES):
            self.classes.append(self.df[self.df['ID'] == i]['Name'].iloc[0])

    def predict(self, face_endcodings):
        face_endcodings = np.asarray(face_endcodings)
        vector = self.model.predict(face_endcodings)
        
        index = np.argmax(vector, axis = 1)
        name = []
        
        for k,i in zip(range(len(vector)), index):
            d = np.linalg.norm(self.data - face_endcodings[k], axis=1)
            print('Max: {} -- Min distance: {}'.format(vector[k][i], d.min()))
            if vector[k][i] > 0.6 and d.min() < 0.5:
                name.append(self.classes[i])
            else:
                name.append('')
      
        
        # name = [self.classes[i] for i in index]
        return name

