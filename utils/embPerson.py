'''
This function is to extract Embeded vector 128 dimensions from face_recognition
and save each file to .npy format

Structure:

    name 1  |
            | IMAGE.npy
    name 2  |
            | IMAGE.npy


'''

import numpy as np
import os
import face_recognition
import pandas as pd


EMB = 'Embeded_Face'
DATA_PATH = 'DataFace'
CSV_PATH = 'CSV_LIST'

def embededFace(newNames):
    '''
    Save 128 dimensions embeded of each face and ErrorFace.csv in  below CSV_LIST folder

    ''' 

    if not os.path.isdir(EMB):
        os.makedirs(EMB)

    noneFace = []
    for name in newNames:
        path = EMB + '/' + name
        if not os.path.exists(path):
            os.makedirs(path)

        for image in os.listdir(os.path.join(DATA_PATH, name)):
            try:
                full_file_path = os.path.join(DATA_PATH, name, image)
                unknown_image = face_recognition.load_image_file(full_file_path)

                face_locations = face_recognition.face_locations(unknown_image, model="cnn")
                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                np.save(path + '/' + image[:-4], face_encodings[0])
        
            except Exception:
                noneFace.append(full_file_path)

    df = pd.DataFrame({'Error Face':noneFace})
    df.to_csv(CSV_PATH + '/Error Face.csv', index=False)