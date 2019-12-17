'''
This function is to add new person to exis csv file (if not then create new)

Structure: 
    Id, Image, Label

'''

import pandas as pd
import os

DATA_PATH = 'DataFace'
CSV_PATH = 'CSV_LIST'
PATH = 'CSV_LIST/CSV_List.csv'

def toCsvFile():
    '''
    Export new .csv and return tuple (dataFrame, newNames)

    '''

    if not os.path.isdir(CSV_PATH):
        os.makedirs(CSV_PATH)
        df = pd.DataFrame({'Name':[],'Image':[], 'ID':[]})
        count = 0
    else:
        df = pd.read_csv(PATH)
        count = df['ID'].values.max() + 1

    Names = (list)(df['Name'])
    ID = (list)(df['ID'])
    Images = (list)(df['Image'])
    newNames = []

    for name in os.listdir(DATA_PATH):
        if name not in Names:
            for image in os.listdir(os.path.join(DATA_PATH, name)):
                Names.append(name)
                Images.append(image)
                ID.append(count)
            count += 1
            newNames.append(name)

    df = pd.DataFrame({'Name':Names,'Image':Images, 'ID':ID})
    df.to_csv(PATH, index=False)

    return df, newNames


