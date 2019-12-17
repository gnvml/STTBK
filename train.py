from utils.listCSV import toCsvFile
from utils.embPerson import embededFace
from utils.embAll import embededAll
from utils.annModel import  Training

if __name__ == '__main__':

    df, newNames = toCsvFile()
    embededFace(newNames)
    x_train, y_train, NUMBER_OF_CLASSES = embededAll(df, newNames)
    Training(x_train, y_train, df, NUMBER_OF_CLASSES)
