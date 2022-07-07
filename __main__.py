from dataset import data_preparation
from model_init import autoencoder, train_model
#import matplotlib.pyplot as plt

def start():

    x_train = data_preparation()
    model = autoencoder()
    pred = train_model(x_train, model)
    
    #print(pred)


if __name__ == '__main__':
    start()