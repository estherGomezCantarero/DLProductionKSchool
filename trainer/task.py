"""A simple main file to showcase the template."""
'''Modulo por defecto para poner linea de comandos es argparse'''
import logging.config
import argparse

from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import  activations
from tensorflow.keras import  optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import utils

#constante para creo logger
LOGGER = logging.getLogger()

#Download data function
def __download_data():
    LOGGER.info("Downloading data...")
    train, test = datasets.mnist.load_data()
    X_train, y_train = train
    X_test, y_test = test
    return X_train, y_train,X_test,y_test

#Preprocess data function
def _preprocess_data(x,y):
    LOGGER.info("Preprocessing data...")
    x = x / 255.0
    y = utils.to_categorical(y)
    return x,y

#Build the model of the neuronal network
def _build_model():
    m = models.Sequential()
    m.add(layers.Input((28,28), name='input_layer'))
    m.add(layers.Flatten())
    m.add(layers.Dense(128,activation=activations.relu))
    m.add(layers.Dense(64,activation=activations.relu))
    m.add(layers.Dense(32,activation=activations.relu))
    m.add(layers.Dense(10,activation=activations.softmax))

    return m

'''Aquí necesitamos una función train and evaluate por convención. AI va a llamar a esta función'''
def train_and_evaluate(batch_size,epochs,job_dir,output_path):

    #Download the data
    X_train, y_train, X_test, y_test  = __download_data()

    #Preprocess the data
    X_train, y_train = _preprocess_data(X_train,y_train)
    X_test, y_test  =_preprocess_data(X_test,y_test)

    # Build the model
    model = _build_model()
    model.compile(optimizer=optimizers.Adam(), metrics=[metrics.categorical_accuracy], loss=losses.categorical_crossentropy)

    #Train the model
    model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs)

    #Evaluate the model
    loss_value, accuracy = model.evaluate(X_test,y_test)
    LOGGER.info("LOSS VALUE:    %f      ACCURACY:   %.4f" % (loss_value,accuracy))
    

def main():
    '''Argumentos que se van a introducir'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int, help='Batch size for the training')
    parser.add_argument('--epochs',type=int,help='Number of epochs for the training')
    parser.add_argument('--job-dir',default=None,required=False, help='Option for AI platform')
    '''El resultado se va  aescribir en un fichero save modele de tensorflow y hay que indicar elpath'''
    parser.add_argument('--model-output-path', help='Path to write the  SaveModel format')

    #Recuperamos las opciones
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size,epochs,job_dir,output_path)

if __name__ == "__main__":
    main()
