"""A simple main file to showcase the template."""
'''Modulo por defecto para poner linea de comandos es argparse'''
import logging.config
import argparse
import os
import time

import tensorflow as tf

from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import  activations
from tensorflow.keras import  optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import utils
from tensorflow.keras import callbacks
from . import __version__

#constante para creo logger
LOGGER = logging.getLogger()
VERSION = __version__

#Download data function
def __download_data():
    LOGGER.info("Downloading data...")
    train, test = datasets.mnist.load_data()
    X_train, y_train = train
    X_test, y_test = test
    return X_train, y_train,X_test,y_test

#Preprocess data function
def _preprocess_data(x,y,model):
    LOGGER.info("Preprocessing data...")
    x = x / 255.0
    y = utils.to_categorical(y)
    if model == 'cnn':
        x= x.reshape(-1, 28, 28, 1)
    return x,y

#Build the model of the neuronal network
def _build_dense_model():
    m = models.Sequential()
    m.add(layers.Input((28,28), name='input_layer'))
    m.add(layers.Flatten())
    m.add(layers.Dense(128,activation=activations.relu))
    m.add(layers.Dense(64,activation=activations.relu))
    m.add(layers.Dense(32,activation=activations.relu))
    m.add(layers.Dense(10,activation=activations.softmax))

    return m

#New Models
def _build_conv_model():
    m = models.Sequential()
    m.add(layers.Input((28, 28, 1), name='my_input_layer'))
    m.add(layers.Conv2D(32, (3, 3), activation=activations.relu))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(16, (3, 3), activation=activations.relu))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Conv2D(8, (3, 3), activation=activations.relu))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Flatten())
    m.add(layers.Dense(10, activation=activations.softmax))

    return m 


'''Aquí necesitamos una función train and evaluate por convención. AI va a llamar a esta función'''
def train_and_evaluate(batch_size,epochs,job_dir,output_path,is_hypertune,model):

    #Download the data
    X_train, y_train, X_test, y_test  = __download_data()

    #Preprocess the data
    X_train, y_train = _preprocess_data(X_train,y_train,model)
    X_test, y_test  =_preprocess_data(X_test,y_test,model)

    #if model is conv
    if model == 'cnn':
        model = _build_conv_model()
    # Build the model
    if model == 'dense':
        model = _build_dense_model()
    model.compile(optimizer=optimizers.Adam(), metrics=[metrics.categorical_accuracy], loss=losses.categorical_crossentropy)
    
 

    #Train the model
    #decidimos dir donde escribiremos los logs
    logdir = os.path.join(job_dir,"logs/scalars/"+ time.strftime("%Y%m%d-%H%M%S"))
    #creamos call back para tensor board
    td_callback = callbacks.TensorBoard(log_dir=logdir)
    model.fit(X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[td_callback]    
            )

    #Evaluate the model
    loss_value, accuracy = model.evaluate(X_test,y_test)
    LOGGER.info("LOSS VALUE:    %f      ACCURACY:   %.4f" % (loss_value,accuracy))
    
    #Communicate results from model evaluation
    
    if is_hypertune:
        '''Writer es un summary de tensro flow'''
        metric_tag = "accuracy_live_class"
        eval_path = os.path.join(job_dir,metric_tag)
        writer = tf.summary.create_file_writer(eval_path)
        with writer.as_default():
            '''
            En este caso usamos accuracy porque es la metrica que hemos puesto en nuestro modelo(se podría poner otras) 
            y hay que poner un valor de steps, vamos a usar las epocas.#
            '''
            tf.summary.scalar(metric_tag, accuracy,step=epochs)
        writer.fluch()
    #hypertune option
    if not is_hypertune:
        #Save model in TF SaveModel format
        model_dir = os.path.join(output_path,VERSION)
        models.save_model(model, model_dir, save_format='tf')

def main():
    '''Argumentos que se van a introducir (recordar que en yaml los hiper parametros se llaman igual que estos param)'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int, help='Batch size for the training')
    parser.add_argument('--epochs',type=int,help='Number of epochs for the training')
    parser.add_argument('--job-dir',default=None,required=False, help='Option for AI platform')
    '''El resultado se va a escribir en un fichero save modele de tensorflow y hay que indicar elpath'''
    parser.add_argument('--model-output-path', help='Path to write the  SaveModel format')
    #Opcion para el tuneo de hiperparametros
    parser.add_argument('--hypertune',action='store_true',help='This is a hypertuning job')
    parser.add_argument('--model',help='Model type to choose')

    #Recuperamos las opciones
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path
    #hypertune variable
    is_hypertune = args.hypertune
    model = args.model

    train_and_evaluate(batch_size,epochs,job_dir,output_path,is_hypertune,model)

if __name__ == "__main__":
    main()
