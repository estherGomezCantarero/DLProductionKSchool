"""A simple main file to showcase the template."""
'''Modulo por defecto para poner linea de comandos es argparse'''
import logging.config
import argparse
'''Aquí necesitamos una función train and evaluate por convención. AI va a llamar a esta función'''

def train_and_evaluate(batch_size,epochs,job_dir,output_path):
    pass

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
