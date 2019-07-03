import sys
import os
import argparse as ap
import math
from setup.settings import hparams, preprocessing
from setup.custom_summary import custom_summary
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/nmt")
from nmt import nmt
import tensorflow as tf
import threading

# Cross-platform colored terminal text
from colorama import init, Fore
init()

'''
Start training model with custom decaying scheme and trainig duration
'''   
def train_model():

    print('\n\n{}Started training model...{}\n'.format(Fore.GREEN, Fore.RESET))

    # Custom epoch training and decaying
    if preprocessing['epochs'] is not None:

        # Load corpus size, calculate number of steps
        with open('{}/corpus_size'.format(preprocessing['train_folder']), 'r') as f:
            corpus_size = int(f.read())

        # Load current train progress
        try:
            with open('{}epochs_passed'.format(hparams['out_dir']), 'r') as f:
                initial_epoch = int(f.read())
        except:
            initial_epoch = 0

        # Iterate thru specified epochs
        for epoch, learning_rate in enumerate(preprocessing['epochs']):

            # Check if model already passed that epoch
            if epoch < initial_epoch:
                print('{}Epoch: {}, learning rate: {} - already passed{}'.format(Fore.GREEN, epoch + 1, learning_rate, Fore.RESET))
                continue

            # Calculate new number of training steps - up to the end of current epoch
            number_training_steps = math.ceil((epoch + 1) * corpus_size / (hparams['batch_size'] if 'batch_size' in hparams else 128))
            
            # Update 
            print("\n{}Epoch: {}, steps per epoch: {}, epoch ends at {} steps, learning rate: {} - training{}\n".format(
                Fore.GREEN,
                epoch + 1,
                math.ceil(corpus_size / (hparams['batch_size'] if 'batch_size' in hparams else 128)),
                number_training_steps,
                learning_rate,
                Fore.RESET
            ))

            # Override the hyperparameters
            hparams['num_train_steps'] = number_training_steps
            hparams['learning_rate'] = learning_rate
            hparams['override_loaded_hparams'] = True

            # Run TensorFlow threaded (exits on finished training, but we want to train more)
            thread = threading.Thread(target=nmt_train)
            thread.start()
            thread.join()

            # Save epoch progress so that we can handle interruption
            with open('{}epochs_passed'.format(hparams['out_dir']), 'w') as f:
                f.write(str(epoch + 1))

    #default training
    else:
        nmt_train()

    print('\n\n{}Finished training model{}\n'.format(Fore.GREEN, Fore.RESET))

'''
Start training model with default settings
'''
def nmt_train():
    nmt_parser = ap.ArgumentParser()
    nmt.add_arguments(nmt_parser)

    # Get settingds from configuration file
    nmt.FLAGS, unparsed = nmt_parser.parse_known_args(['--' + key + '=' + str(value) for key, value in hparams.items()])

    # Custom summary callback hook
    nmt.summary_callback = custom_summary

    # Run tensorflow with modified arguments
    tf.app.run(main=nmt.main, argv=[os.getcwd() + '\nmt\nmt\nmt.py'] + unparsed)

# Start training model
train_model()