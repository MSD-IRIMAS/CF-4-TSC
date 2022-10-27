import numpy as np
import pandas as pd

import os
import argparse

from sklearn.metrics import accuracy_score

from utils.utils import load_data, znormalisation, create_directory, encode_labels

from classifiers.CO_FCN import COFCN
from classifiers.H_FCN import HFCN
from classifiers.H_Inception import HINCEPTION

def get_args():

    parser = argparse.ArgumentParser(
    description="Choose to apply which classifier on which dataset with number of runs.")

    parser.add_argument(
        '--dataset',
        help="which dataset to run the experiment on.",
        type=str,
        default='Coffee'
    )

    parser.add_argument(
        '--classifier',
        help='which classifier to use',
        type=str,
        choices=['CO-FCN', 'H-FCN', 'H-Inception'],
        default='H-Inception'
    )

    parser.add_argument(
        '--runs',
        help="number of runs to do",
        type=int,
        default=5
    )

    parser.add_argument(
        '--output-directory',
        help="output directory parent",
        type=str,
        default='results/'
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    use_ensemble = {

        'CO-FCN' : False,
        'H-FCN' : True,
        'H-Inception' : True

    }

    args = get_args()

    output_dir_parent = args.output_directory
    create_directory(output_dir_parent)

    output_dir_clf = output_dir_parent + args.classifier + '/'
    create_directory(output_dir_clf)

    if os.path.exists(output_dir_clf + 'results_UCR_128.csv'):

        df = pd.read_csv(output_dir_clf + 'results_UCR_128.csv')

        dataset_names = list(df['dataset_name'])
        if args.dataset in dataset_names:
            
            print("Already done")
            exit()
    
    else:

        if use_ensemble[args.classifier]:
            df = pd.DataFrame(columns=['dataset_name', args.classifier+'-mean', args.classifier+'-std', args.classifier+'Time'])
        else:
            df = pd.DataFrame(columns=['dataset_name', args.classifier+'-mean', args.classifier+'-std'])
    
    xtrain, ytrain, xtest, ytest = load_data(file_name=args.dataset)

    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    xtrain = np.expand_dims(xtrain, axis=2)
    xtest = np.expand_dims(xtest, axis=2)

    ytrain = encode_labels(ytrain)
    ytest = encode_labels(ytest)

    length_TS = int(xtrain.shape[1])
    n_classes = len(np.unique(ytrain))

    scores = []

    if use_ensemble[args.classifier]:
        ypred_ensemble = np.zeros(shape=(len(ytest), len(np.unique(ytest))))
    
    for _run in range(args.runs):

        output_dir = output_dir_clf + 'run_' + str(_run) + '/'
        create_directory(output_dir)

        output_dir = output_dir + args.dataset + '/'
        create_directory(output_dir)

        if args.classifier == 'CO-FCN':
            clf = COFCN(output_directory=output_dir, length_TS=length_TS, n_classes=n_classes)

        elif args.classifier == 'H-FCN':
            clf = HFCN(output_directory=output_dir, length_TS=length_TS, n_classes=n_classes)
        
        elif args.classifier == 'H-Inception':
            clf = HINCEPTION(output_directory=output_dir, length_TS=length_TS, n_classes=n_classes)
        
        if not os.path.exists(output_dir + 'loss.pdf'):
            clf.fit(xtrain=xtrain, ytrain=ytrain, xval=xtest, yval=ytest, plot_test=True)
        
        if use_ensemble[args.classifier]:

            ypred, score = clf.predict(xtest=xtest, ytest=ytest)

            ypred_ensemble = ypred_ensemble + ypred
        
        else:
            score = clf.predict(xtest=xtest, ytest=ytest)
        
        scores.append(score)
    
    if use_ensemble[args.classifier]:

        ypred_ensemble = ypred_ensemble / (1.0 * args.runs)
        ypred_ensemble = np.argmax(ypred_ensemble, axis=1)

        score_ensemble = accuracy_score(y_true=ytest, y_pred=ypred_ensemble, normalize=True)

        df = df.append({

            'dataset_name' : args.dataset,
            args.classifier+'-mean' : np.mean(scores),
            args.classifier+'-std' : np.std(scores),
            args.classifier+'Time' : score_ensemble}, ignore_index=True)
    
    else:

        df = df.append({

            'dataset_name' : args.dataset,
            args.classifier+'-mean' : np.mean(scores),
            args.classifier+'-std' : np.std(scores)}, ignore_index=True)
        
    df.to_csv(output_dir_clf + 'results_UCR_128.csv', index=False)