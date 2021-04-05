import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import isfile, join

from util import str2bool


def read_file(file):
    return torch.load(file)


def determine_best_model(path):
    files = [f for f in os.listdir(path) if isfile(join(path, f))]

    max_acc = -1
    best_model = None
    for file in files:
        state = read_file(file)
        try:
            val_acc = min(state['validation_accuracy'])

            if val_acc > max_acc:
                max_acc = val_acc
                best_model = file
        except KeyError:
            print("Cannot find validation accuracy in file {:s}. "
                  "Exiting program...".format(file))
            exit(1)

    print(best_model)


def plot_data(train, val, data_type):
    x = list(range(len(train)))

    plt.figure(figsize=(8, 6))
    plt.plot(x, train, c='b', label='Training {:s}'.format(data_type))

    if val is not None:
        plt.plot(x, val, c='r', label='Validation {:s}'.format(data_type))
        plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("{:s} per batch".format(data_type))
    plt.title("Plot of training {:s}".format(data_type))
    plt.savefig("training_{:s}.png".format(data_type))


def plot_training_data(state, val):
    if val:
        plot_data(state['training_accuracy'], state['validation_accuracy'],
                  "accuracy")
        plot_data(state['training_loss'], state['validation_loss'], "loss")
    else:
        plot_data(state['training_accuracy'], None, "accuracy")
        plot_data(state['training_loss'], None, "loss")


def display_test_results(state, num_res):
    print(f"Test accuracy: {state['accuracy']:.6f}")

    class_labels = state['labels']
    class_pred = state['predictions']
    class_prob = state['probabilities']

    df = pd.DataFrame(zip(class_labels, class_pred, class_prob),
                      columns=['True Labels', 'Predicted Class', 'Probability'])
    print(df.head(num_res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load metrics.')
    parser.add_argument('--metrics', type=str,
                        help='Path to metrics file you want to load.')
    parser.add_argument('--training', type=str2bool,
                        help='Whether the metrics includes training data.',
                        default=True)
    parser.add_argument('--validation', type=str2bool,
                        help='If metrics is training data, whether that '
                             'includes validation.', default=True)
    parser.add_argument('--test', type=str2bool,
                        help='Whether to display first num_res test results.',
                        default=False)
    parser.add_argument('--num_res', type=int,
                        help='How many test results to display - should be '
                             'less than the number of test instances.',
                        default=50)
    parser.add_argument('--best_model', type=str2bool,
                        help='Whether to determine the best model from among '
                             'a directory of models. Pass the directory to '
                             'metrics. Each model must have validation data.',
                        default=False)

    args = parser.parse_args()

    if args.training and args.test and args.best_model or \
            ((args.training or args.test) and args.best_model):
        raise ValueError("You must choose one of training, test, or best "
                         "model.")

    if args.best_model:
        determine_best_model(args.metrics)
    else:
        metrics = read_file(args.metrics)
        if args.training:
            plot_training_data(metrics, args.validation)
        elif args.test:
            display_test_results(metrics, args.num_res)
