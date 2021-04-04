import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from util import str2bool


def read_file(file):
    return torch.load(file)


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

    class_pred = state['predictions']
    class_prob = state['probabilities']

    df = pd.DataFrame(zip(class_pred, class_prob), columns=['Class',
                                                            'Probability'])
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

    args = parser.parse_args()

    metrics = read_file(args.metrics)
    print(metrics)

    if args.training:
        plot_training_data(metrics, args.validation)
    elif args.test:
        display_test_results(metrics, args.num_res)
