import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

from data_util import get_data_iterator
from sklearn.metrics import accuracy_score


class ModelUtil:

    def __init__(self, model, batch_size, fields, device, optimizer=None,
                 criterion=None, model_path='./gru_models',
                 metrics_path='./gru_metrics.pt'):
        self.model = model
        self.batch_size = batch_size
        self.fields = fields
        self.device = device
        self.optimizer = optimizer if optimizer is not None else \
            optim.Adam(model.parameters())
        self.criterion = criterion if criterion is not None else \
            BCEWithLogitsLoss()
        self.model_path = model_path
        self.metrics_path = metrics_path

        self.threshold = torch.Tensor([0.5])
        self.threshold = self.threshold.to(self.device)

    def load_model(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def save_metrics(self, state, path=None):
        if path is not None:
            torch.save(state, path)
        else:
            torch.save(state, self.metrics_path)

    def accuracy_score(self, data, shuffle=True):
        data_loader = get_data_iterator(data[0], data[1], self.fields,
                                        self.batch_size, self.device, shuffle)
        return self._accuracy(data_loader[0])

    def _accuracy(self, data_iterator):
        preds, labels, probs = self._predict(data_iterator, predict_class=True)
        return accuracy_score(labels, preds), labels, preds, probs

    def predict_class(self, data, shuffle=True):
        data_loader = get_data_iterator(data[0], data[1], self.fields,
                                        self.batch_size, self.device, shuffle)
        return self._predict(data_loader[0], True)[0]

    def predict_prob(self, data, shuffle=True):
        data_loader = get_data_iterator(data[0], data[1], self.fields,
                                        self.batch_size, self.device, shuffle)
        return self._predict(data_loader[0], False)[0]

    def _predict(self, iterator, predict_class):
        self.model.eval()

        pred = []
        labels = []
        probs = []
        with torch.no_grad():
            for batch in iterator:
                text = batch.text.to(self.device)
                output = self.model(text)
                output = torch.sigmoid(output)

                probs += list(output.cpu().numpy())
                if predict_class:
                    output = (output >= self.threshold).float()
                labels += list(batch.label.cpu().numpy())
                pred += list(output.cpu().numpy())
        return pred, labels, probs

    def _set_up_train_vars(self, data):
        train_iterator, val_iterator = get_data_iterator(
            data[0], data[1], self.fields, self.batch_size, self.device)
        return (train_iterator, [], []), (val_iterator, [], [])

    def _evaluate_data(self, data):
        self.model.eval()

        loss = 0
        num_loop = 0
        with torch.no_grad():
            for batch in data:
                text = batch.text.to(self.device)
                labels = batch.label.to(self.device)

                output = self.model(text)
                loss += self.criterion(output, labels).item()
                num_loop += 1
        return loss / num_loop

    def fit(self, data, num_epochs, validation=False, save_final=False):

        (train_iterator, training_loss, training_accuracy), \
            (val_iterator, validation_loss, validation_accuracy) = \
            self._set_up_train_vars(data)

        max_accuracy = -float("inf")
        max_epoch = -1

        print("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()

            train_loss_epoch = 0
            num_loop = 0
            for batch in train_iterator:
                text = batch.text.to(self.device)
                label = batch.label.to(self.device)

                output = self.model(text)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_epoch += loss.item()
                num_loop += 1

            train_loss_epoch /= num_loop
            training_loss.append(train_loss_epoch)
            training_acc_epoch = self._accuracy(train_iterator)[0]
            training_accuracy.append(training_acc_epoch)

            print("Finished epoch {:d}\n"
                  "\tTraining accuracy: {:.6f}\n"
                  "\tTotal training loss: {:.6f}"
                  .format(epoch, training_acc_epoch, train_loss_epoch))

            if validation:
                val_loss_epoch = self._evaluate_data(val_iterator)
                validation_loss.append(val_loss_epoch)
                val_acc_epoch = self._accuracy(val_iterator)[0]
                validation_accuracy.append(val_acc_epoch)

                print("\tValidation accuracy: {:.6f}\n"
                      "\tTotal validation loss: {:.6f}"
                      .format(val_acc_epoch, val_loss_epoch))
                accuracy_epoch = val_acc_epoch
            else:
                accuracy_epoch = training_acc_epoch

            if accuracy_epoch > max_accuracy:
                max_accuracy = accuracy_epoch
                max_epoch = epoch

                # Save best model so far
                self.save_model(self.model_path + "_epoch_{:d}.pt".format(
                    epoch))

        state = {
            'training_accuracy': training_accuracy,
            'training_loss': training_loss,
            'validation_accuracy': validation_accuracy,
            'validation_loss': validation_loss
        }
        self.save_metrics(state, None)

        print("Finished training!")

        if validation:
            print("\tBest validation accuracy achieved after epoch: {:d}\n"
                  "\tValidation accuracy: {:.6f}\n"
                  "\tTraining accuracy: {:.6f}\n"
                  "\tValidation loss: {:.6f}\n"
                  "\tTraining loss: {:.6f}\n"
                  .format(max_epoch, max_accuracy, training_accuracy[max_epoch],
                          validation_loss[max_epoch],
                          training_loss[max_epoch]))
        else:
            print("\tBest training accuracy achieved after epoch: {:d}\n"
                  "\tTraining accuracy: {:.6f}\n"
                  "\tTraining loss: {:.6f}\n"
                  .format(max_epoch, max_accuracy,
                          training_loss[max_epoch]))

        if save_final:
            self.save_model(self.model_path + "_final_model.pt")
