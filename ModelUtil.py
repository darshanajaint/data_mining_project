import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from data_util import get_data_iterator
from util import get_labels
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

        self.threshold = torch.Tensor(0.5)

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

    def save_metrics(self, training_accuracy, training_loss, val=None,
                     validation_accuracy=None, validation_loss=None):
        state = {
            'training_accuracy': training_accuracy,
            'training_loss': training_loss
        }

        if val:
            state['validation_accuracy'] = validation_accuracy
            state['validation_loss'] = validation_loss

        torch.save(state, self.metrics_path)

    def accuracy_score(self, data):
        data_loader, _ = get_data_iterator(data, self.batch_size,
                                           self.fields, self.device)
        labels = get_labels(data_loader, self.device)
        predictions = self._predict(data_loader, predict_class=True)
        return accuracy_score(labels, predictions)

    def predict_class(self, data):
        data_loader, _ = get_data_iterator(data, self.batch_size,
                                           self.fields, self.device)
        return self._predict(data_loader, True)

    def predict_prob(self, data):
        data_loader, _ = get_data_iterator(data, self.batch_size,
                                           self.fields, self.device)
        return self._predict(data_loader, False)

    def _predict(self, iterator, predict_class):
        self.model.eval()

        pred = []
        with torch.no_grad():
            for batch in iterator:
                text = batch.text.to(self.device)
                output = self.model(text)
                output = torch.sigmoid(output)
                if predict_class:
                    output = (output >= self.threshold).float()

                pred += list(output.cpu().numpy())
        return pred

    def _set_up_train_vars(self, data):
        if data:
            iterator, _ = get_data_iterator(data, self.batch_size, self.fields,
                                            self.device)
        else:
            iterator = None

        loss = []
        accuracy = []
        return iterator, loss, accuracy

    def _evaluate_data(self, data):
        self.model.eval()

        loss = 0
        with torch.no_grad():
            for batch in data:
                text = batch.text.to(self.device)
                labels = batch.label.to(self.device)

                output = self.model(text)
                loss += self.criterion(output, labels).item()
        return loss

    def fit(self, train, num_epochs, val=None):
        train_iterator, training_loss, training_accuracy = \
            self._set_up_train_vars(train)

        val_iterator, validation_loss, validation_accuracy = \
            self._set_up_train_vars(val)

        min_loss = float("inf")
        min_epoch = -1

        print("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()
            train_loss_epoch = 0
            for batch in train_iterator:
                text = batch.text.to(self.device)
                label = batch.label.to(self.device)

                output = self.model(text)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_epoch += loss.item()

            training_loss.append(train_loss_epoch)
            training_acc_epoch = self.accuracy_score(train_iterator)
            training_accuracy.append(training_acc_epoch)

            print("Finished epoch {:d}\n"
                  "\tTraining accuracy: {:.6f}\n"
                  "\tTotal training loss: {:.6f}"
                  .format(epoch, training_acc_epoch, train_loss_epoch))

            if val:
                val_loss_epoch = self._evaluate_data(val_iterator)
                validation_loss.append(val_loss_epoch)
                val_acc_epoch = self.accuracy_score(val_iterator)
                validation_accuracy.append(val_acc_epoch)

                print("\tValidation accuracy: {:.6f}\n"
                      "\tTotal validation loss: {:.6f}"
                      .format(val_acc_epoch, val_loss_epoch))
                loss_epoch = val_loss_epoch
            else:
                loss_epoch = train_loss_epoch

            if loss_epoch < min_loss:
                min_loss = loss_epoch
                min_epoch = epoch

                # Save best model so far
                self.save_model(self.model_path + "_epoch_{:d}.pt".format(
                    epoch))

        self.save_metrics(training_accuracy, training_loss, val,
                          validation_accuracy, validation_loss)

        print("Finished training!")

        if val:
            print("\tBest validation loss achieved after epoch: {:d}\n"
                  "\tValidation loss: {:.6f}"
                  "\tTraining loss: {:.6f}"
                  "\tValidation accuracy: {:.6f}"
                  "\tTraining accuracy: {:.6f}"
                  .format(min_epoch, min_loss, training_loss[min_epoch],
                          validation_accuracy[min_epoch],
                          training_accuracy[min_epoch]))
        else:
            print("\tBest training loss achieved after epoch: {:d}\n"
                  "\tTraining loss: {:.6f}"
                  "\tTraining accuracy: {:.6f}"
                  .format(min_epoch, min_loss, training_accuracy[min_epoch]))
