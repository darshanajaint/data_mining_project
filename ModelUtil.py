import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from data_util import DataFrameDataset, get_data_iterator


class ModelUtil:

    def __init__(self, model, device, optimizer=None, criterion=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer if optimizer is not None else \
            optim.Adam(model.parameters())
        self.criterion = criterion if criterion is not None else \
            BCEWithLogitsLoss()

        self.threshold = torch.Tensor(0.5)

    def load_model(self, path):
        # Todo: should load both optimizer and model ckpts
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        # Todo: should save both optimizer and model ckpts
        torch.save(self.model.state_dict(), path)

    def predict_class(self, data):
        # Todo: create data loader function
        data_loader = data
        return self._predict(data_loader, True)

    def predict_prob(self, data):
        data_loader = data
        return self._predict(data_loader, False)

    def _predict(self, iterator, pred_class):
        self.model.eval()

        pred = []
        with torch.no_grad():
            for batch in iterator:
                text = batch.text.to(self.device)
                output = self.model(text)
                output = torch.sigmoid(output)
                if pred_class:
                    output = (output >= self.threshold).float()

                pred += list(output.cpu().numpy())
        return pred

    def fit(self, train, val=None):
        # Todo: allow training for whole data or test and train data
        pass
