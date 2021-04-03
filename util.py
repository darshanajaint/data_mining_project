import torch


def save_model(path, model, optimizer):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(state, path)


def load_model(path, model, optimizer):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])


def save_metrics(path, epoch, train_acc, val_acc, train_loss, val_loss):
    state = {
        'epoch': epoch,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(state, path)


def load_metrics(path):
    return torch.load(path)
