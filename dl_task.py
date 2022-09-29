import torch


class DLTask:
    def __init__(self, name, device, model, model_save_dir, optimizer, config, scheduler=None, class_weights=None):
        self.name = name
        self.model_save_dir = model_save_dir
        self.class_weights = class_weights
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def train(self, loader, epoch, save_model_every_epoch=5):
        # implement in child
        return None

    @torch.no_grad()
    def eval(self, loader, load_from_path=None, mode='val', epoch=0):
        # implement in child
        return None

    def get_accuracy(self, out, target):
        correct_nodes = out.argmax(dim=1).eq(target).sum().item()
        return correct_nodes / len(target)
