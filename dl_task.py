import torch
import numpy as np
import json
import os


class DLTask:
    def __init__(self, name, device, model, model_save_dir, optimizer, config, mode, num_classes, scheduler=None,
                 class_weights=None):
        self.name = name
        self.model_save_dir = model_save_dir
        self.class_weights = class_weights
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.mode = mode
        self.num_classes = num_classes

    def train(self, loader, epoch, loss_fn=None, save_model_every_epoch=5, ignored_labels=None):
        # implement in child
        return None

    @torch.no_grad()
    def eval(self, loader, loss_fn=None, load_from_path=None, mode='val', epoch=0, ignored_labels=None):
        # implement in child
        return None

    def get_accuracy(self, out, target):
        correct_nodes = out.argmax(dim=1).eq(target).sum().item()
        return correct_nodes / len(target)

    def print_res(self, res_dict, title, classwise=False, print_overall_mean=True, mean_over_nonzero=True):
        print(title)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        id_name_dict_path = f'{current_dir}/mode{self.mode}_num_classes{self.num_classes}_res_label_map.json'

        with open(id_name_dict_path) as f:
            id_name_dict = json.load(f)
        id_name_dict = {int(k): v for k, v in id_name_dict.items()}

        avgs = []
        for key, value in res_dict.items():
            avg = 0
            if len(value) > 0:
                if mean_over_nonzero:
                    non_zeros = np.nonzero(value)
                    if len(non_zeros[0]) > 0:
                        avg = np.mean(np.array(value)[non_zeros])
                        avgs.append(avg)
                else:
                    avg = np.mean(value)
                    avgs.append(avg)
                if avg > 0:
                    if classwise:
                        print(f'{id_name_dict[key]}: {avg}')
                    else:
                        print(f'{key}: {avg}')
        if print_overall_mean: print(f'mIoU over all: {np.mean(avgs)}')
