import torch
import torch.nn.functional as F
import wandb
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

from dl_task import DLTask


class GroundClassificationTask(DLTask):

    def train(self, loader, epoch, save_model_every_epoch=5):
        self.model.train()
        losses = []
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            target = torch.squeeze(data.y).type(torch.LongTensor).to(self.device)
            loss = F.nll_loss(out, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            # target_remapped = remap_label_for_drawing(data[0].y.cpu())
            train_acc = self.get_accuracy(out, target)
            wandb.log({"train_loss": loss.item(),
                       "train_acc": train_acc,
                       "train_iteration": step})
            print(f'[{i + 1}/{len(loader)}] Loss: {loss.item():.4f} '
                  f'Train Acc: {train_acc:.4f}')

        if (epoch + 1) % save_model_every_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
                f'{self.model_save_dir}/epoch_{epoch + 1}_model.pth')

        self.scheduler.step()
        return np.mean(losses)

    def compute_metrics(self, target, out):
        metrics_dict = defaultdict(list)

        val_loss = F.nll_loss(out, target.type(torch.LongTensor))
        metrics_dict['loss'].append(val_loss)
        acc = self.get_accuracy(out, target)
        metrics_dict['accuracy'].append(acc)
        precision, recall, _, _ = precision_recall_fscore_support(y_true=target, y_pred=out.argmax(dim=1))
        metrics_dict['precision0'].append(precision[0])
        metrics_dict['precision1'].append(precision[1])
        metrics_dict['recall0'].append(precision[0])
        metrics_dict['recall1'].append(precision[1])

        return metrics_dict

    @torch.no_grad()
    def eval(self, loader, load_from_path=None, mode='val', epoch=0):
        if load_from_path:
            self.model.load_state_dict(torch.load(load_from_path)['model_state_dict'])
        self.model.eval()
        metrics_dict_all = {}
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.to(self.device)
            out = self.model(data)
            out = out.cpu()
            target = torch.squeeze(data.y).cpu()

            metrics_dict = self.compute_metrics(target, out)
            metrics_dict_all = {key: metrics_dict.get(key, []) + metrics_dict_all.get(key, [])
                                for key in set(list(metrics_dict.keys()) + list(metrics_dict_all.keys()))}

            wandb.log({"val_loss": metrics_dict['loss'][0],
                       "val_acc": metrics_dict['accuracy'][0],
                       "val_iteration": step})
            print(f'[{i + 1}/{len(loader)}]'
                  f'Eval Acc: {metrics_dict["accuracy"][0]:.4f}')

        return metrics_dict_all
