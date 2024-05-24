import torch
import torch.nn.functional as F
import wandb
import numpy as np
from torchgeometry.losses import FocalLoss

from dl_task import DLTask
from data.pcd_utils import get_accuracy, compute_metrics, draw_pc_with_labels, get_as_pc


class ObjWSegmentationTask(DLTask):
    def train(self, loader, epoch, loss_fn=None, save_model_every_epoch=5, ignored_labels=None):
        self.model.train()
        losses = []
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            target = torch.squeeze(data.y).type(torch.LongTensor).to(self.device)
            # draw_pc_with_labels(np.asarray(data.pos.cpu()), np.asarray(target.cpu()), num_clusters=2, title="GT")
            #draw_pc_with_labels(np.asarray(data.pos.cpu()), np.asarray(np.argmax(out.cpu().detach().numpy(), axis=-1)), num_clusters=2, title="Prediction")

            if ignored_labels is None:
                loss = F.nll_loss(out, target) if not loss_fn else loss_fn(out.t().unsqueeze(0).unsqueeze(2),
                                                                           target.unsqueeze(0).unsqueeze(1))
            else:
                out_t = out.t().unsqueeze(0).unsqueeze(2)
                target_t = target.unsqueeze(0).unsqueeze(1)
                losses_no_reduct = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')(out_t, target_t)
                weights = torch.logical_not(sum(target_t.squeeze() == i for i in ignored_labels).bool()).int()
                losses_w = losses_no_reduct * weights
                loss = torch.mean(losses_w)

            loss.backward()
            self.optimizer.step()
            accuracy = get_accuracy(out, target)
            wandb.log({"train_loss": loss.item(),
                       "train_acc": accuracy,
                       "train_iteration": step
                       })
            if (i + 1) % 10 == 0:
                print(f'[{i + 1}/{len(loader)}] Loss: {loss.item():.4f} '
                      f'Train Acc: {accuracy:.4f}')
            losses.append(loss.item())
        if (epoch + 1) % save_model_every_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
                f'{self.model_save_dir}/epoch_{epoch}.pth')
        if self.scheduler:
            self.scheduler.step()
        return np.mean(losses)

    @torch.no_grad()
    def eval(self, loader, loss_fn=None, load_from_path=None, mode='val', epoch=0, ignored_labels=None):
        if load_from_path:
            self.model.load_state_dict(torch.load(load_from_path)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        metrics_dict_all, metrics_dict_all_post, iou_classwise_all, iou_classwise_all_post = {}, {}, {}, {}
        for i, data in enumerate(loader):
            data = data.to(self.device)
            step = i + len(loader) * epoch
            out = self.model(data)
            out = out.cpu()
            target = torch.squeeze(data.y).type(torch.LongTensor).cpu()
            pred = np.argmax(out, axis=-1)


            metrics_dict, iou_classwise = compute_metrics(target=target, out=out, pred=pred,
                                                          loss_fn=loss_fn, mode=mode)

            metrics_dict_all = {key: metrics_dict.get(key, []) + metrics_dict_all.get(key, [])
                                for key in set(list(metrics_dict.keys()) + list(metrics_dict_all.keys()))}
            iou_classwise_all = {key: iou_classwise.get(key, []) + iou_classwise_all.get(key, [])
                                 for key in set(list(iou_classwise.keys()) + list(iou_classwise_all.keys()))}

            accuracy = metrics_dict["accuracy"][0]
            loss = metrics_dict['loss'][0]
            print(f'[{i + 1}/{len(loader)}]'
                  f'Eval Acc: {accuracy:.4f}')
            if accuracy > 0.90:
                draw_pc_with_labels(np.asarray(data.pos.cpu()), np.asarray(pred), num_clusters=2, title="Prediction")
                get_as_pc(np.asarray(data.pos.cpu()), colors=np.asarray(data.x[:, :3].cpu()), visualise=True)
                draw_pc_with_labels(np.asarray(data.pos.cpu()), np.asarray(target.cpu()), num_clusters=2, title="GT")

            wandb.log(
                {
                    "val_loss": loss,
                    "val_acc": accuracy,
                    "val_iteration": step
                })

        if mode != 'val':
            self.print_res(metrics_dict_all, title='ALL METRICS NN', print_overall_mean=False,
                           mean_over_nonzero=False, classwise=False)
            self.print_res(iou_classwise_all, title='Classwise NN results:', classwise=False, mean_over_nonzero=False)
        return metrics_dict_all, metrics_dict_all_post
