import torch

from dl_task import DLTask


class GroundClassificationTask(DLTask):

    # def train(self):
    #     return None

    @torch.no_grad()
    def eval(self):
        return None
