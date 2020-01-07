from argus import Model, optimizer
from argus.utils import deep_detach

from src.models.cnn_finetune import get_cnn_finetune_model
from src.losses import BengaliAiCrossEntropy
from src.optimizers import Over9000, RAdam


class BengaliAiModel(Model):
    nn_module = {
        'cnn_finetune': get_cnn_finetune_model
    }
    loss = {
        'BengaliAiCrossEntropy': BengaliAiCrossEntropy
    }
    optimizer = {
        **optimizer.get_pytorch_optimizers(),
        "RAdam": RAdam,
        "Over9000": Over9000
    }

    def __init__(self, params):
        super().__init__(params)
        self.amp = None

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        if self.amp is not None:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }
