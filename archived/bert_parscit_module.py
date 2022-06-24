from typing import Any, List

import torch
import wandb
import seaborn as sn
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.bert_tokenclassification_model import BertTokenClassifier

from src.datamodules.components.synthetic_label import num_labels
from src.datamodules.components.process import postprocess


class BertParsCitLitModule(LightningModule):

    def __init__(
        self,
        model: BertTokenClassifier,
        lr: float = 2e-5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model

        self.val_acc = Accuracy(num_classes=num_labels, ignore_index=num_labels-1)
        self.test_acc = Accuracy(num_classes=num_labels, ignore_index=num_labels - 1)
        self.val_micro_f1 = F1Score(num_classes=num_labels, ignore_index=num_labels-1, average="micro")
        self.test_micro_f1 = F1Score(num_classes=num_labels, ignore_index=num_labels-1, average="micro")
        self.val_macro_f1 = F1Score(num_classes=num_labels, ignore_index=num_labels-1, average="macro")
        self.test_macro_f1 = F1Score(num_classes=num_labels, ignore_index=num_labels-1, average="macro")

        self.conf_matrix = ConfusionMatrix(num_classes=num_labels)

        self.val_acc_best = MaxMetric()
        self.val_micro_f1_best = MaxMetric()
        self.val_macro_f1_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.val_acc_best.reset()
        self.val_macro_f1_best.reset()
        self.val_micro_f1_best.reset()

    def step(self, batch: Any):
        inputs, labels = batch, batch["labels"]
        outputs = self.forward(inputs)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        input_ids = batch["input_ids"]
        loss, preds, labels = self.step(batch)

        true_preds, true_labels = postprocess(
            input_ids=input_ids,
            predictions=preds,
            labels=labels,
            label_names=LABEL_NAMES
        )

        y_true = torch.flatten(true_labels)
        preds = torch.flatten(true_preds)

        # log val metrics
        acc = self.val_acc(preds, y_true)
        micro_f1 = self.val_micro_f1(preds, y_true)
        macro_f1 = self.val_macro_f1(preds, y_true)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/micro_f1", micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/macro_f1", macro_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "labels": y_true}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        micro_f1 = self.val_micro_f1.compute()
        self.val_micro_f1_best.update(micro_f1)
        self.log("val/micro_f1_best", self.val_micro_f1_best.compute(), on_epoch=True, prog_bar=True)

        macro_f1 = self.val_macro_f1.compute()
        self.val_macro_f1_best.update(macro_f1)
        self.log("val/macro_f1_best", self.val_micro_f1_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        input_ids = batch["input_ids"]
        loss, preds, labels = self.step(batch)

        true_preds, true_labels = postprocess(
            input_ids=input_ids,
            predictions=preds,
            labels=labels,
            label_names=LABEL_NAMES
        )

        y_true = torch.flatten(true_labels)
        preds = torch.flatten(true_preds)

        # log test metrics
        acc = self.test_acc(preds, y_true)
        micro_f1 = self.test_micro_f1(preds, y_true)
        macro_f1 = self.test_macro_f1(preds, y_true)
        confmat = self.conf_matrix(preds, y_true).tolist()

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/micro_f1", micro_f1, on_step=False, on_epoch=True)
        self.log("test/macro_f1", macro_f1, on_step=False, on_epoch=True)
        
        plt.figure(figsize=(24, 24))
        sn.heatmap(confmat, annot=True, xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, fmt='d')
        wandb.log({"Confusion Matrix": wandb.Image(plt)})

        return {"loss": loss, "preds": preds, "labels": y_true}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        self.val_acc.reset()
        self.val_micro_f1.reset()
        self.val_macro_f1.reset()

        self.test_acc.reset()
        self.test_micro_f1.reset()
        self.test_macro_f1.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.AdamW(
            params=self.model.parameters(), lr=self.hparams.lr
        )
