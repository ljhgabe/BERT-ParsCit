from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.bert_token_classifier import BertTokenClassifier
from src.models.components.bert_tokenizer import bert_tokenizer

from src.datamodules.components.class_label import LABEL_LIST
from src.datamodules.components.preprocess import postprocess


class BertParsCitLitModule(LightningModule):

    def __init__(
        self,
        model: BertTokenClassifier,
        lr: float = 2e-5,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = model
        self.tokenizer = bert_tokenizer

        # Calculating accuracy
        self.train_acc = Accuracy(num_classes=len(LABEL_LIST)+1, mdmc_average="global", ignore_index=len(LABEL_LIST))
        self.val_acc = Accuracy(num_classes=len(LABEL_LIST)+1, mdmc_average="global", ignore_index=len(LABEL_LIST))
        self.test_acc = Accuracy(num_classes=len(LABEL_LIST)+1, mdmc_average="global", ignore_index=len(LABEL_LIST))

        # Calculating F1 score
        self.train_f1 = F1Score(num_classes=len(LABEL_LIST)+1, mdmc_average="global", ignore_index=len(LABEL_LIST), average="micro") 
        self.val_f1 = F1Score(num_classes=len(LABEL_LIST)+1, mdmc_average="global", ignore_index=len(LABEL_LIST), average="micro")
        self.test_f1 = F1Score(num_classes=len(LABEL_LIST)+1, mdmc_average="global", ignore_index=len(LABEL_LIST), average="micro")

        # Calculating confusion matrix
        self.conf_matrix = ConfusionMatrix(num_classes=len(LABEL_LIST))

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        inputs, labels = batch, batch["labels"]
        outputs = self.forward(inputs)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        # log train metrics (not needed)
        # acc = self.train_acc(true_preds, true_targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        input_ids = batch["input_ids"]
        loss, preds, labels = self.step(batch)

        true_preds, true_labels = postprocess(
            input_ids=input_ids,
            predictions=preds,
            labels=labels,
            label_names=LABEL_LIST
        )

        # log val metrics
        acc = self.val_acc(true_preds, true_labels)
        f1 = self.val_f1(true_preds, true_labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": true_preds, "labels": true_labels}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        input_ids = batch["input_ids"]
        loss, preds, labels = self.step(batch)

        true_preds, true_labels = postprocess(
            input_ids=input_ids,
            predictions=preds,
            labels=labels,
            label_names=LABEL_LIST
        )

        # log test metrics
        acc = self.test_acc(true_preds, true_labels)
        f1 = self.test_f1(true_preds, true_labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": true_preds, "labels": true_labels}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        # self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()

        # self.train_f1.reset()
        self.val_f1.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.AdamW(
            params=self.model.parameters(), lr=self.hparams.lr
        )
