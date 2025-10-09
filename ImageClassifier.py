import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
class ImageClassifier(pl.LightningModule):
    """
    A convolutional neural network for image classification.
    
    This LightningModule defines a simple CNN with three convolutional
    blocks followed by fully connected layers. It logs training and
    validation loss/accuracy during training.
    
    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for classification. Default is 2.
    lr : float, optional
        Learning rate for the Adam optimizer. Default is 1e-3.
    
    Attributes
    ----------
    model : torch.nn.Sequential
        The convolutional neural network.
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function used for training.
    
    Methods
    -------
    forward(x)
        Forward pass through the network.
    training_step(batch, batch_idx)
        Computes loss and accuracy on a training batch, logs results.
    validation_step(batch, batch_idx)
        Computes loss and accuracy on a validation batch, logs results.
    configure_optimizers()
        Returns the optimizer (Adam).
    
    See Also
    --------
    pl.LightningModule : Base class for all PyTorch Lightning models.
    
    Examples
    --------
    >>> model = ImageClassifier(num_classes=10)
    >>> x = torch.randn(8, 3, 128, 128)
    >>> out = model(x)
    >>> out.shape
    torch.Size([8, 10])
    """
    def __init__(self,num_classes=2,lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(64*16*16,128), nn.ReLU(),
            nn.Linear(128,num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x): return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits=self(x)
        loss=self.loss_fn(logits,y)
        acc=(logits.argmax(dim=1)==y).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits=self(x)
        loss=self.loss_fn(logits,y)
        acc=(logits.argmax(dim=1)==y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
