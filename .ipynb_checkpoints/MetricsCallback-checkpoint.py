from pytorch_lightning.callbacks import Callback
class MetricsCallback(Callback):
    """
    A PyTorch Lightning callback for collecting metrics across epochs.

    This callback stores the training and validation loss/accuracy at
    the end of each epoch, which can be used for plotting or reporting.

    Attributes
    ----------
    train_loss : list of float
        Training loss per epoch.
    val_loss : list of float
        Validation loss per epoch.
    train_acc : list of float
        Training accuracy per epoch.
    val_acc : list of float
        Validation accuracy per epoch.

    Methods
    -------
    on_train_epoch_end(trainer, pl_module)
        Records training loss and accuracy after each epoch.
    on_validation_epoch_end(trainer, pl_module)
        Records validation loss and accuracy after each epoch.

    See Also
    --------
    pytorch_lightning.callbacks.Callback : Base class for callbacks.

    Examples
    --------
    >>> metrics_cb = MetricsCallback()
    >>> trainer = pl.Trainer(callbacks=[metrics_cb], max_epochs=3)
    >>> trainer.fit(model, train_loader, val_loader)
    >>> print(metrics_cb.train_acc)
    [0.75, 0.82, 0.88]
    """
    def __init__(self):
        self.train_loss=[]
        self.val_loss=[]
        self.train_acc=[]
        self.val_acc=[]
    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())
        self.train_acc.append(trainer.callback_metrics["train_acc"].item())
    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(trainer.callback_metrics["val_loss"].item())
        self.val_acc.append(trainer.callback_metrics["val_acc"].item())
