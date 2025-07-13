import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class OneLMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # quant stubs (must prepare externally for QAT)
        self.quant=torch.ao.quantization.QuantStub()
        self.fc1= nn.Linear(input_dim, hidden_dim//2)
        self.fc2= nn.Linear(hidden_dim//2, hidden_dim)
        self.classif= nn.Linear(hidden_dim, output_dim)
        self.dequant= torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        logits = self.classif(x)
        return self.dequant(logits)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
