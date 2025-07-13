import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from perceptron import OneLMLP
from preprocess import get_loaders

# Hyper-Param
input_dim =  13  
hidden_dim = 128 
output_dim = 1  
learn_rate =  0.0001
batch_size =  32
epochs = 150

train_loader, test_loader = get_loaders(batch_size = batch_size)

model = OneLMLP(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim, lr=learn_rate)

trainer = pl.Trainer(max_epochs=epochs,callbacks=[TQDMProgressBar(refresh_rate=1)])

trainer.fit(model, train_loader, test_loader)

torch.save(model.state_dict(), "model_weights.pth")

