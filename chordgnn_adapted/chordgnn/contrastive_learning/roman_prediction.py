import chordgnn as st
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.set_float32_matmul_precision('medium')
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.44)
parser.add_argument('--batch_size', type=int, default=32)  #100
parser.add_argument('--lr', type=float, default=0.0015)
parser.add_argument('--weight_decay', type=float, default=0.0035)
parser.add_argument('--num_workers', type=int, default=20) 
parser.add_argument("--collection", type=str, default="all",
                choices=["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"],  help="Collection to test on.")
parser.add_argument("--predict", action="store_true", help="Obtain Predictions using wandb cloud stored artifact.")
parser.add_argument('--use_jk', action="store_true", help="Use Jumping Knowledge In graph Encoder.")
parser.add_argument('--mtl_norm', default="none", choices=["none", "Rotograd", "NADE", "GradNorm", "Neutral"], help="Which MLT optimization to use.")
parser.add_argument("--include_synth", action="store_true", help="Include synthetic data.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--use_ckpt", type=str, default=None, help="Use checkpoint for prediction.")
parser.add_argument("--num_tasks", type=int, default=11, choices=[5, 11, 14], help="Number of tasks to train on.")
parser.add_argument("--data_version", type=str, default="v1.0.0", choices=["v1.0.0", "latest"], help="Version of the dataset to use.")
parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train for.") #train loss converges at 25 epochs
parser.add_argument("--use-teacher", type=bool, default=True, help="Teacher model for contrastive learning.")

# for reproducibility
torch.manual_seed(0)
random.seed(0)
# torch.use_deterministic_algorithms(True)

args = parser.parse_args()
if isinstance(eval(args.gpus), int):
    if eval(args.gpus) >= 0:
        devices = [eval(args.gpus)]
        dev = devices[0]
    else:
        devices = None
        dev = "cpu"
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
    dev = None
n_layers = args.n_layers
n_hidden = args.n_hidden
force_reload = False
num_workers = args.num_workers
use_teacher = args.use_teacher

first_name = args.mtl_norm if args.mtl_norm != "none" else "Wloss" #Wloss is used by default (best results in paper)
name = "{}-{}x{}-lr={}-wd={}-dr={}".format(first_name, n_layers, n_hidden,
                                            args.lr, args.weight_decay, args.dropout)
weight_loss = args.mtl_norm not in ["Neutral", "Rotograd", "GradNorm"] # true by default

datamodule = st.contrastive_learning.datamodule.ContrastiveGraphDatamodule(batch_size=args.batch_size, num_workers=8, num_tasks=args.num_tasks) 

model = st.contrastive_learning.train.UnsupervisedContrastiveLearning(datamodule.features, args.n_hidden, datamodule.tasks, args.n_layers, lr=args.lr, dropout=args.dropout,
    weight_decay=args.weight_decay, use_jk=args.use_jk, device=dev, weight_loss=weight_loss, use_teacher=use_teacher)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="global_step", mode="max")
early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.02, patience=5, verbose=False, mode="min")
use_ddp = len(devices) > 1 if isinstance(devices, list) else False

logger = CSVLogger("logs", name="contrastive")
trainer = Trainer(
    max_epochs=args.n_epochs,
    accelerator="auto", devices=devices, 
    num_sanity_val_steps=1,
    logger=logger,
    plugins=DDPStrategy(find_unused_parameters=False) if use_ddp else None,
    callbacks=[checkpoint_callback],
    reload_dataloaders_every_n_epochs=5,
    )


# training
trainer.fit(model, datamodule)

df = pd.read_csv("logs/contrastive/version_0/metrics.csv") #load the appropriate metrics file
df = df[df['train_loss'].notna()]
plt.plot(df["epoch"], df["train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("training_loss.png") 