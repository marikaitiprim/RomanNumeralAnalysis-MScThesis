from chordgnn.models.chord import ChordPrediction
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from info_nce import InfoNCE #pip install info-nce-pytorch


class UnsupervisedContrastiveLearning(LightningModule):
    def __init__(self, encoder, temperature=0.07, lr=0.0001, use_teacher=True):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.lr = lr
        self.use_teacher = use_teacher
        self.automatic_optimization = True
        
        # Use consistent temperature
        self.train_loss = InfoNCE(temperature=temperature)
        self.val_loss = InfoNCE(temperature=temperature)
        
        # Teacher model for pseudo-labeling (optional) -> fix this
        if use_teacher:
            self.teacher_model = ChordPrediction.load_from_checkpoint("checkpoint/epoch=99-step=23800.ckpt") #pretrained chordgnn model
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        else:
            self.teacher_model = None
    
    def forward(self, graph):
        return self.encoder(graph)
    
    def create_teacher_mask(self, graph_1, graph_2, original_graph):
        """Create mask based on teacher model predictions"""
        if self.teacher_model is None:
            return None
            
        with torch.no_grad():
            # Get predictions for augmented views
            pred_1 = self.teacher_model(graph_1)
            pred_2 = self.teacher_model(graph_2)
            
            # Create mask: 1 where predictions differ (keep as negatives)
            # 0 where predictions match (potential false negatives to ignore)
            if pred_1.dim() > 1:  # Multi-class predictions
                pred_1 = pred_1.argmax(dim=-1)
                pred_2 = pred_2.argmax(dim=-1)
            
            mask = (pred_1 != pred_2).float()
            return mask
    
    def training_step(self, batch, batch_idx):
        # Expect: batch = (graph_1, graph_2, original_graph)
        graph_1, graph_2, original_graph = batch
        
        # Get embeddings
        z1 = self.encoder(graph_1)
        z2 = self.encoder(graph_2)
        
        # Compute contrastive loss
        if self.use_teacher:
            mask = self.create_teacher_mask(graph_1, graph_2, original_graph)
            # Note: Current InfoNCE library doesn't support masking
            # You'd need to implement custom InfoNCE with masking
            loss = self.train_loss(z1, z2)
        else:
            loss = self.train_loss(z1, z2)
        
        # Logging
        self.log("train_contrastive_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
        return loss