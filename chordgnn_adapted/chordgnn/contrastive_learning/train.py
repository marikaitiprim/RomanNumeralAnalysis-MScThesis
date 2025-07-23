import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from pytorch_lightning import LightningModule
from chordgnn.models.core import HGCN, MLP
from info_nce import InfoNCE #pip install info-nce-pytorch

class OnsetEdgePoolingVersion2(nn.Module):
    def __init__(self, in_channels, dropout=0):
        super(OnsetEdgePoolingVersion2, self).__init__()
        self.in_channels = in_channels
        self.trans = nn.Linear(in_channels, in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.trans.reset_parameters()

    def forward(self, x, edge_index, idx=None):
        """Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.
        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
        Return types:
            * **x** *(Tensor)* - The pooled node features.
        """
        device = x.get_device() if x.get_device()>=0 else "cpu"
        if device >= 0:
            adj = torch.sparse_coo_tensor(
                edge_index, torch.ones(len(edge_index[0])).to(device), (len(x), len(x))).to_dense().to(device)
        else:
            adj = torch.sparse_coo_tensor(
                edge_index, torch.ones(len(edge_index[0])), (len(x), len(x))).to_dense().type(x.dtype)
        adj = adj.fill_diagonal_(1)
        h = torch.mm(adj, self.trans(x)) / adj.sum(dim=1).reshape(adj.shape[0], -1)
        # add self loops to edge_index with size (2, num_edges + num_nodes)
        edge_index_sl = torch.cat([edge_index, torch.arange(x.size(0)).view(1, -1).repeat(2, 1).to(device)], dim=1)
        h = scatter(self.trans(x)[edge_index_sl[0]], edge_index_sl[1], 0, out=torch.zeros(x.shape).to(device), reduce='mean')
        if idx is not None:
            out = h[idx]
        else:
            out, idx = self.__merge_edges__(h, edge_index)
        return out, idx

    def __merge_edges__(self, x, edge_index):
        nodes_remaining = torch.ones(x.size(0), dtype=torch.long)
        nodes_discarded = torch.zeros(x.size(0), dtype=torch.long)
        nodes_discarded[torch.unique(edge_index[0])] = 1

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        for edge_idx in range(edge_index.shape[-1]):
            source = edge_index[0, edge_idx].item()
            # I source not in the remaining nodes move to the next edge.
            if not nodes_remaining[source].item():
                continue

            target = edge_index[1, edge_idx].item()
            # If target is not in the remaining nodes move to the next edge.
            if not nodes_remaining[target].item():
                continue

            # if target is not in the discarded nodes move to the next edge.
            if not nodes_discarded[target].item():
                continue

            if source == target:
                continue


            # remove the source node from the remaining nodes
            nodes_remaining[source] = 0
            # remove the target node from the discarded nodes
            nodes_discarded[target] = 0

        # We compute the new features by trimming with the remaining nodes.
        new_x = x[nodes_remaining == 1]
        return new_x, nodes_remaining

class MultiTaskMLP(nn.Module):
    def __init__(self, in_feats, n_hidden, tasks: dict, n_layers, activation=F.relu, dropout=0.5):
        super(MultiTaskMLP, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.tasks = tasks
        self.classifier = nn.ModuleDict(
            {task: MLP(in_feats, n_hidden, tdim, n_layers, activation, dropout) for task, tdim in tasks.items()}
        )

    def reset_parameters(self):
        for task in self.tasks.keys():
            self.classifier[task].reset_parameters()

    def forward(self, x):
        prediction = {}
        for task in self.tasks.keys():
            prediction[task] = self.classifier[task](x)
        return prediction

class ChordEncoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, use_jk=False):
        super(ChordEncoder, self).__init__()
        self.activation = activation
        self.spelling_embedding = nn.Embedding(49, 16)
        self.pitch_embedding = nn.Embedding(128, 16)
        self.embedding = nn.Linear(in_feats-3, 32)
        # self.embedding = nn.Linear(in_feats-1, n_hidden)
        self.encoder = HGCN(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        # self.encoder = HResGatedConv(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        # self.etypes = {"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}
        # self.encoder = HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes=self.etypes, reduction="none")
        # self.reduction = HeteroAttention(n_hidden, len(self.etypes.keys()))
        self.pool = OnsetEdgePoolingVersion2(n_hidden, dropout=dropout)
        self.proj1 = nn.Linear(n_hidden+1, n_hidden)
        self.layernorm1 = nn.BatchNorm1d(n_hidden)
        self.proj2 = nn.Linear(n_hidden, n_hidden//2)
        self.layernorm2 = nn.BatchNorm1d(n_hidden//2)
        self.gru = nn.GRU(input_size=n_hidden//2, hidden_size=int(n_hidden/2), num_layers=2, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.layernormgru = nn.LayerNorm(n_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.constant_(self.proj1.bias, 0)
        nn.init.constant_(self.proj2.bias, 0)
        self.layernormgru.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
        h_pitch = self.pitch_embedding(x[:, 0].long())
        h_spelling = self.spelling_embedding(x[:, 1].long())
        h = self.embedding(x[:, 2:-1])
        h = torch.cat([h, h_pitch, h_spelling], dim=-1)
        # h = F.normalize(self.embedding(x[:, :-1]))
        h = self.encoder(h, edge_index, edge_type)
        h = F.normalize(self.activation(h))
        h, idx = self.pool(h, onset_index, onset_idx)
        h = torch.cat([h, x[:, -1][idx].unsqueeze(-1)], dim=-1)
        h = self.layernorm1(self.activation(self.proj1(h)))
        h = self.layernorm2(self.activation(self.proj2(h)))
        if lengths is not None:
            lengths = lengths.tolist()
            h = torch.split(h, lengths, dim=0)
            h = nn.utils.rnn.pad_sequence(h, batch_first=True, padding_value=-1)
            h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True)
            h, _ = self.gru(h)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=-1)
            h = self.layernormgru(h)
            h = h.reshape(-1, h.shape[-1])
        else:
            h, _ = self.gru(h.unsqueeze(0))
            h = self.layernormgru(h.squeeze(0))
        return h

class ChordPredictionModel(nn.Module):
    def __init__(self, in_feats, n_hidden=256, tasks: dict = {
        "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
        "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
        "alto": 35, "soprano": 35}, n_layers=1, activation=F.relu, dropout=0.5, use_nade=False, use_jk=False):
        super(ChordPredictionModel, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.tasks = tasks
        self.encoder = ChordEncoder(in_feats, n_hidden, n_layers, activation=activation, dropout=dropout, use_jk=use_jk)
        self.classifier = MultiTaskMLP(n_hidden, n_hidden, tasks=tasks, n_layers=1, activation=activation, dropout=dropout)

    # def forward(self, batch):
    #     x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
    #     h = self.encoder((x, edge_index, edge_type, onset_index, onset_idx, lengths))
    #     prediction = self.classifier(h)
    #     return prediction

    def forward(self, batch, return_embedding=False):
        x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
        h = self.encoder((x, edge_index, edge_type, onset_index, onset_idx, lengths))
        if return_embedding:
            return h
        prediction = self.classifier(h)
        return prediction

    def predict(self, score):
        from chordgnn.utils import hetero_graph_from_note_array, select_features, add_reverse_edges_from_edge_index
        note_array = score.note_array(include_time_signature=True, include_pitch_spelling=True)
        onsets = torch.unique(torch.tensor(note_array["onset_beat"]))
        unique_onset_divs = torch.unique(torch.tensor(note_array["onset_div"]))
        measures = torch.tensor([[m.start.t, m.end.t] for m in score.parts[0].measures])
        measure_names = [m.number for m in score.parts[0].measures]
        s_measure = torch.zeros((len(unique_onset_divs)))
        for idx, measure_num in enumerate(measure_names):
            s_measure[torch.where((unique_onset_divs >= measures[idx, 0]) & (unique_onset_divs < measures[idx, 1]))] = measure_num
        nodes, edges = hetero_graph_from_note_array(note_array=note_array)
        note_features = select_features(note_array, "chord")
        onset_idx = unique_onsets(torch.tensor(note_array["onset_div"]))
        edge_index = torch.tensor(edges[:2, :]).long()
        x = torch.tensor(note_features).float()
        edge_type = torch.tensor(edges[2, :]).long()
        onset_edges = edge_index[:, edge_type == 0]
        edge_index, edge_type = add_reverse_edges_from_edge_index(edge_index, edge_type)
        onset_predictions = self.forward((x, edge_index, edge_type, onset_edges, onset_idx, None))
        onset_predictions["onset"] = onsets
        onset_predictions["s_measure"] = s_measure
        return onset_predictions


class UnsupervisedContrastiveLearning(LightningModule):
    def __init__(self,
                 in_feats: int,
                 n_hidden: int,
                 tasks: dict,
                 n_layers: int,
                 activation=F.relu,
                 dropout=0.5,
                 lr=0.0001,
                 weight_decay=5e-4,
                 use_jk=False,
                 weight_loss=True,
                 device=0
                 ):
        super(UnsupervisedContrastiveLearning, self).__init__()
        self.tasks = tasks
        self.save_hyperparameters()
        self.num_tasks = len(tasks.keys())
        self.module = ChordPredictionModel(
            in_feats, n_hidden, tasks, n_layers, activation, dropout, use_jk=use_jk).float().to(self.device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_roman = list()
        self.test_roman_ts = list()
        self.train_loss = InfoNCE(temperature=0.07)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        # batch_inputs, edges, edge_type, batch_labels, onset_divs, lengths = batch
        # batch_size = lengths.shape[0]
        # onset_edges = edges[:, edge_type == 0]
        # onset_idx = unique_onsets(onset_divs)

        # batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, lengths))
        
        # loss = self.train_loss(batch_pred, batch_labels)
        # self.log('train_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size) 
        # self.log("global_step", self.global_step, on_step=True, prog_bar=False, batch_size=batch_size)
        # return loss["total"]
    

        # Unpack the paired views
        batch_view1, batch_view2 = batch

        # Unpack each batch (from your collate_fn)
        x1, edge_index1, edge_type1, onset_divs1, lengths1 = batch_view1
        x2, edge_index2, edge_type2, onset_divs2, lengths2 = batch_view2

        onset_idx1 = unique_onsets(onset_divs1)
        onset_idx2 = unique_onsets(onset_divs2)

        # Encode both views
        z1 = self.module((x1, edge_index1, edge_type1, onset_idx1, onset_idx1, lengths1), return_embedding=True)
        z2 = self.module((x2, edge_index2, edge_type2, onset_idx2, onset_idx2, lengths2), return_embedding=True)

        loss = self.train_loss(z1, z2)  # Should return a dict

        self.log('train_loss', loss['total'].item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss['total']

    
    # def predict_step(self, batch, batch_idx):  #used in validation step!
    #     batch_inputs, edges, edge_type, name = batch
    #     onset_edges = edges[:, edge_type == 0]
    #     onset_idx = unique_onsets(batch_labels["onset"])
    #     batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx))
    #     return batch_pred

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW([
            {'params': self.module.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {'params': self.train_loss.parameters(), 'weight_decay': 0, "lr": self.lr}]
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


def unique_onsets(onsets):
    unique, inverse = torch.unique(onsets, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm


# class UnsupervisedContrastiveLearning(LightningModule):
#     def __init__(self, encoder, temperature=0.07, lr=0.0001, use_teacher=True):
#         super().__init__()
#         self.encoder = encoder
#         self.temperature = temperature
#         self.lr = lr
#         self.use_teacher = use_teacher
#         self.automatic_optimization = True
        
#         # Use consistent temperature
#         self.train_loss = InfoNCE(temperature=temperature)
#         self.val_loss = InfoNCE(temperature=temperature)
        
#         # Teacher model for pseudo-labeling (optional) -> fix this
#         if use_teacher:
#             self.teacher_model = ChordPrediction.load_from_checkpoint("checkpoint/epoch=99-step=23800.ckpt") #pretrained chordgnn model
#             self.teacher_model.eval()
#             for p in self.teacher_model.parameters():
#                 p.requires_grad = False
#         else:
#             self.teacher_model = None
    
#     def forward(self, graph):
#         return self.encoder(graph)
    
#     def create_teacher_mask(self, graph_1, graph_2, original_graph):
#         """Create mask based on teacher model predictions"""
#         if self.teacher_model is None:
#             return None
            
#         with torch.no_grad():
#             # Get predictions for augmented views
#             pred_1 = self.teacher_model(graph_1)
#             pred_2 = self.teacher_model(graph_2)
            
#             # Create mask: 1 where predictions differ (keep as negatives)
#             # 0 where predictions match (potential false negatives to ignore)
#             if pred_1.dim() > 1:  # Multi-class predictions
#                 pred_1 = pred_1.argmax(dim=-1)
#                 pred_2 = pred_2.argmax(dim=-1)
            
#             mask = (pred_1 != pred_2).float()
#             return mask
    
#     def training_step(self, batch, batch_idx):
#         # Expect: batch = (graph_1, graph_2, original_graph)
#         graph_1, graph_2, original_graph = batch
        
#         # Get embeddings
#         z1 = self.encoder(graph_1)
#         z2 = self.encoder(graph_2)
        
#         # Compute contrastive loss
#         if self.use_teacher:
#             mask = self.create_teacher_mask(graph_1, graph_2, original_graph)
#             # Note: Current InfoNCE library doesn't support masking
#             # You'd need to implement custom InfoNCE with masking
#             loss = self.train_loss(z1, z2)
#         else:
#             loss = self.train_loss(z1, z2)
        
#         # Logging
#         self.log("train_contrastive_loss", loss, on_epoch=True, prog_bar=True)
#         self.log("train_loss", loss, on_step=True, on_epoch=True)
        
#         return loss