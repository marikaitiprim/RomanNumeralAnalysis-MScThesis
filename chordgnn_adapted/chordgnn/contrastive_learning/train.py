import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from pytorch_lightning import LightningModule
from chordgnn.models.core import HGCN, MLP
from info_nce import InfoNCE #pip install info-nce-pytorch
from chordgnn.models.chord import ChordPrediction

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
        self.encoder = HGCN(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        self.pool = OnsetEdgePoolingVersion2(n_hidden, dropout=dropout)
        self.proj2 = nn.Linear(n_hidden, n_hidden//2)
        self.layernorm2 = nn.BatchNorm1d(n_hidden//2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.embedding.bias, 0)
        # nn.init.constant_(self.proj1.bias, 0)
        nn.init.constant_(self.proj2.bias, 0)
        self.layernorm2.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
        h_pitch = self.pitch_embedding(x[:, 0].long())
        h_spelling = self.spelling_embedding(x[:, 1].long())

        h_other = self.embedding(x[:, 2:-1])
        h = torch.cat([h_other, h_pitch, h_spelling], dim=-1)
        h = self.encoder(h, edge_index, edge_type)
        h = F.normalize(self.activation(h))
        h, idx = self.pool(h, onset_index, onset_idx)
        if lengths is not None:
            lengths = lengths.tolist()
            h_split = torch.split(h, lengths, dim=0)
            h = torch.stack([s.mean(dim=0) for s in h_split])  # shape: [batch_size, emb_dim]
        else:
            h = h.mean(dim=0, keepdim=True)  # just in case single graph, no batching

        h = self.layernorm2(self.activation(self.proj2(h)))

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
                 device=0,
                 use_teacher=True
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
        self.use_teacher = use_teacher

        if use_teacher:
            self.teacher_model = ChordPrediction.load_from_checkpoint("chordgnn/contrastive_learning/checkpoint/epoch=99-step=23800.ckpt") #pretrained chordgnn model
            self.teacher_model.eval()
            self.teacher_model.freeze() 
            self.teacher_model.to(device) 
        else:
            self.teacher_model = None

    def pool_by_graph(self, node_embeddings, lengths):
        """Mean-pool node embeddings into graph-level (B, D) shape"""
        graph_embeddings = []
        idx = 0
        for length in lengths:
            graph_embed = node_embeddings[idx:idx+length].mean(dim=0)
            graph_embeddings.append(graph_embed)
            idx += length
        return torch.stack(graph_embeddings, dim=0)

    def get_teacher_embeddings(self, batch):
        """Returns embeddings from the frozen teacher model"""
        x, edge_index, edge_type, onset_divs, lengths = batch
        onset_edges = edge_index[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)

        with torch.no_grad():
            embeddings = self.teacher_model.module.encoder((
                x, edge_index, edge_type, onset_edges, onset_idx, lengths
            ))
        return self.pool_by_graph(embeddings, lengths)

    def create_teacher_mask(self, embedding_1, embedding_2, threshold=0.9):
        """
        Efficient cosine similarity-based mask between two embedding sets without expanding (B, B, D)
        """
        if self.teacher_model is None:
            return None

        # Normalize the embeddings
        embedding_1 = F.normalize(embedding_1, dim=1)  # (B, D)
        embedding_2 = F.normalize(embedding_2, dim=1)  # (B, D)

        # Cosine similarity = dot product for normalized vectors
        sim_matrix = torch.matmul(embedding_1, embedding_2.T)  # (B, B)

        # Mask where similarity is too high (false negatives)
        mask = sim_matrix < threshold  # True = valid negative

        # Ensure the diagonal is always included (positive pairs)
        mask.fill_diagonal_(True)

        return mask

    def masked_info_nce(self, z1, z2, mask, temperature=0.07):
        """
        z1, z2: shape (B, D)
        mask: shape (B, B) — True where negatives are valid, False = masked
        """
        B = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / temperature  # (B, B)
        sim_matrix = sim_matrix.masked_fill(~mask, -1e9)

        labels = torch.arange(B).to(z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        batch_view1, batch_view2 = batch

        x1, edge_index1, edge_type1, onset_divs1, lengths1 = batch_view1
        x2, edge_index2, edge_type2, onset_divs2, lengths2 = batch_view2

        onset_edges1 = edge_index1[:, edge_type1 == 0]
        onset_edges2 = edge_index2[:, edge_type2 == 0]

        onset_idx1 = unique_onsets(onset_divs1)
        onset_idx2 = unique_onsets(onset_divs2)

        # Encode both views
        z1 = self.module((x1, edge_index1, edge_type1, onset_edges1, onset_idx1, lengths1), return_embedding=True)
        z2 = self.module((x2, edge_index2, edge_type2, onset_edges2, onset_idx2, lengths2), return_embedding=True)

        if self.use_teacher:

            t1 = self.get_teacher_embeddings(batch_view1)
            t2 = self.get_teacher_embeddings(batch_view2)
            
            mask = self.create_teacher_mask(t1, t2)
            loss = self.masked_info_nce(z1, z2, mask)
        else:
            loss = self.train_loss(z1, z2) #the original InfoNCE loss

        batch_size = lengths1.shape[0]
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size) 
        self.log("global_step", self.global_step, on_step=True, prog_bar=False, batch_size=batch_size)
        return loss

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
            "monitor": "train_loss"
        }


def unique_onsets(onsets):
    unique, inverse = torch.unique(onsets, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm