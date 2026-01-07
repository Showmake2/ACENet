"""
Protein pH Prediction - GNN Model Inference Module
Function: Predict optimal pH values of proteins using ACENet model
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
from tqdm import tqdm

# ============== Configuration ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 4  # Adjust based on actual CPU cores


# ============== Dataset ==============
class GraphDataset(torch.utils.data.Dataset):
    """
    Graph Dataset Class
    
    Load preprocessed graph structures and node features
    """
    def __init__(self, dir_path, indexes=None, add_self_loop=False):
        super().__init__()
        self.dir_path = dir_path
        self.add_self_loop = add_self_loop
        
        # Load graphs and metadata
        self.graphs, _ = dgl.load_graphs(os.path.join(dir_path, 'dgl_graph.bin'))
        self.df = pd.read_csv(os.path.join(dir_path, 'overview_df.csv'), index_col=0)
        
        self.indexes = indexes if indexes is not None else self.df.index

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        idx = self.indexes[i]
        row = self.df.loc[idx]
        
        # 1. Load graph structure
        graph = self.graphs[row.graph_index].clone()
        if self.add_self_loop:
            graph = dgl.add_self_loop(graph)
        
        # 2. Load sequence features (ESM 1280-dim + physicochemical 16-dim)
        seq_feature = np.load(os.path.join(self.dir_path, row.seq_feature_path))
        seq = torch.tensor(seq_feature['seq'])
        seq = torch.cat([seq[:, :1280], seq[:, -16:]], dim=1)  # Select specific dimensions
        
        # 3. Load surface residue features
        surface_feature = np.load(os.path.join(self.dir_path, row.surface_aa_feature_path))
        surface_aa_seq = torch.tensor(surface_feature['surface_aa_seq'])
        surface_aa_seq = torch.cat([surface_aa_seq[:, :1280], surface_aa_seq[:, -16:]], dim=1)
        surface_pos = surface_feature['surface_pos']
        
        # 4. Set graph node features
        graph.ndata['seq'] = seq
        graph.ndata['surface_aa_seq'] = surface_aa_seq
        graph.ndata['surface_pos'] = torch.from_numpy(surface_pos)
        
        # 5. Get label
        label = row.get('pHmin', np.nan)
        label_valid = not np.isnan(label)
        
        return graph, label, label_valid, idx


# ============== Model Definition ==============
class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network
    
    Feature: Concatenate outputs from each layer, forming JK-Net style feature aggregation
    """
    def __init__(self, hidden_dim, layer_num=3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for _ in range(layer_num):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.activations.append(nn.LeakyReLU())
        
        self.layer_num = layer_num
        self.out_dim = hidden_dim * layer_num  # Note: does not include initial features
    
    def forward(self, g, h):
        """
        Args:
            g: DGL graph
            h: Node features (N, hidden_dim)
        Returns:
            Concatenated multi-layer features (N, hidden_dim * layer_num)
        """
        hs = [h]  # Include initial features
        
        for conv, bn, act in zip(self.convs, self.batch_norms, self.activations):
            h = conv(g, h)
            h = bn(h)
            h = act(h)
            hs.append(h)
        
        return torch.cat(hs, dim=-1)


class GNNModel(nn.Module):
    """
    ACENet: Protein pH Prediction Model
    
    Architecture:
        1. Linear layer to compress input features
        2. GCN to extract graph features
        3. Global pooling + MLP for prediction
    """
    def __init__(self, in_dim=1296, hidden_dim=256, dropout_rate=0.5):
        super().__init__()
        
        # Feature compression layer (using 'comp' to match saved model weights)
        self.comp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Graph convolutional layer
        self.gcn = GCN(hidden_dim, layer_num=3)
        
        # Prediction head (using 'head' to match saved model weights)
        # GCN output: hidden_dim * 3 = 768 (based on checkpoint)
        # Concatenate wildtype and surface: 768 * 2 = 1536
        # But checkpoint shows input to head.1 is 2048, so actual GCN output is 1024 per branch
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 768),  # Match checkpoint dimensions
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 1),
        )
    
    def forward(self, g, wildtype_seq, surface_aa_seq, surface_pos):
        """
        Args:
            g: Batched graph
            wildtype_seq: Full sequence features (N, 1296)
            surface_aa_seq: Surface residue features (N, 1296)
            surface_pos: Surface position mask (N,)
        Returns:
            pH prediction values (batch_size,)
        """
        # 1. Feature compression
        wildtype_h = self.comp(wildtype_seq)
        surface_h = self.comp(surface_aa_seq)
        
        # 2. Graph convolution
        wildtype_h = self.gcn(g, wildtype_h)
        surface_h = self.gcn(g, surface_h)
        
        # 3. Global pooling (graph-level summation)
        with g.local_scope():
            g.ndata['h'] = wildtype_h
            wildtype_graph = dgl.readout_nodes(g, 'h', op='sum')
        
        with g.local_scope():
            g.ndata['h'] = surface_h
            surface_graph = dgl.readout_nodes(g, 'h', op='sum')
        
        # 4. Concatenate and predict
        h_combined = torch.cat([wildtype_graph, surface_graph], dim=-1)
        pred = self.head(h_combined).squeeze(-1)
        
        return pred


# ============== Inference Function ==============
def predict(model, dataloader, device):
    """
    Batch inference
    
    Returns:
        predictions: List of predicted values
        indices: Corresponding original indices
    """
    model.eval()
    predictions = []
    indices = []
    
    with torch.no_grad():
        for graph, label, label_valid, original_index in tqdm(dataloader, desc="Predicting"):
            graph = graph.to(device)
            
            # Get node features
            seq = graph.ndata['seq']
            surface_aa_seq = graph.ndata['surface_aa_seq']
            surface_pos = graph.ndata['surface_pos']
            
            # Predict
            pred = model(graph, seq, surface_aa_seq, surface_pos)
            
            predictions.extend(pred.cpu().numpy().tolist())
            
            # Handle original_index - it can be a tuple, list, or tensor
            if isinstance(original_index, (tuple, list)):
                indices.extend(original_index)
            elif isinstance(original_index, torch.Tensor):
                indices.extend(original_index.cpu().numpy().tolist())
            else:
                # Single value
                indices.append(original_index)
    
    return predictions, indices


# ============== Main Program ==============
def main(data_dir, model_path, output_csv=None):
    """
    Main inference pipeline
    
    Args:
        data_dir: Data directory (containing dgl_graph.bin and overview_df.csv)
        model_path: Model weights path
        output_csv: Output CSV path (optional)
    """
    # 1. Load model
    print(f"Loading model from {model_path}...")
    model = GNNModel(in_dim=1296, hidden_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 2. Prepare data
    print(f"Loading data from {data_dir}...")
    dataset = GraphDataset(data_dir)
    dataloader = dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    # 3. Inference
    print("Running inference...")
    predictions, indices = predict(model, dataloader, DEVICE)
    
    # 4. Save results
    if output_csv:
        df = pd.read_csv(os.path.join(data_dir, 'overview_df.csv'), index_col=0)
        df['pHmin_pred'] = np.nan
        for idx, pred in zip(indices, predictions):
            df.loc[idx, 'pHmin_pred'] = pred
        df.to_csv(output_csv)
        print(f"Results saved to {output_csv}")
    
    return predictions, indices


# ============== Run ==============
if __name__ == "__main__":
    # Configure paths
    DATA_DIR = "1fhe"
    MODEL_PATH = "ACENet.pth"
    OUTPUT_CSV = "1fhe_predictions.csv"
    
    predictions, indices = main(DATA_DIR, MODEL_PATH, OUTPUT_CSV)
    
    print(f"\nPredictions: {predictions[:5]}...")  # Print first 5 predictions