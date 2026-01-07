"""
Protein pKa Prediction - Feature Extraction Module
Function: Extract ESM2 sequence features + physicochemical features + graph structure from PDB files
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from Bio import SeqIO

# ============== Configuration ==============
DISTANCE_THRESHOLD = 8.0  # Residue contact distance threshold (Å)
ESM_LAYER = 33            # ESM2 feature extraction layer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== Amino Acid Encoding ==============
AA_MAP = {
    'VAL': 'V', 'PRO': 'P', 'ASN': 'N', 'GLU': 'E', 'ASP': 'D',
    'ALA': 'A', 'THR': 'T', 'SER': 'S', 'LEU': 'L', 'LYS': 'K',
    'GLY': 'G', 'GLN': 'Q', 'ILE': 'I', 'PHE': 'F', 'CYS': 'C',
    'TRP': 'W', 'ARG': 'R', 'TYR': 'Y', 'HIS': 'H', 'MET': 'M'
}

# ============== Load Model ==============
def load_esm_model():
    """Load ESM2 pre-trained model"""
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model.to(DEVICE)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

# ============== Utility Functions ==============
def get_distance_matrix(coords):
    """Calculate Euclidean distance matrix between coordinate points"""
    diff = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def get_protein_sequence_from_pdb(pdb_path):
    """Extract protein sequence from PDB file"""
    ppdb = PandasPdb().read_pdb(pdb_path)
    atom_df = ppdb.df['ATOM']
    ca_df = atom_df[atom_df['atom_name'] == 'CA'].drop_duplicates('residue_number')
    seq = ''.join([AA_MAP.get(res, 'X') for res in ca_df['residue_name']])
    return seq


def generate_graph(pdb_path, distance_threshold=DISTANCE_THRESHOLD):
    """Generate residue contact graph from PDB file"""
    ppdb = PandasPdb().read_pdb(pdb_path)
    atom_df = ppdb.df['ATOM']
    
    # Calculate centroid coordinates for each residue
    residue_df = atom_df.groupby('residue_number', as_index=False)[
        ['x_coord', 'y_coord', 'z_coord']
    ].mean().sort_values('residue_number')
    
    coords = residue_df[['x_coord', 'y_coord', 'z_coord']].values
    distance_matrix = get_distance_matrix(coords)
    
    # Build adjacency matrix
    adj = distance_matrix < distance_threshold
    u, v = np.nonzero(adj)
    graph = dgl.graph((torch.from_numpy(u), torch.from_numpy(v)), num_nodes=len(coords))
    
    return graph

# ============== Feature Encoding ==============
def load_aa_properties(csv_path='aminoacids.csv'):
    """Load and normalize amino acid physicochemical properties"""
    aa_props = pd.read_csv(csv_path).set_index('Letter')
    
    feature_cols = [
        'Molecular Weight', 'Residue Weight', 'pKa1', 'pKb2', 'pl4',
        'H', 'VSC', 'P1', 'P2', 'SASA', 'NCISC',
        'carbon', 'hydrogen', 'nitrogen', 'oxygen', 'sulfur'
    ]
    aa_props = aa_props[feature_cols].fillna(aa_props[feature_cols].mean())
    
    # Min-Max normalization
    aa_props = (aa_props - aa_props.min()) / (aa_props.max() - aa_props.min())
    
    return aa_props

AA_PROPS = load_aa_properties()


def esm_encode(seq, model, batch_converter):
    """ESM2 sequence encoding -> (seq_len, 1280)"""
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=True)
    
    # Remove special tokens at beginning and end
    return results["representations"][ESM_LAYER].squeeze()[1:-1, :].cpu().numpy()


def physchem_encode(seq):
    """Physicochemical feature encoding -> (seq_len, 16)"""
    return AA_PROPS.loc[list(seq)].values.astype(np.float32)

# ============== Main Processing Function ==============
def get_embedding(path_to_csv, root_dir, pdb_dir='pdbs'):
    """
    Batch process proteins and generate feature files
    
    Input:
        path_to_csv: CSV file containing 'entry' and 'surface_index' columns
        root_dir: Output directory
        pdb_dir: PDB file directory
    
    Output:
        - overview_df.csv: Updated data table
        - dgl_graph.bin: DGL graph file
        - seq_feature_*.npz: Sequence features
        - surface_aa_feature_*.npz: Surface residue features
    """
    warnings.filterwarnings('ignore')
    os.makedirs(root_dir, exist_ok=True)
    
    # Load model
    model, batch_converter = load_esm_model()
    
    # Read data
    df = pd.read_csv(path_to_csv)
    df['surface_aa_feature_path'] = None
    df['seq_feature_path'] = None
    df['graph_index'] = None
    
    graphs = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            entry = row['entry']
            pdb_path = os.path.join(pdb_dir, f"{entry}.pdb")
            
            # 1. Generate graph structure
            seq = get_protein_sequence_from_pdb(pdb_path)
            graph = generate_graph(pdb_path)
            graphs.append(graph)
            graph_index = len(graphs) - 1
            
            # 2. Encode sequence features (ESM + physicochemical)
            esm_feat = esm_encode(seq, model, batch_converter)
            physchem_feat = physchem_encode(seq)
            seq_feat = np.concatenate([esm_feat, physchem_feat], axis=-1)  # (L, 1296)
            
            # 3. Save sequence features
            seq_feature_path = f'seq_feature_{graph_index}.npz'
            np.savez_compressed(os.path.join(root_dir, seq_feature_path), seq=seq_feat)
            
            # 4. Process surface residue features
            surface_indices = eval(row['surface_index'])
            surface_mask = np.zeros(graph.num_nodes())
            for idx in surface_indices:
                surface_mask[idx - 1] = 1  # Convert to 0-indexed
            
            surface_feat = seq_feat.copy()
            surface_feat[surface_mask == 0] = 0  # Zero out non-surface residues
            
            surface_aa_feature_path = f'surface_aa_feature_{index}.npz'
            np.savez_compressed(
                os.path.join(root_dir, surface_aa_feature_path),
                surface_aa_seq=surface_feat,
                surface_pos=surface_mask
            )
            
            # 5. Update DataFrame
            df.loc[index, 'graph_index'] = graph_index
            df.loc[index, 'seq_feature_path'] = seq_feature_path
            df.loc[index, 'surface_aa_feature_path'] = surface_aa_feature_path
            
        except Exception as e:
            print(f"Error processing {row.get('entry', index)}: {e}")
            df = df.drop(index)
            continue
    
    # Save results
    df.to_csv(os.path.join(root_dir, 'overview_df.csv'), index=False)
    dgl.save_graphs(os.path.join(root_dir, 'dgl_graph.bin'), graphs)
    
    print(f"Processing complete! Total {len(graphs)} proteins")
    return df


# ============== Run ==============
if __name__ == "__main__":
    get_embedding('1fhe.csv', '1fhe')