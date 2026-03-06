# Protein Property Prediction using Graph Neural Networks

This project implements a Graph Neural Network (GNN) approach for predicting protein properties, specifically focusing on pH-related characteristics (pHmin). The system combines protein sequence information, 3D structural data, and surface amino acid analysis to make accurate predictions.

## Overview

The project consists of three main components:
1. **Data Processing & Feature Extraction** - Converts protein structures into graph representations with rich feature encodings
2. **Surface Analysis** - Identifies and analyzes surface-exposed amino acids using molecular surface computation
3. **Prediction Model** - Uses a trained GNN model to predict protein properties

## Features

- **Multi-modal Feature Integration**: Combines ESM protein language model embeddings with physicochemical properties
- **Graph-based Representation**: Converts protein structures to graphs based on spatial proximity
- **Surface Analysis**: Identifies surface-exposed residues using MSMS triangulation
- **GNN Architecture**: Uses Graph Convolutional Networks with attention mechanisms
- **End-to-end Pipeline**: From PDB files to property predictions

## Requirements

### Core Dependencies

```bash
# Core scientific computing
numpy
pandas
torch
dgl

# Protein analysis
biopandas
Bio (biopython)
transformers
tokenizers

# Molecular visualization and analysis
scipy
scikit-learn
tqdm
```

### Additional Files Required

- `aminoacids.csv` - Amino acid physicochemical properties database
- `ACENet.pth` - Pre-trained model weights
- PDB files in `pdbs/` directory

## Installation

1. **Clone the repository** (or set up your project directory):
```bash
git clone https://github.com/Showmake2/ACENet.git
cd ACENet
```

2. **Create conda environment**:
```bash
conda create -n ACENet python=3.8
conda activate ACENet
```

3. **Install core dependencies**:
```bash
conda install -c dglteam/label/th21_cu118 dgl
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c bioconda msms
pip install biopython IPython pandas transformers biopandas -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```




## Usage

### 1. Surface Analysis and Data Preparation

First, prepare your input CSV file with protein entries:

```csv
entry
1fhe
```

Run surface analysis to identify surface amino acids:

```bash
# Run exact_surface.py
python exact_surface.py
```

This will:
- Compute molecular surfaces using MSMS
- Identify surface-exposed amino acid positions
- Add `surface_index` column to your CSV

### 2. Feature Extraction and Graph Generation

Process protein structures and extract features:

```bash
# Run gnn_data.ipynb  
python gnn_data.py
```

This step:
- Loads PDB structures
- Generates protein sequence embeddings using ESM
- Adds physicochemical properties
- Creates graph representations based on spatial proximity
- Saves processed data in compressed format

### 3. Prediction

Make predictions using the trained model:

```bash
# Run predict.ipynb
python predict.py
```

This will:
- Load the pre-trained ACENet model
- Process your protein data
- Generate pHmin predictions
- Save results to CSV

## File Descriptions


## Model Architecture

The GNN model (`ACENet`) consists of:

1. **Feature Compression**: Linear layers to process input features
2. **Graph Convolution**: Multi-layer GCN with batch normalization and LeakyReLU
3. **Dual Processing**: Separate pathways for wildtype and surface features
4. **Global Pooling**: Sum pooling for graph-level representations
5. **Prediction Head**: Fully connected layers with dropout for final prediction

**Input Features**: 1296 dimensions (1280 ESM + 16 physicochemical)
**Hidden Dimensions**: 256
**Output**: Single value (pHmin prediction)

## Data Format

### Input CSV Format
```csv
entry,surface_index
1fhe,"[1, 5, 8, 12, 15, ...]"
```

### Output Files
- `dgl_graph.bin`: Serialized DGL graphs
- `seq_feature_*.npz`: Sequence embeddings
- `surface_aa_feature_*.npz`: Surface-specific features
- `overview_df.csv`: Metadata and file paths
