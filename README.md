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

### ESM Model Dependencies

The project uses Facebook's ESM (Evolutionary Scale Modeling) protein language model:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install fair-esm  # Optional: for direct ESM access
```

### MaSIF Dependencies (Required for Surface Analysis)

For the surface analysis component (`exact_surface.ipynb`), you need to install MaSIF dependencies:

1. **Clone MaSIF Repository**:
```bash
git clone https://github.com/LPDI-EPFL/masif.git
cd masif
```

2. **Install MaSIF Dependencies**:
```bash
# Follow MaSIF installation instructions
# Key dependencies include:
- MSMS (Molecular Surface Mesh Solver)
- PyMesh
- Open3D
- scipy
- sklearn
```

3. **Set up MSMS**:
   - Download MSMS from https://cgl.ucsf.edu/msms/
   - Ensure MSMS executable is in your PATH
   - Update the `triangulation/computeMSMS.py` path in your code

### Additional Files Required

- `aminoacids.csv` - Amino acid physicochemical properties database
- `ACENet.pth` - Pre-trained model weights
- PDB files in `pdbs/` directory

## Installation

1. **Clone the repository** (or set up your project directory):
```bash
git clone <your-repository-url>
cd protein-gnn-analysis
```

2. **Create conda environment**:
```bash
conda create -n protein-gnn python=3.8
conda activate protein-gnn
```

3. **Install core dependencies**:
```bash
pip install torch torchvision torchaudio
pip install dgl
pip install pandas numpy scipy scikit-learn
pip install biopandas biopython
pip install transformers tokenizers
pip install tqdm
```

4. **Install MaSIF for surface analysis**:
```bash
# Follow MaSIF installation guide
git clone https://github.com/LPDI-EPFL/masif.git
# Configure MSMS and other surface analysis tools
```

5. **Set up directory structure**:
```
project/
├── pdbs/                    # PDB files directory
├── triangulation/           # MaSIF triangulation module
├── aminoacids.csv          # Amino acid properties
├── ACENet.pth             # Pre-trained model
├── gnn_data.ipynb         # Data processing
├── exact_surface.ipynb    # Surface analysis
├── predict.ipynb          # Prediction script
└── README.md              # This file
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
# Run exact_surface.ipynb
jupyter notebook exact_surface.ipynb
```

This will:
- Compute molecular surfaces using MSMS
- Identify surface-exposed amino acid positions
- Add `surface_index` column to your CSV

### 2. Feature Extraction and Graph Generation

Process protein structures and extract features:

```bash
# Run gnn_data.ipynb  
jupyter notebook gnn_data.ipynb
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
jupyter notebook predict.ipynb
```

This will:
- Load the pre-trained ACENet model
- Process your protein data
- Generate pHmin predictions
- Save results to CSV

## File Descriptions

### `gnn_data.ipynb`
- **Purpose**: Data preprocessing and feature extraction
- **Key Functions**:
  - `get_distance_matrix()`: Computes distance matrices from 3D coordinates
  - `generate_graph()`: Creates DGL graphs from protein structures
  - `esm_encode()`: Generates ESM embeddings for protein sequences
  - `physchem_encode()`: Adds physicochemical properties
  - `get_embedding()`: Main processing pipeline

### `exact_surface.ipynb`
- **Purpose**: Surface amino acid identification
- **Key Functions**:
  - `get_surface_aa()`: Identifies surface residues using MSMS
  - `selected_surface_aa()`: Extracts surface amino acid sequences
  - `get_surface_pos()`: Batch processes proteins for surface analysis
- **Dependencies**: Requires MaSIF installation

### `predict.ipynb`
- **Purpose**: Model inference and prediction
- **Key Components**:
  - `GraphDataset`: Custom dataset class for loading processed data
  - `GCN`: Graph Convolutional Network implementation
  - `GNNModel`: Complete model architecture
  - Prediction pipeline with pre-trained weights

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

## Example Usage

```python
# Load and process a single protein
from your_modules import get_embedding, get_surface_pos

# 1. Extract surface information
get_surface_pos('input.csv', 'output_with_surface.csv')

# 2. Generate embeddings and graphs
get_embedding('output_with_surface.csv', 'processed_data/')

# 3. Make predictions
# Run prediction notebook with processed data
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: 
   - Ensure PyTorch is installed with correct CUDA version
   - Modify device settings if GPU unavailable

2. **ESM Model Loading**:
   - First run may take time to download ESM weights
   - Ensure internet connection for model download

3. **MaSIF/MSMS Issues**:
   - Verify MSMS installation and PATH configuration
   - Check triangulation module imports

4. **Memory Issues**:
   - Reduce batch size for large proteins
   - Use CPU mode if GPU memory insufficient

### Performance Tips

- Use GPU acceleration when available
- Process proteins in batches for better efficiency
- Pre-compute surface analysis for large datasets
- Monitor memory usage during embedding generation
