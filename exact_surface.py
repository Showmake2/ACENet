import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import warnings
from Bio import BiopythonParserWarning

# Local includes
from src.computeMSMS import computeMSMS

def get_surface_aa(path_to_pdb):
    try:
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(path_to_pdb, protonate=True)
        input_list = names1
        extracted_numbers = [x.split('_')[1] for x in input_list]
        aa_faces = list(set(extracted_numbers))
        aa_faces = [int(x) for x in aa_faces]
        return aa_faces
    except Exception as e:
        print(f"Error in computeMSMS for {path_to_pdb}: {e}")
        return None

def selected_surface_aa(surface_index, sequence):
    if surface_index is None:
        return ""
    amino_acid_positions = surface_index
    protein_sequence = sequence
    amino_acid_positions.sort()
    selected_amino_acids = [protein_sequence[pos-1] for pos in amino_acid_positions]
    surface_aa_sequence = ''.join(selected_amino_acids)
    return surface_aa_sequence

def get_protein_sequence_from_pdb(pdb_path):
    try:
        with open(pdb_path, "r") as pdb_file:
            for record in SeqIO.parse(pdb_file, "pdb-atom"):
                return str(record.seq)
    except Exception as e:
        print(f"Error reading PDB file {pdb_path}: {e}")
        return None

def is_valid_protein_sequence(sequence):
    if sequence is None:
        return False
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(amino in standard_amino_acids for amino in sequence)

def get_surface_pos(path_to_csv, output_csv):
    df = pd.read_csv(path_to_csv)
    warnings.filterwarnings("ignore")

    surface_index_list = []
    bad_data = []
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        id = row['UniProt entry']
        try:
            pdb_path = f"pdbs/{id}.pdb"
            if not os.path.exists(pdb_path):
                print(f"PDB file not found: {pdb_path}")
                bad_data.append(id)
                surface_index_list.append(None)
                continue
                
            seq = get_protein_sequence_from_pdb(pdb_path)
            if not is_valid_protein_sequence(seq):
                print(f"Invalid sequence for {id}")
                bad_data.append(id)  
                surface_index_list.append(None)
                continue
                
            surface_index = get_surface_aa(pdb_path)
            if surface_index is None:
                bad_data.append(id)
                surface_index_list.append([])
            else:
                surface_index_list.append(surface_index)
            
        except Exception as e:
            print(f"Unexpected error for {id}: {e}")
            bad_data.append(id) 
            surface_index_list.append(None)
            continue
            
    print(f"Bad data entries: {bad_data}")  
    
    if len(surface_index_list) != len(df):
        print(f"Warning: Length mismatch. DataFrame: {len(df)}, surface_index_list: {len(surface_index_list)}")
        while len(surface_index_list) < len(df):
            surface_index_list.append(None)
    
    df["surface_index"] = surface_index_list
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main():
    get_surface_pos('../benchmark_dataset_pHmin.csv', '../benchmark_dataset_pHmin.csv')

if __name__ == "__main__":
    main()