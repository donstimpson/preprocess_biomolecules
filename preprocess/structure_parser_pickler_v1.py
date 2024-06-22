import warnings
from Bio import PDB
from Bio.PDB import PDBParser
import numpy as np
import os
import re
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pickle




"""
    Hi! This python script is written to preprocess biomolecular systems for a neural network!
    The goal is to condense the data into as few features as possible, so I've kept only:
        -partial charges of atoms
        -coordinates of atoms (cartesian)
        -covalent bonds between atoms

    I'll be writing another version of this code at some point in the near future that contains
    bond orders and test out which one works better with a simple neural network.

    The files you'll need are a structure file in pdb format, and the itp files for each novel
    molecule in the pdb file. That means that this should ideally work both for individual molecules
    and systems of molecules as well. Because there may be more than one itp file, the code only asks
    for the directory where they're located and does the rest itself.

    I'm working on a larger project and this was step one, I thought it'd be useful to share with anyone
    else in the field to speed up ML algorithms by minimizing data. I hope it can help you out!!

    Best,
    Don Stimpson

    P.S. I've left debugging print statements that I've been using while putting this together, I'm just commenting
    them out in case I have to use them later, so sorry for the code clutter! Once I've settled on a final version
    of this code I'll remove them but, until then, feel free to use them if needed.

"""


###___________USER INPUT__________________________________________###

#replace these with the path to your itp files and your pdb file
itp_directory_path = '/Users/don/ML_solo/SlowKI/itp'
pdb_file_path = 'lipids.pdb'


###_______________________________________________________________###







# Suppress Biopython warnings
#from Bio import BiopythonWarning
#warnings.simplefilter('ignore', BiopythonWarning)

def parse_pdb_file(pdb_file_path):
    """
    Parses the structure (.pdb) file to build 3 dictionaries:
        -atom_coords
        -residue_atom_map
        -residue_to_atoms

    """
    atom_coords = {}
    residue_atom_map = {}
    residue_to_atoms = {}

    with open(pdb_file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:21].strip()  # Residue name
                residue_seq = int(line[22:26].strip())  # Residue sequence number
                coords = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                atom_coords[atom_id] = coords
                residue_atom_map[(residue_name, residue_seq, atom_name)] = atom_id

                if residue_name not in residue_to_atoms:
                    residue_to_atoms[residue_name] = []
                residue_to_atoms[residue_name].append((atom_id, atom_name, residue_seq))

    # Debugging: Print the residue_atom_map
    #print("Residue Atom Map:")
    #for key, value in residue_atom_map.items():
        #print(key, value)

    return atom_coords, residue_atom_map, residue_to_atoms

def get_itp_files_from_directory(directory_path):
    """The name kind of says it all here, doesn't it?"""
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.itp')]

def parse_itp_file(itp_file_path):
    """
    For each itp file, thi

    """
    atoms = {}
    bonds = []
    partial_charges = {}

    with open(itp_file_path, 'r') as file:
        lines = file.readlines()

    atom_section = False
    bond_section = False

    for line in lines:
        line = line.strip()
        if line.startswith(';') or not line:
            continue

        if re.match(r'\[\s*atoms\s*\]', line):
            atom_section = True
            bond_section = False
            continue
        elif re.match(r'\[\s*bonds\s*\]', line):
            atom_section = False
            bond_section = True
            continue
        elif re.match(r'\[.+\]', line):
            atom_section = False
            bond_section = False
            continue

        if atom_section:
            parts = line.split()
            if len(parts) >= 7:
                atom_id = int(parts[0])
                residue_id = parts[3]  # residu (4th column, zero-indexed 3)
                atom_name = parts[4]  # atom (5th column, zero-indexed 4)
                charge = float(parts[6])
                atoms[atom_id] = (residue_id, atom_name)
                partial_charges[atom_id] = charge

        if bond_section:
            parts = line.split()
            if len(parts) >= 3:
                atom1 = int(parts[0])
                atom2 = int(parts[1])
                bond_type = int(parts[2])
                bonds.append((atom1, atom2, bond_type))

    return atoms, bonds, partial_charges

def parse_itp_directory(directory_path, residue_to_atoms):

    """
    Collects the itp information for all itp files in the specified directory. This is a parent function
    of "get_itp_files_from_directory" and "parse_itp_file"

    """

    itp_files = get_itp_files_from_directory(directory_path)
    all_atoms = {}
    all_bonds = []
    all_partial_charges = {}

    for residue_name in residue_to_atoms:
        matching_itp_files = [f for f in itp_files if residue_name in f]
        if matching_itp_files:
            itp_file = matching_itp_files[0]
            atoms, bonds, partial_charges = parse_itp_file(itp_file)
            all_atoms[residue_name] = atoms
            all_bonds.extend(bonds)
            all_partial_charges[residue_name] = partial_charges

    # Debugging: Print the all_atoms dictionary
    #print("All Atoms Dictionary:")
    #for residue_name, atoms in all_atoms.items():
        #for key, value in atoms.items():
            #print(residue_name, key, value)

    return all_atoms, all_bonds, all_partial_charges

def create_node_attributes(residue_to_atoms, all_atoms, all_partial_charges, atom_coords, residue_atom_map):

    """
    Creates a dictionary of node attributes for the preprocessed graph for a GNN. the dictionary has partial charges and
    coordinates. It iterates through all atoms to provide a comprehensive list.

    """


    node_attributes = {}
    missing_residues_atoms = []

    for residue_name, atoms in residue_to_atoms.items():
        for atom_id, atom_name, residue_seq in atoms:
            for itp_atom_id, (itp_residue_name, itp_atom_name) in all_atoms[residue_name].items():
                if itp_atom_name == atom_name:
                    if itp_atom_id in all_partial_charges[residue_name] and atom_id in atom_coords:
                        charge = all_partial_charges[residue_name][itp_atom_id]
                        coords = atom_coords[atom_id]
                        node_attributes[atom_id] = {'charge': charge, 'coords': coords}
                    else:
                        if itp_atom_id not in all_partial_charges[residue_name]:
                            missing_residues_atoms.append((residue_name, atom_name, "Partial charge missing"))
                        if atom_id not in atom_coords:
                            missing_residues_atoms.append((residue_name, atom_name, "Coordinates missing"))
                    break

    # Debugging: Print the node attributes
    #print("Node Attributes:")
    #for key, value in node_attributes.items():
        #print(key, value)

    if missing_residues_atoms:
        print("Missing residues/atoms:")
        for res_atom in missing_residues_atoms:
            print(f"Residue {res_atom[0]} and atom {res_atom[1]} - {res_atom[2]}")

    return node_attributes


def create_adjacency_matrix_template(bonds, num_atoms):
    """
    again, pretty straightforward, just builds the template for the adjacency matrix
    """
    adjacency_matrix = np.zeros((num_atoms, num_atoms))

    for bond in bonds:
        atom1, atom2, bond_type = bond
        adjacency_matrix[atom1 - 1, atom2 - 1] = bond_type
        adjacency_matrix[atom2 - 1, atom1 - 1] = bond_type

    return adjacency_matrix

def create_full_adjacency_matrix(residue_to_atoms, all_atoms, all_bonds):
    """
    creates a complete adjacency matrix for the bond data. This should output a matrix of size (X by X), where
    X is the number of atoms in your system. That's a big matrix and a lot of zeros though, so we'll make it a
    sparse matrix later

    """

    # Calculate total number of atoms
    total_atoms = sum(len(atoms) for atoms in residue_to_atoms.values())
    full_adjacency_matrix = np.zeros((total_atoms, total_atoms))

    current_offset = 0
    for residue_name, atoms in residue_to_atoms.items():
        num_atoms = len(atoms)
        residue_bonds = [bond for bond in all_bonds if bond[0] in all_atoms[residue_name] and bond[1] in all_atoms[residue_name]]
        max_atom_id = max(max(bond[0], bond[1]) for bond in residue_bonds)

        # Create adjacency matrix template for the current residue type
        adjacency_matrix_template = create_adjacency_matrix_template(residue_bonds, max_atom_id)

        # Insert the template into the full adjacency matrix at the correct positions
        for i in range(0, num_atoms, max_atom_id):
            full_adjacency_matrix[current_offset:current_offset + max_atom_id,
                                  current_offset:current_offset + max_atom_id] = adjacency_matrix_template
            current_offset += max_atom_id

    return full_adjacency_matrix



def validate_adjacency_matrix(all_atoms, all_bonds, adjacency_matrix):

    """
    This just performs some validations to make sure that the adjacency matrix generation worked as expected

    """
    num_atoms = sum(len(atoms) for atoms in all_atoms.values())

    assert np.allclose(adjacency_matrix, adjacency_matrix.T), "Adjacency matrix is not symmetric"

    expected_num_bonds = len(all_bonds)
    actual_num_bonds = 0

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if adjacency_matrix[i, j] > 0:
                actual_num_bonds += 1

    assert actual_num_bonds == expected_num_bonds, f"Number of bonds does not match (expected: {expected_num_bonds}, actual: {actual_num_bonds})"

    for (i, j, bond_type) in all_bonds:
        i -= 1
        j -= 1
        assert adjacency_matrix[i, j] == bond_type, f"Bond mismatch at ({i}, {j})"
        assert adjacency_matrix[j, i] == bond_type, f"Bond mismatch at ({j}, {i})"

    print("Validation passed: Adjacency matrix is correct")


def save_matrices(sparse_matrix, feature_matrix):
    sp.save_npz('sparse_matrix.npz', sparse_matrix)
    with open('feature_matrix.pkl', 'wb') as f:
        pickle.dump(feature_matrix, f)

def load_matrices():
    sparse_matrix = sp.load_npz('sparse_matrix.npz')
    with open('feature_matrix.pkl', 'rb') as f:
        feature_matrix = pickle.load(f)
    return sparse_matrix, feature_matrix


def main(itp_directory_path, pdb_file_path):
    atom_coords, residue_atom_map, residue_to_atoms = parse_pdb_file(pdb_file_path)
    all_atoms, all_bonds, all_partial_charges = parse_itp_directory(itp_directory_path, residue_to_atoms)
    node_attributes = create_node_attributes(residue_to_atoms, all_atoms, all_partial_charges, atom_coords, residue_atom_map)

    full_adjacency_matrix = create_full_adjacency_matrix(residue_to_atoms, all_atoms, all_bonds)
    sparse_adjacency_matrix = csr_matrix(full_adjacency_matrix)

    feature_list = []
    for node_id, attributes in sorted(node_attributes.items()):
        charge = attributes['charge']
        coords = attributes['coords']
        feature_list.append([charge] + coords.tolist())

    feature_matrix = torch.tensor(feature_list, dtype=torch.float)

    print(f"Size of the Adjacency Matrix: {sparse_adjacency_matrix.shape}")

    return sparse_adjacency_matrix, feature_matrix



sparse_adjacency_matrix, feature_matrix = main(itp_directory_path, pdb_file_path)

save_matrices(sparse_adjacency_matrix, feature_matrix)

print("Data generation and pickling completed.")


# Print results to verify
print(f"Sparse Adjacency Matrix:\n{sparse_adjacency_matrix}")
print(f"Feature Matrix:\n{feature_matrix}")
print(f"Feature Matrix Dimensions:{feature_matrix.shape}")

#trying to use half-precision to save data. Not sure if worth it yet, will test when training
"""float16_tensor = feature_matrix.to(torch.float16)
print(float16_tensor)
difference = feature_matrix - float16_tensor.to(torch.float32)
max_difference = difference.abs().max()
print(f'Maximum difference: {max_difference}')
"""
