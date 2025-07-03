# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The dataset class and customized collate function for preparing dataset that can be handled by DualBind."""

import os
import logging
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from Bio import PDB
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from Bio.PDB import Residue
from chemprop.features import mol2graph
from rdkit import Chem
from rdkit.Chem import PandasTools

from .constants import ALPHABET, ATOM_TYPES, RES_ATOM14, PROTEIN_LETTERS_3TO1


class NoWatersSelect(PDB.Select):
    """ Class to select everything but water molecules """
    def accept_residue(self, residue):
        if residue.get_resname() in ['HOH', 'WAT']:
            return 0
        else:
            return 1


def mol2coords(binder_mol) -> torch.Tensor:
    """
    Convert molecular conformer to coordinates tensor.

    Args:
        binder_mol: An RDKit molecule object with 3D conformer information.

    Returns:
        torch.Tensor: A tensor of shape (num_atoms, 3) containing the x, y, z coordinates of each atom in the molecule.
    """
    conf = binder_mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i, _ in enumerate(binder_mol.GetAtoms())]
    return torch.tensor([[p.x, p.y, p.z] for p in coords]).float()


def strip_water_molecule_from_structure(structure: PDB.Structure.Structure) -> PDB.Structure.Structure:
    """
    A function for stripping water molecules from a PDB structure.

    Args:
        strcture (PDB.Structure.Structure): a PDB structure.

    Returns:
        strcture (PDB.Structure.Structure): a PDB structure without water molecules.
    """
    with NamedTemporaryFile(mode="w+") as f:  # reload structure to update len(structure[0])
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(f.name, select=NoWatersSelect())

        parser = PDB.PDBParser(QUIET=True)
        new_structure = parser.get_structure(structure.id, f.name)

    return new_structure


def featurize_tgt(batch: List[Dict[str, Any]], vocab: List = ALPHABET) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A function for preparing target pocket info.

    Args:
        batch (List[Dict]): a list of data (Dict) as constructed by the __init__ function of the DualBindDataset class.

    Returns:
       X (torch.Tensor): a tensor with shape (batch_size, max_residue_length_in_batch, 14, 3), which represents the target atom coordinates.
       S (torch.Tensor): a tensor with shape (batch_size, max_residue_length_in_batch), which represents the residue types.
       A (torch.Tensor): a tensor with shape (batch_size, max_residue_length_in_batch, 14), which represents the target atom types.
    """
    B = len(batch)
    L_max = max([len(b['pocket_seq']) for b in batch])
    X = torch.zeros([B, L_max, 14, 3])
    S = torch.zeros([B, L_max]).long()
    A = torch.zeros([B, L_max, 14]).long()

    # Build the batch
    for i, b in enumerate(batch):
        l = len(b['pocket_seq'])
        indices = torch.tensor([vocab.index(a) for a in b['pocket_seq']])
        S[i, :l] = indices
        X[i, :l] = b['pocket_coords']
        A[i, :l] = b['pocket_atypes']

    return X, S, A

def get_residue_atoms(residue: Residue, expected_atoms: List[str]) -> np.ndarray:
    """
    Get the coordinates of atoms for a given residue. Appended values are zeros.

    Args:
        residue (Bio.PDB.Residue): A residue object from the Biopython PDB module.
        expected_atoms (List[str]): A list of expected atom names in the residue.

    Returns:
        np.ndarray: A numpy array of shape (14, 3) containing the coordinates of the atoms.
                    If an expected atom is not found in the residue, its coordinates are set to zeros.
    """
    coords = np.full((14, 3), 0, dtype=np.float32)  # Prepare a zero-filled array for coordinates
    atom_dict = {atom.get_name(): atom.get_coord() for atom in residue.get_atoms()}
    for i, atom_name in enumerate(expected_atoms):
        if atom_name in atom_dict:
            coords[i] = atom_dict[atom_name]
    return coords

def preprocess_entry(entry: Dict[str, Any], patch_size: int) -> Dict[str, Any]:
    """
    Preprocess a single entry from the dataset to prepare it for the DualBind model.

    Args:
        entry (Dict): A dictionary containing the data for one protein-ligand complex.
        patch_size (int): The number of residues to be considered as in pocket.

    Returns:
        Dict: The preprocessed entry with additional fields for model input.
    """

    # make target
    entry['target_coords'] = torch.tensor(entry['target_coords']).float()
    entry['target_atypes'] = torch.tensor(
            [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
        )
    
    # make binder
    entry['binder_coords'] = mol2coords(entry['binder_mol'])

    # make pocket
    dist = entry['target_coords'][:, 1] - entry['binder_coords'].mean(dim=0, keepdims=True)
    entry['pocket_idx'] = dist.norm(dim=-1).sort().indices[:patch_size].sort().values  # TODO [sichu] check if pocket size exceeds patch size
    entry['pocket_seq'] = ''.join([entry['target_seq'][i] for i in entry['pocket_idx'].tolist()])
    entry['pocket_coords'] = entry['target_coords'][entry['pocket_idx']]
    entry['pocket_atypes'] = entry['target_atypes'][entry['pocket_idx']]

    return entry


def pl_collate_fn(batch: List[Dict], max_residue_atoms: int, aa_size: int, use_affinity_label: bool) -> Tuple:
    """
    Custom collate function for handling batches of protein-ligand data for DualBind model.

    Args:
        batch (List[Dict]): a list of data (Dict) as constructed by the __init__ function.
        max_residue_atoms (int): maximum number of atoms in each residue for batching

    Returns:
        batched_binder (Tuple[torch.Tensor, List[Mol], torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second list is a list of RDKit molecules. The third tensor is a mask for indicating ligand atoms.
        batched_target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket.
        affinity (Optional[float]): the batched affinity values.
        use_affinity_label (bool): whether to use affinity label or not.
    """
    mols = [entry['binder_mol'] for entry in batch]
    N = max([mol.GetNumAtoms() for mol in mols])
    bind_X = torch.zeros([len(batch), N, max_residue_atoms, 3])
    bind_A = torch.zeros([len(batch), N, max_residue_atoms]).long()
    tgt_X, tgt_S, tgt_A = featurize_tgt(batch)
    tgt_S = torch.zeros([tgt_S.size(0), tgt_S.size(1), aa_size])
    for i, b in enumerate(batch):
        L = b['binder_mol'].GetNumAtoms()
        bind_X[i, :L, 1, :] = b['binder_coords']
        bind_A[i, :L, 1] = 1
        L = len(b['pocket_seq'])
        residue_embedding = torch.zeros(len(b['pocket_seq']), aa_size)
        for j, aa in enumerate(b['pocket_seq']):
            residue_embedding[j, ALPHABET.index(aa)] = 1  # One-hot embedding for residue
        tgt_S[i, :L] = residue_embedding

    mol_batch = mol2graph(mols)
    batched_binder = (bind_X, mol_batch, bind_A)
    batched_target = (tgt_X, tgt_S, tgt_A)

    if use_affinity_label:
        affinity = torch.tensor([entry['affinity'] for entry in batch])
    else:
        affinity = None
    return batched_binder, batched_target, affinity


class ToxBenchDataset(Dataset):
    def __init__(
            self,
            csv_file: str,
            aa_size: int,
            max_residue_atoms: int = 14,
            patch_size: int = 50,
            prefix_path: str = None,
            split: str="train",
            permissive: bool = True
        ):
        """
        Initializes the ToxBench dataset.

        Args:
            csv_file (str): Path to the CSV file containing data sample infomation, including data paths, affinity labels, and splitting.
            aa_size (int): Number of residue types.
            max_residue_atoms (int): Maximum number of atoms per residue. Defaults to 14.
            patch_size (int): Number of residues to be considered as in pocket. Defaults to 50.
            prefix_path (str): Prefix path of the data samples.
            split (str): the name of the split ('train', 'val', or 'test').
            permissive (bool): If True, skips invalid samples and moves to the next. Does not work with DDP. Defaults to True.
        """
        # Load the CSV file containing paths
        self.data_frame = pd.read_csv(csv_file)
        self.aa_size = aa_size
        self.prefix_path = prefix_path
        self.max_residue_atoms = max_residue_atoms
        self.patch_size = patch_size

        # Validate the split name
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split name '{self.split}'. Expected 'train', 'val', or 'test'.")

        self.permissive = permissive
        self._get_split()

    def _get_split(self):
        if self.split == "train":
            self.data_frame = self.data_frame[self.data_frame['is_train']]
        elif self.split == "val":
            self.data_frame = self.data_frame[self.data_frame['is_valid']]
        elif self.split == "test":
            self.data_frame = self.data_frame[self.data_frame['is_test']]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        try:
            entry = {} # The data entry that will be returned

            pdb_id = self.data_frame.iloc[idx]['pdb_id']
            raw_protein_path = self.data_frame.iloc[idx]['protein_path']
            raw_ligand_path = self.data_frame.iloc[idx]['ligand_path']
            # Fetch the paths for the protein and ligand files
            protein_pdb = os.path.join(self.prefix_path, raw_protein_path)
            ligand_sdf = os.path.join(self.prefix_path, raw_ligand_path)

            # Load pdb file and strip out water molecules
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb_id, protein_pdb)
            structure = strip_water_molecule_from_structure(structure)
            if len(structure) > 1:
                raise ValueError(f'Expected 1 model in {protein_pdb}, but got {len(structure)}')
            model = structure[0]
            sequence = []
            coordinates = []
            for residue in model.get_residues():
                resname = PROTEIN_LETTERS_3TO1.get(residue.resname, '#')
                res_index = ALPHABET.index(resname) if resname in ALPHABET else 0
                expected_atoms = RES_ATOM14[res_index]
                sequence.append(resname)
                res_coords = get_residue_atoms(residue, expected_atoms)
                coordinates.append(res_coords)

            # prepare data entry
            entry = {
                'target_seq': ''.join(sequence),
                'target_coords': np.array(coordinates),  # Shape will be (len(target_seq), 14, 3)
            }

            # Load the ligand
            supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=True)
            entry['binder_mol'] = supplier[0]

            affinity_value = float(self.data_frame.iloc[idx]['abfep_affinity'])
            # Qualify labels for non-binders
            if affinity_value > -3.0:
                affinity_value = -3.0  # Change it to -3
            entry['affinity'] = torch.tensor(affinity_value)
            entry = preprocess_entry(entry, self.patch_size)
            return entry
        except Exception as e:
            if self.permissive and idx+1 < len(self.data_frame):
                logging.warning(e)
                return self.__getitem__(idx+1) # jump to the next sample
            elif self.permissive and idx+1 == len(self.data_frame):
                logging.warning(e)
                return self.__getitem__(0) # jump to the first sample for this corner case
            else:
                raise e


    def pl_collate_fn(self, batch: List[Dict]):
        """
        Custom collate function for handling batches of protein-ligand data for DualBind model.

        Args:
            batch (List[Dict]): a list of data (Dict) as constructed by the __init__ function.

        Returns:
            batched_binder (Tuple[torch.Tensor, BatchMolGraph, torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second list is a batched molecular graph. The third tensor is a mask for indicating ligand atoms.
            batched_target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket.
            affinity (Optional[torch.Tensor]): the batched affinity values.
        """
        return pl_collate_fn(
            batch=batch,
            max_residue_atoms=self.max_residue_atoms,
            aa_size=self.aa_size,
            use_affinity_label=True,
        )

# This is the dataset class for inference on any protein-ligand dataset
class InferenceDataset(Dataset):
    def __init__(
            self,
            protein_files: List[str],
            ligand_files: List[str],
            aa_size: int = 21,
            max_residue_atoms: int = 14,
            patch_size: int = 50,
            min_heavy_atoms: int = 3,
            permissive: bool = True,
        ):
        """
        A dataset module for inference.

        Args:
            protein_files (List[str]): List of Path to the protein PDB files.
            ligand_files  (List[str]): List of Path to the ligand SDF files.
            aa_size (int): Number of residue types.
            max_residue_atoms (int): Maximum number of atoms per residue. Defaults to 14.
            patch_size (int): Number of residues to be considered as in pocket. Defaults to 50.
            system_dir (str): Directory containing the system files.
            min_heavy_atoms (int): Minimum number of heavy atoms required in a ligand. Defaults to 3.
            permissive (bool): If True, skips invalid samples and moves to the next. Does not work with DDP. Defaults to True.
        """
        # Load the CSV file containing paths
        self.protein_files = protein_files
        self.ligand_files = ligand_files
        self.aa_size = aa_size
        self.max_residue_atoms = max_residue_atoms
        self.min_heavy_atoms = min_heavy_atoms
        self.patch_size = patch_size
        self.permissive = permissive
        assert len(self.protein_files) == len(self.ligand_files)

    def __len__(self):
        return len(self.protein_files)

    def __getitem__(self, idx: int):
        try:
            # Fetch the paths for the protein and ligand files
            protein_pdb = self.protein_files[idx]
            ligand_sdf = self.ligand_files[idx]

            # Load pdb file and strip out water molecules
            parser = PDB.PDBParser()
            structure = parser.get_structure('random_id', protein_pdb)
            structure = strip_water_molecule_from_structure(structure)

            if len(structure) > 1:
                raise ValueError(f'Expected 1 model in {protein_pdb}, but got {len(structure)}')
            model = structure[0]

            # Collecting residue sequence and all atom coordinates
            sequence = []
            coordinates = []
            for residue in model.get_residues():
                resname = PROTEIN_LETTERS_3TO1.get(residue.resname, '#')
                res_index = ALPHABET.index(resname) if resname in ALPHABET else 0
                expected_atoms = RES_ATOM14[res_index]
                sequence.append(resname)
                res_coords = get_residue_atoms(residue, expected_atoms)
                coordinates.append(res_coords)

            # prepare data entry
            entry = {
                'target_seq': ''.join(sequence),
                'target_coords': np.array(coordinates),  # Shape will be (len(target_seq), 14, 3)
            }
            
            # Load and process the ligand
            df = PandasTools.LoadSDF(ligand_sdf, molColName='Molecule', includeFingerprints=False)
            if len(df['Molecule']) == 1:
                ValueError('More than one ligand in the sdf file')

            binder_mol = df['Molecule'][0]
            if binder_mol.GetNumHeavyAtoms() < self.min_heavy_atoms:
                raise ValueError(f'Ligand has less than {self.min_heavy_atoms} heavy atoms')
        except Exception as e:
            if self.permissive and idx+1 < len(self.system_ids):
                logging.warning(e)
                return self.__getitem__(idx+1) # jump to the next sample
            raise e
        
        entry['binder_mol'] = binder_mol
        entry['binder_coords'] = mol2coords(entry['binder_mol'])
        entry = preprocess_entry(entry, self.patch_size)
        return entry

    def pl_collate_fn(self, batch: List[Dict]):
        """
        Custom collate function for handling batches of protein-ligand data for DSMBind model.

        Args:
            batch (List[Dict]): a list of data (Dict) as constructed by the __init__ function.

        Returns:
            batched_binder (Tuple[torch.Tensor, List[Mol], torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second list is a list of RDKit molecules. The third tensor is a mask for indicating ligand atoms.
            batched_target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket.
        """
        return pl_collate_fn(
            batch=batch,
            max_residue_atoms=self.max_residue_atoms,
            aa_size=self.aa_size,
            use_affinity_label=False,
        )
