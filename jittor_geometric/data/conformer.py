# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import warnings
from scipy.spatial import distance_matrix
from .dictionary import Dictionary
from multiprocessing import Pool
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)

class ConformerGen(object):
    '''
    This class designed to generate conformers for molecules represented as SMILES strings using provided parameters and configurations. The `transform` method uses multiprocessing to speed up the conformer generation process.
    '''
    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the ConformerGen object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        self.dictionary = Dictionary.load('default')
        self.dictionary.add_symbol("[MASK]", is_special=True)

    def single_process(self, smiles):
        """
        Processes a single SMILES string to generate conformers using the specified method.

        :param smiles: (str) The SMILES string representing the molecule.
        :return: A unimolecular data representation (dictionary) of the molecule.
        :raises ValueError: If the conformer generation method is unrecognized.
        """
        if self.method == 'rdkit_random':
            atoms, coordinates = inner_smi2coords(smiles, seed=self.seed, mode=self.mode, remove_hs=self.remove_hs)
            return coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms, remove_hs=self.remove_hs)
        else:
            raise ValueError('Unknown conformer generation method: {}'.format(self.method))
        
    def transform_raw(self, atoms_list, coordinates_list):

        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform(self, smiles_list):
        pool = Pool()
        logger.info('Start generating conformers...')
        inputs = [item for item in tqdm(pool.imap(self.single_process, smiles_list))]
        pool.close()
        failed_cnt = np.mean([(item['src_coord']==0.0).all() for item in inputs])
        logger.info('Succeed to generate conformers for {:.2f}% of molecules.'.format((1-failed_cnt)*100))
        failed_3d_cnt = np.mean([(item['src_coord'][:,2]==0.0).all() for item in inputs])
        logger.info('Succeed to generate 3d conformers for {:.2f}% of molecules.'.format((1-failed_3d_cnt)*100))
        return inputs


def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
    pass
#     '''
#     This function is responsible for converting a SMILES (Simplified Molecular Input Line Entry System) string into 3D coordinates for each atom in the molecule. It also allows for the generation of 2D coordinates if 3D conformation generation fails, and optionally removes hydrogen atoms and their coordinates from the resulting data.

#     :param smi: (str) The SMILES representation of the molecule.
#     :param seed: (int, optional) The random seed for conformation generation. Defaults to 42.
#     :param mode: (str, optional) The mode of conformation generation, 'fast' for quick generation, 'heavy' for more attempts. Defaults to 'fast'.
#     :param remove_hs: (bool, optional) Whether to remove hydrogen atoms from the final coordinates. Defaults to True.

#     :return: A tuple containing the list of atom symbols and their corresponding 3D coordinates.
#     :raises AssertionError: If no atoms are present in the molecule or if the coordinates do not align with the atom count.
#     '''
#     mol = Chem.MolFromSmiles(smi)
#     mol = AllChem.AddHs(mol)
#     atoms = []
#     for atom in mol.GetAtoms():
#         atoms.append(atom.GetSymbol())
#     assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
#     try:
#         # will random generate conformer with seed equal to -1. else fixed random seed.
#         res = AllChem.EmbedMolecule(mol, randomSeed=seed)
#         if res == 0:
#             try:
#                 # some conformer can not use MMFF optimize
#                 AllChem.MMFFOptimizeMolecule(mol)
#                 coordinates = mol.GetConformer().GetPositions().astype(np.float32)
#             except:
#                 coordinates = mol.GetConformer().GetPositions().astype(np.float32)
#         ## for fast test... ignore this ###
#         elif res == -1 and mode == 'heavy':
#             AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
#             try:
#                 # some conformer can not use MMFF optimize
#                 AllChem.MMFFOptimizeMolecule(mol)
#                 coordinates = mol.GetConformer().GetPositions().astype(np.float32)
#             except:
#                 AllChem.Compute2DCoords(mol)
#                 coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
#                 coordinates = coordinates_2d
#         else:
#             AllChem.Compute2DCoords(mol)
#             coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
#             coordinates = coordinates_2d
#     except:
#         logger.info("Failed to generate conformer, replace with zeros.")
#         coordinates = np.zeros((len(atoms),3))
#     assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
#     if remove_hs:
#         idx = [i for i, atom in enumerate(atoms) if atom != 'H']
#         atoms_no_h = [atom for atom in atoms if atom != 'H']
#         coordinates_no_h = coordinates[idx]
#         assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
#         return atoms_no_h, coordinates_no_h
#     else:
#         return atoms, coordinates

def inner_coords(atoms, coordinates, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates. 

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.
    
    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.
    
    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }
