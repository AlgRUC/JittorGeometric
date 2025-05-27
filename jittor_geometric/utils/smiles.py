from typing import Any, Dict, List
import jittor as jt
import jittor_geometric
import os, platform
import warnings


x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def from_rdmol(mol: Any) -> 'jittor_geometric.data.Data':
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`jittor_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """
    if platform.system() == 'Linux':
        os.RTLD_GLOBAL = os.RTLD_GLOBAL | os.RTLD_DEEPBIND
        import jittor_utils
        with jittor_utils.import_scope(os.RTLD_GLOBAL | os.RTLD_NOW):
            from rdkit import Chem
    from jittor_geometric.data import Data
    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = jt.array(xs, dtype='int32').view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = jt.array(edge_indices, dtype='int32')
    if len(edge_index) != 0:
        edge_index = edge_index.t().view(2, -1)
    else:
        edge_index = edge_index.view(2, -1)
    edge_attr = jt.array(edge_attrs, dtype='int32').view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()[0]
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> 'jittor_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`jittor_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    if platform.system() == 'Linux':
        os.RTLD_GLOBAL = os.RTLD_GLOBAL | os.RTLD_DEEPBIND
        import jittor_utils
        with jittor_utils.import_scope(os.RTLD_GLOBAL | os.RTLD_NOW):
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = from_rdmol(mol)
    data.smiles = smiles
    return data


def to_rdmol(
    data: 'jittor_geometric.data.Data',
    kekulize: bool = False,
) -> Any:
    """Converts a :class:`jittor_geometric.data.Data` instance to a
    :class:`rdkit.Chem.Mol` instance.

    Args:
        data (jittor_geometric.data.Data): The molecular graph data.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    if platform.system() == 'Linux':
        os.RTLD_GLOBAL = os.RTLD_GLOBAL | os.RTLD_DEEPBIND
        import jittor_utils
        with jittor_utils.import_scope(os.RTLD_GLOBAL | os.RTLD_NOW):
            from rdkit import Chem

    mol = Chem.RWMol()

    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(data.x[i, 1])])
        atom.SetFormalCharge(x_map['formal_charge'][int(data.x[i, 3])])
        atom.SetNumExplicitHs(x_map['num_hs'][int(data.x[i, 4])])
        atom.SetNumRadicalElectrons(x_map['num_radical_electrons'][int(
            data.x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(
            data.x[i, 6])])
        atom.SetIsAromatic(bool(data.x[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0])]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[int(data.edge_attr[i, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return mol


def to_smiles(
    data: 'jittor_geometric.data.Data',
    kekulize: bool = False,
) -> str:
    """Converts a :class:`jittor_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (jittor_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    if platform.system() == 'Linux':
        os.RTLD_GLOBAL = os.RTLD_GLOBAL | os.RTLD_DEEPBIND
        import jittor_utils
        with jittor_utils.import_scope(os.RTLD_GLOBAL | os.RTLD_NOW):
            from rdkit import Chem
    mol = to_rdmol(data, kekulize=kekulize)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
