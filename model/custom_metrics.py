from __future__ import absolute_import, division, print_function
from builtins import range
import os
import numpy as np
import pickle
import gzip
import math
import random
import pymatgen as mg
import rdkit
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, Descriptors
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from copy import deepcopy
from math import exp, log
from collections import OrderedDict
from nn_metrics import KerasNN
from gp_metrics import GaussianProcess
from mol_methods import *

"""
Metrics
"""


def batch_validity(smiles, train_smiles=None):
    """
    Assigns 1.0 if the SMILES is correct, and 0.0
    if not.
    """
    vals = [1.0 if verify_sequence(s) else 0.0 for s in smiles]
    return vals

def batch_diversity(smiles, train_smiles):
    """
    Compares the Tanimoto distance of a given molecule
    with a random sample of the training smiles.
    """
    rand_smiles = random.sample(train_smiles, 100)
    rand_mols = [MolFromSmiles(s) for s in rand_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in rand_mols]
    vals = [apply_to_valid(s, diversity, fps=fps) for s in smiles]
    return vals

def batch_variety(smiles, train_smiles=None):
    """
    Compares the Tanimoto distance of a given molecule
    with a random sample of the other generated smiles.
    """
    filtered = filter_smiles(smiles)
    if filtered:
        mols = [Chem.MolFromSmiles(smile) for smile in
                np.random.choice(filtered, int(len(filtered) / 10))]
        setfps = [Chem.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048) for mol in mols]
        vals = [apply_to_valid(s, variety, setfps=setfps) for s in smiles]
        return vals
    else:
        return np.zeros(len(smiles))

def batch_novelty(smiles, train_smiles):
    """
    Assigns 1.0 if the molecule is not in the training
    set, and 1.0 otherwise.
    """
    vals = [novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

def batch_hardnovelty(smiles, train_smiles):
    """
    Assigns 1.0 if the molecule's canonical SMILES is
    notin the training set, and 1.0 otherwise.
    """
    vals = [hard_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

def batch_softnovelty(smiles, train_smiles):
    """
    Assigns 1.0 if the molecule is not in the training
    set, and 0.3 otherwise.
    """
    vals = [soft_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

def batch_creativity(smiles, train_smiles):
    """
    Computes the Tanimoto distance of a smile
    to the training set, as a measure of how different
    these molecules are from the provided ones.
    """
    mols = [Chem.MolFromSmiles(smile) for smile in filter_smiles(train_smiles)]
    setfps = [Chem.GetMorganFingerprintAsBitVect(
        mol, 4, nBits=2048) for mol in mols]
    vals = [apply_to_valid(s, creativity, setfps=setfps) for s in smiles]
    return vals

def batch_symmetry(smiles, train_smiles=None):
    """
    Yields 1.0 if the generated molecule has 
    any element of symmetry, and 0.0 if the point group is C1.
    """
    vals = [apply_to_valid(s, symmetry) for s in smiles]
    return vals

def batch_logP(smiles, train_smiles=None):
    """
    This metric computes the logarithm of the water-octanol partition
    coefficient, using RDkit's implementation of Wildman-Crippen method,
    and then remaps it to the 0.0-1.0 range.

    Wildman, S. A., & Crippen, G. M. (1999). 
    Prediction of physicochemical parameters by atomic contributions. 
    Journal of chemical information and computer sciences, 39(5), 868-873.
    """
    vals = [apply_to_valid(s, logP) for s in smiles]
    return vals

def batch_conciseness(smiles, train_smiles=None):
    """
    This metric penalizes SMILES strings that are too long, 
    assuming that the canonical representation is the shortest.
    """
    vals = [conciseness(s) if verify_sequence(s) else 0 for s in smiles]
    return vals

def batch_lipinski(smiles, train_smiles):
    """
    This metric assigns 0.25 for every rule of Lipinski's
    rule-of-five that is obeyed.
    """
    vals = [apply_to_valid(s, Lipinski) for s in smiles]
    return vals

def batch_SA(smiles, train_smiles=None, SA_model=None):
    """
    This metric checks whether a given molecule is easy to synthesize or not.
    It is based on (although not completely equivalent to) the work of Ertl
    and Schuffenhauer.

    Ertl, P., & Schuffenhauer, A. (2009).
    Estimation of synthetic accessibility score of drug-like molecules
    based on molecular complexity and fragment contributions.
    Journal of cheminformatics, 1(1), 8.
    """
    vals = [apply_to_valid(s, SA_score, SA_model=SA_model) for s in smiles]
    return vals

def batch_NPLikeliness(smiles, train_smiles=None, NP_model=None):
    """
    This metric computes the likelihood that a given molecule is
    a natural product.
    """
    vals = [apply_to_valid(s, NP_score, NP_model=NP_model) for s in smiles]
    return vals

def batch_beauty(smiles, train_smiles=None):
    """
    Computes chemical beauty.

    Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). 
    Quantifying the chemical beauty of drugs. 
    Nature chemistry, 4(2), 90-98.
    """
    vals = [apply_to_valid(s, chemical_beauty) for s in smiles]
    return vals

def batch_substructure_match_all(smiles, train_smiles=None, ALL_POS_PATTS=None):
    """
    Assigns 1.0 if all the specified substructures are present
    in the molecule.
    """
    if ALL_POS_PATTS == None:
        raise ValueError('No substructures has been specified')

    vals = [apply_to_valid(s, substructure_match_all, ALL_POS_PATTS=ALL_POS_PATTS) for s in smiles]
    return vals

def batch_substructure_match_any(smiles, train_smiles=None, ANY_POS_PATTS=None):
    """
    Assigns 1.0 if any of the specified substructures are present
    in the molecule.
    """
    if ANY_POS_PATTS == None:
        raise ValueError('No substructures has been specified')

    vals = [apply_to_valid(s, substructure_match_any, ANY_POS_PATTS=ANY_POS_PATTS) for s in smiles]
    return vals

def batch_substructure_absence(smiles, train_smiles=None, ALL_NEG_PATTS=None):
    """
    Assigns 0.0 if any of the substructures are present in the
    molecule, and 1.0 otherwise.
    """
    if ALL_NEG_PATTS == None:
        raise ValueError('No substructures has been specified')

    vals = [apply_to_valid(s, substructure_match_any, ALL_NEG_PATTS=ALL_NEG_PATTS) for s in smiles]
    return vals

def batch_PCE(smiles, train_smiles=None, cnn=None):
    """
    Power conversion efficiency as computed by a neural network
    acting on Morgan fingerprints.
    """
    if cnn == None:
        raise ValueError('The PCE metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(cnn.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_bandgap(smiles, train_smiles=None, cnn=None):
    """
    HOMO-LUMO energy difference as computed by a neural network
    acting on Morgan fingerprints.
    """
    if cnn == None:
        raise ValueError('The bandgap metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(cnn.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_mp(smiles, train_smiles=None, cnn=None):
    """
    Melting point as computed by a neural network acting on 
    Morgan fingerprints.
    """
    if cnn == None:
        raise ValueError('The melting point metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(cnn.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_bp(smiles, train_smiles=None, gp=None):
    """
    Boiling point as computed by a gaussian process acting
    on Morgan fingerprints.
    """
    if gp == None:
        raise ValueError('The boiling point metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(gp.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_density(smiles, train_smiles=None, gp=None):
    """
    Density as computed by a gaussian process acting on
    Morgan fingerprints.
    """
    if gp == None:
        raise ValueError('The density metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(gp.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_mutagenicity(smiles, train_smiles=None, gp=None):
    """
    Mutagenicity as estimated by a gaussian process acting on
    Morgan fingerprints.
    """
    if gp == None:
        raise ValueError('The mutagenicity was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(gp.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_pvap(smiles, train_smiles=None, gp=None):
    """
    Vapour pressure as computed by a gaussian process acting on
    Morgan fingerprints.
    """
    if gp == None:
        raise ValueError('The vapour pressure was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(gp.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_solubility(smiles, train_smiles=None, gp=None):
    """
    Solubility in water as computed by a gaussian process acting on
    Morgan fingerprints.
    """
    if gp == None:
        raise ValueError('The solubility was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(gp.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_viscosity(smiles, train_smiles=None, gp=None):
    """
    Viscosity as computed by a gaussian process acting on
    Morgan fingerprints.
    """
    if gp == None:
        raise ValueError('The viscosity was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(gp.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

"""
Loadings
"""


def load_NP(filename=None):
    """
    Loads the parameters required by the naturalness
    metric.
    """
    if filename is None:
        filename = os.path.join(MOD_PATH, '../data/pkl/NP_score.pkl.gz')
    NP_model = pickle.load(gzip.open(filename))
    return ('NP_model', NP_model)

def load_SA(filename=None):
    """
    Loads the parameters required by the synthesizability
    metric.
    """
    if filename is None:
        filename = os.path.join(MOD_PATH, '../data/pkl/SA_score.pkl.gz')
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return ('SA_model', SA_model)

def load_beauty(filename=None):
    """
    Loads the parameters required by the chemical
    beauty metric.
    """
    if filename is None:
        filename = os.path.join(MOD_PATH, '../data/pkl/QED_score.pkl.gz')
    QED = pickle.load(gzip.open(filename))
    global AliphaticRings, Acceptors, StructuralAlerts, pads1, pads2
    AliphaticRings = QED[0]
    Acceptors = QED[1]
    StructuralAlerts = QED[2]
    pads1 = QED[3]
    pads2 = QED[4]
    return ('QED_model', QED_model)

def load_substructure_match_any():
    """
    Loads substructures for the 'MATCH ANY' metric.
    """
    ANY_POS_PATTS = readSubstructuresFile('any_positive.smi', 'any_positive')
    return ('ANY_POS_PATTS', ANY_POS_PATTS)

def load_substructure_match_all():
    """
    Loads substructures for the 'MATCH ALL' metric.
    """
    ALL_POS_PATTS = readSubstructuresFile('all_positive.smi', 'all_positive')
    return ('ALL_POS_PATTS', ALL_POS_PATTS)

def load_substructure_absence():
    """
    Loads substructures for the 'ABSENCE' metric.
    """
    ALL_NEG_PATTS = readSubstructuresFile('all_negative.smi', 'all_negative')
    return ('ALL_NEG_PATTS', ALL_NEG_PATTS)

def load_PCE():
    """
    Loads the Keras NN model for Power Conversion
    Efficiency.
    """
    cnn_pce = KerasNN('pce')
    cnn_pce.load('../data/nns/pce.h5')
    return ('cnn', cnn_pce)

def load_bandgap():
    """
    Loads the Keras NN model for HOMO-LUMO energy
    difference.
    """
    cnn_bandgap = KerasNN('bandgap')
    cnn_bandgap.load('../data/nns/bandgap.h5')
    return ('cnn', cnn_bandgap)

def load_mp():
    """
    Loads the Keras NN model for melting point.
    """
    cnn_mp = KerasNN('mp')
    cnn_mp.load('../data/nns/mp.h5')
    return ('cnn', cnn_mp)

def load_bp():
    """
    Loads the GPmol GP model for boiling point.
    """
    gp_bp = GaussianProcess('bp')
    gp_bp.load('../data/gps/bp.json')
    return ('gp', gp_bp)

def load_density():
    """
    Loads the GPmol GP model for density.
    """
    gp_density = GaussianProcess('density')
    gp_density.load('../data/gps/density.json')
    return ('gp', gp_density)

def load_mutagenicity():
    """
    Loads the GPmol GP model for mutagenicity.
    """
    gp_mutagenicity = GaussianProcess('mutagenicity')
    gp_mutagenicity.load('../data/gps/mutagenicity.json')
    return ('gp', gp_mutagenicity)

def load_pvap():
    """
    Loads the GPmol GP model for vapour pressure.
    """
    gp_pvap = GaussianProcess('pvap')
    gp_pvap.load('../data/gps/pvap.json')
    return ('gp', gp_pvap)

def load_solubility():
    """
    Loads the GPmol GP model for solubility.
    """
    gp_solubility = GaussianProcess('solubility')
    gp_solubility.load('../data/gps/solubility.json')
    return ('gp', gp_solubility)

def load_viscosity():
    """
    Loads the GPmol GP model for viscosity.
    """
    gp_viscosity = GaussianProcess('viscosity')
    gp_viscosity.load('../data/gps/viscosity.json')
    return ('gp', gp_viscosity)

"""
Metrics functions
"""
def diversity(mol, fps):
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    ref_fps = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(
        ref_fps, fps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    val = remap(mean_dist, low_rand_dst, mean_div_dst)
    val = np.clip(val, 0.0, 1.0)
    return val

def variety(mol, setfps):
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    fp = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(fp, setfps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    if NORMALIZE:
        val = remap(mean_dist, low_rand_dst, mean_div_dst)
        val = np.clip(val, 0.0, 1.0)
        return val
    else:
        return mean_dist

def novelty(smile, train_smiles):
    return 1.0 if smile not in train_smiles else 0.0

def hard_novelty(smile, train_smiles):
    return 1.0 if canon_smile(smile) not in train_smiles else 0.0

def soft_novelty(smile, train_smiles):
    return 1.0 if smile not in train_smiles else 0.3

def creativity(mol, setfps):
    return np.mean(DataStructs.BulkTanimotoSimilarity(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048), setfps))

def symmetry(mol):
    try:
        ids, xyz = get3DCoords(mol)
        sch_symbol = getSymmetry(ids, xyz)
        return 1.0 if sch_symbol != 'C1' else 0.0
    except:
        return 0.0


def get3DCoords(mol):
    m = Chem.AddHs(mol)
    m.UpdatePropertyCache(strict=False)
    Chem.EmbedMolecule(m)
    Chem.MMFFOptimizeMolecule(m)
    molblock = Chem.MolToMolBlock(m)
    mblines = molblock.split('\n')[4:len(m.GetAtoms())]
    parsed = [entry.split() for entry in mblines]
    coords = [[coord[3], np.asarray([float(coord[0]), float(
        coord[1]), float(coord[2])])] for coord in parsed]
    ids = [coord[0] for coord in coords]
    xyz = [[coord[1][0], coord[1][1], coord[1][2]] for coord in coords]
    return ids, xyz


def getSymmetry(ids, xyz):
    mol = PointGroupAnalyzer(mg.Molecule(ids, xyz))
    return mol.sch_symbol


def logP(mol, train_smiles=None):
    val = Crippen.MolLogP(mol)
    if NORMALIZE:
        low_logp = -2.12178879609
        high_logp = 6.0429063424
        val = remap(val, low_logp, high_logp)
        val = np.clip(val, 0.0, 1.0)
    return val

def SA_score(mol, SA_model):

    if SA_model is None:
        raise ValueError("Synthesizability metric was not properly loaded.")
    # fragment score
    fp = Chem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    val = remap(sascore, 5, 1.5)
    val = np.clip(val, 0.0, 1.0)
    return val


def conciseness(smile, train_smiles=None):
    canon = canon_smile(smile)
    diff_len = len(smile) - len(canon)
    val = np.clip(diff_len, 0.0, 20)
    val = 1 - 1.0 / 20.0 * val
    return val

def NP_score(mol, NP_model=None):
    fp = Chem.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()

    # calculating the score
    val = 0.
    for bit in bits:
        val += NP_model.get(bit, 0)
    val /= float(mol.GetNumAtoms())

    # preventing score explosion for exotic molecules
    if val > 4:
        val = 4. + math.log10(val - 4. + 1.)
    if val < -4:
        val = -4. - math.log10(-4. - val + 1.)

    if NORMALIZE:
        val = np.clip(remap(val, -3, 1), 0.0, 1.0)
    return val

def Lipinski(mol):
    druglikeness = 0.0
    druglikeness += 0.25 if logP(mol) <= 5 else 0.0
    druglikeness += 0.25 if rdkit.Chem.Descriptors.MolWt(mol) <= 500 else 0.0
    # Look for hydrogen bond aceptors
    acceptors = 0
    for atom in mol.GetAtoms():
        acceptors += 1 if atom.GetAtomicNum() == 8 else 0.0
        acceptors += 1 if atom.GetAtomicNum() == 7 else 0.0
    druglikeness += 0.25 if acceptors <= 10 else 0.0
    # Look for hydrogen bond donors
    donors = 0
    for bond in mol.GetBonds():
        a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
        a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
        donors += 1 if ((a1, a2) == (1, 8)) or ((a1, a2) == (8, 1)) else 0.0
        donors += 1 if ((a1, a2) == (1, 7)) or ((a1, a2) == (7, 1)) else 0.0
    druglikeness += 0.25 if donors <= 5 else 0.0
    return druglikeness

def substructure_match_all(mol, train_smiles=None):
    val = all([mol.HasSubstructMatch(patt) for patt in ALL_POS_PATTS])
    return int(val)

def substructure_match_any(mol, train_smiles=None):
    val = any([mol.HasSubstructMatch(patt) for patt in ANY_POS_PATTS])
    return int(val)

def substructure_absence(mol, train_smiles=None):
    val = all([not mol.HasSubstructMatch(patt) for patt in ANY_NEG_PATTS])
    return int(val)

def readSubstructuresFile(filename, label='positive'):
    if os.path.exists(filename):
        smiles = read_smi(filename)
        patterns = [Chem.MolFromSmarts(s) for s in smiles]
    else:
        patterns = None
    return patterns

def ads(x, a, b, c, d, e, f, dmax):
    return ((a + (b / (1 + exp(-1 * (x - c + d / 2) / e)) * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f))))) / dmax)


def properties(mol):
    matches = []
    if (mol is None):
        raise WrongArgument("properties(mol)", "mol argument is \'None\'")
    x = [0] * 8
    # MW
    x[0] = Descriptors.MolWt(mol)
    # ALOGP
    x[1] = Descriptors.MolLogP(mol)
    for hba in Acceptors:                                                       # HBA
        if (mol.HasSubstructMatch(hba)):
            matches = mol.GetSubstructMatches(hba)
            x[2] += len(matches)
    x[3] = Descriptors.NumHDonors(
        mol)                                          # HBD
    # PSA
    x[4] = Descriptors.TPSA(mol)
    x[5] = Descriptors.NumRotatableBonds(
        mol)                                   # ROTB
    x[6] = Chem.GetSSSR(Chem.DeleteSubstructs(
        deepcopy(mol), AliphaticRings))   # AROM
    for alert in StructuralAlerts:                                              # ALERTS
        if (mol.HasSubstructMatch(alert)):
            x[7] += 1
    return x


def qed(w, p, gerebtzoff):
    d = [0.00] * 8
    if (gerebtzoff):
        for i in range(0, 8):
            d[i] = ads(p[i], pads1[i][0], pads1[i][1], pads1[i][2], pads1[
                       i][3], pads1[i][4], pads1[i][5], pads1[i][6])
    else:
        for i in range(0, 8):
            d[i] = ads(p[i], pads2[i][0], pads2[i][1], pads2[i][2], pads2[
                       i][3], pads2[i][4], pads2[i][5], pads2[i][6])
    t = 0.0
    for i in range(0, 8):
        t += w[i] * log(d[i])
    return (exp(t / sum(w)))


def weights_mean(mol, gerebtzoff=True):
    props = properties(mol)
    return qed([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], props, gerebtzoff)


def chemical_beauty(mol, gerebtzoff=True):
    return weights_mean(mol, gerebtzoff)

"""
Rewards loading
"""

def metrics_loading():
    """
    Feeds the loading procedures to the main program
    """
    global MOD_PATH
    MOD_PATH = os.path.dirname(os.path.realpath(__file__))

    load = OrderedDict()
    # Cheminformatics
    load['validity'] = lambda *args: None
    load['diversity'] = lambda *args: None
    load['variety'] = lambda *args: None
    load['novelty'] = lambda *args: None
    load['hard_novelty'] = lambda *args: None
    load['soft_novelty'] = lambda *args: None
    load['creativity'] = lambda *args: None
    load['symmetry'] = lambda *args: None
    load['conciseness'] = lambda *args: None
    load['lipinski'] = lambda *args: None
    load['synthesizability'] = load_SA
    load['naturalness'] = load_NP
    load['chemical_beauty'] = load_beauty
    load['substructure_match_all'] = load_substructure_match_all
    load['substructure_match_any'] = load_substructure_match_any
    load['substructure_absence'] = load_substructure_absence
    load['mutagenicity'] = load_mutagenicity

    # Physical properties
    load['logP'] = lambda *args: None
    load['pce'] = load_PCE
    load['bandgap'] = load_bandgap
    load['mp'] = load_mp
    load['bp'] = load_bp
    load['density'] = load_density
    load['pvap'] = load_pvap
    load['solubility'] = load_solubility
    load['viscosity'] = load_viscosity

    return load

def get_metrics():
    """
    Feeds the metrics to the main program
    """
    metrics = OrderedDict()

    # Cheminformatics
    metrics['validity'] = batch_validity
    metrics['novelty'] = batch_novelty
    metrics['creativity'] = batch_creativity
    metrics['hard_novelty'] = batch_hardnovelty
    metrics['soft_novelty'] = batch_softnovelty
    metrics['diversity'] = batch_diversity
    metrics['variety'] = batch_variety
    metrics['symmetry'] = batch_symmetry
    metrics['conciseness'] = batch_conciseness
    metrics['logP'] = batch_logP
    metrics['lipinski'] = batch_lipinski
    metrics['synthesizability'] = batch_SA
    metrics['naturalness'] = batch_NPLikeliness
    metrics['chemical_beauty'] = batch_beauty
    metrics['substructure_match_all'] = batch_substructure_match_all
    metrics['substructure_match_any'] = batch_substructure_match_any
    metrics['substructure_absence'] = batch_substructure_absence
    metrics['mutagenicity'] = batch_mutagenicity

    # Physical properties
    metrics['pce'] = batch_PCE
    metrics['bandgap'] = batch_bandgap
    metrics['mp'] = batch_mp
    metrics['bp'] = batch_bp
    metrics['density'] = batch_density
    metrics['pvap'] = batch_pvap
    metrics['solubility'] = batch_solubility
    metrics['viscosity'] = batch_viscosity

    return metrics
