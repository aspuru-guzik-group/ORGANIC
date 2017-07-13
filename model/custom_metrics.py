from __future__ import absolute_import, division, print_function
from builtins import range
import os
import numpy as np
import time
import pickle
import gzip
import math
import random
import pymatgen as mg
import rdkit
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, Descriptors
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from copy import deepcopy
from math import exp, log
from collections import OrderedDict
from mol_methods import *


# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

MOD_PATH = os.path.dirname(os.path.realpath(__file__))
print(MOD_PATH)


def readNPModel(filename=None):
    print("mol_metrics: reading NP model ...", end=' ')
    if filename is None:
        filename = os.path.join(MOD_PATH, 'NP_score.pkl.gz')
    start = time.time()
    NP_model = pickle.load(gzip.open(filename))
    end = time.time()
    print("loaded in {}".format(end - start))
    return NP_model


def readSubstructuresFile(filename, label='positive'):
    if os.path.exists(filename):
        smiles = read_smi(filename)
        print("mol_metrics: reading {} substructures from {} ...".format(
            len(smiles), label))
        patterns = [Chem.MolFromSmarts(s) for s in smiles]
    else:
        print('\tno substurctures file found, if using substructure scoring save smiles/smarts in {}smi'.format(label))
        patterns = None
    return patterns


def readSAModel(filename=None):
    print("mol_metrics: reading SA model ...", end=' ')
    if filename is None:
        filename = os.path.join(MOD_PATH, 'SA_score.pkl.gz')
    start = time.time()
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    end = time.time()
    print("loaded in {}".format(end - start))
    return SA_model

SA_model = readSAModel()
NP_model = readNPModel()
ALL_POS_PATTS = readSubstructuresFile('all_positive.smi', 'all_positive')
ANY_POS_PATTS = readSubstructuresFile('any_positive.smi', 'any_positive')
ALL_NEG_PATTS = readSubstructuresFile('all_negative.smi', 'all_negative')

NORMALIZE = False
#__all__ = ['weights_max', 'weights_mean', 'weights_none', 'default']
AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')
AcceptorSmarts = [
    '[oH0;X2]',
    '[OH1;X2;v2]',
    '[OH0;X2;v2]',
    '[OH0;X1;v2]',
    '[O-;X1]',
    '[SH0;X2;v2]',
    '[SH0;X1;v2]',
    '[S-;X1]',
    '[nH0;X2]',
    '[NH0;X1;v3]',
    '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
]
Acceptors = []
for hba in AcceptorSmarts:
    Acceptors.append(Chem.MolFromSmarts(hba))
StructuralAlertSmarts = [
    '*1[O,S,N]*1',
    '[S,C](=[O,S])[F,Br,Cl,I]',
    '[CX4][Cl,Br,I]',
    '[C,c]S(=O)(=O)O[C,c]',
    '[$([CH]),$(CC)]#CC(=O)[C,c]',
    '[$([CH]),$(CC)]#CC(=O)O[C,c]',
    'n[OH]',
    '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]',
    'C=C(C=O)C=O',
    'n1c([F,Cl,Br,I])cccc1',
    '[CH1](=O)',
    '[O,o][O,o]',
    '[C;!R]=[N;!R]',
    '[N!R]=[N!R]',
    '[#6](=O)[#6](=O)',
    '[S,s][S,s]',
    '[N,n][NH2]',
    'C(=O)N[NH2]',
    '[C,c]=S',
    '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
    'C1(=[O,N])C=CC(=[O,N])C=C1',
    'C1(=[O,N])C(=[O,N])C=CC=C1',
    'a21aa3a(aa1aaaa2)aaaa3',
    'a31a(a2a(aa1)aaaa2)aaaa3',
    'a1aa2a3a(a1)A=AA=A3=AA=A2',
    'c1cc([NH2])ccc1',
    '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
    'I',
    'OS(=O)(=O)[O-]',
    '[N+](=O)[O-]',
    'C(=O)N[OH]',
    'C1NC(=O)NC(=O)1',
    '[SH]',
    '[S-]',
    'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
    'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
    '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
    '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
    '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
    '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    'C#C',
    '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
    '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
    '[C,c]=N[OH]',
    '[C,c]=NOC=O',
    '[C,c](=O)[CX4,CR0X3,O][C,c](=O)',
    'c1ccc2c(c1)ccc(=O)o2',
    '[O+,o+,S+,s+]',
    'N=C=O',
    '[NX3,NX4][F,Cl,Br,I]',
    'c1ccccc1OC(=O)[#6]',
    '[CR0]=[CR0][CR0]=[CR0]',
    '[C+,c+,C-,c-]',
    'N=[N+]=[N-]',
    'C12C(NC(N1)=O)CSC2',
    'c1c([OH])c([OH,NH2,NH])ccc1',
    'P',
    '[N,O,S]C#N',
    'C=C=O',
    '[Si][F,Cl,Br,I]',
    '[SX2]O',
    '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
    'O1CCCCC1OC2CCC3CCCCC3C2',
    'N=[CR0][N,n,O,S]',
    '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
    'C=[C!r]C#N',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
    '[OH]c1ccc([OH,NH2,NH])cc1',
    'c1ccccc1OC(=O)O',
    '[SX2H0][N]',
    'c12ccccc1(SC(S)=N2)',
    'c12ccccc1(SC(=S)N2)',
    'c1nnnn1C=O',
    's1c(S)nnc1NC=O',
    'S1C=CSC1=S',
    'C(=O)Onnn',
    'OS(=O)(=O)C(F)(F)F',
    'N#CC[OH]',
    'N#CC(=O)',
    'S(=O)(=O)C#N',
    'N[CH2]C#N',
    'C1(=O)NCC1',
    'S(=O)(=O)[O-,OH]',
    'NC[F,Cl,Br,I]',
    'C=[C!r]O',
    '[NX2+0]=[O+0]',
    '[OR0,NR0][OR0,NR0]',
    'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]',
    '[CX2R0][NX3R0]',
    'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
    '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
    '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
    '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
    '[*]=[N+]=[*]',
    '[SX3](=O)[O-,OH]',
    'N#N',
    'F.F.F.F',
    '[R0;D2][R0;D2][R0;D2][R0;D2]',
    '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
    'C=!@CC=[O,S]',
    '[#6,#8,#16][C,c](=O)O[C,c]',
    'c[C;R0](=[O,S])[C,c]',
    'c[SX2][C;!R]',
    'C=C=C',
    'c1nc([F,Cl,Br,I,S])ncc1',
    'c1ncnc([F,Cl,Br,I,S])c1',
    'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
    '[C,c]S(=O)(=O)c1ccc(cc1)F',
    '[15N]',
    '[13C]',
    '[18O]',
    '[34S]'
]
StructuralAlerts = []
for smarts in StructuralAlertSmarts:
    StructuralAlerts.append(Chem.MolFromSmarts(smarts))

# ADS parameters for the 8 molecular properties: [row][column]
#   rows[8]:    MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
#   columns[7]: A, B, C, D, E, F, DMAX
# ALOGP parameters from Gregory Gerebtzoff (2012, Roche)
pads1 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [0.486849448, 186.2293718, 2.066177165, 3.902720615,
          1.027025453, 0.913012565, 145.4314800],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202,
             0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001,
          0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614,
          12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684,
          1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001,
          1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]
# ALOGP parameters from the original publication
pads2 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [3.172690585, 137.8624751, 2.534937431, 4.581497897,
          0.822739154, 0.576295591, 131.3186604],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202,
             0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001,
          0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614,
          12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684,
          1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001,
          1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]


############################################
#
#   2.MOLECULAR METRICS
#
#       2.1. Validity
#       2.2. Diversity
#       2.3. Variety
#       2.4. Novelty, hardnovelty and softnovelty
#       2.5. Creativity
#       2.6. Symmetry
#       2.7. Solubility
#       2.8. Conciseness
#       2.9. Synthetic accesibility
#       2.10. Druglikeness
#       2.11. Lipinski's rule of five
#       2.12. NP-likeness
#       2.13. PCE
#       2.14. Bandgap
#       2.15. Substructure match
#
#############################################
#
#
#

#
# 2.1. Validity
#

"""
Simplest metric. Assigns 1.0 if the SMILES is correct, and 0.0
if not.
"""


def batch_validity(smiles, train_smiles=None):
    vals = [1.0 if verify_sequence(s) else 0.0 for s in smiles]
    return vals

#
# 2.2. Diversity
#

"""
This metric compares the Tanimoto distance of a given molecule
with a random sample of the training smiles.
"""


def batch_diversity(smiles, train_smiles):
    rand_smiles = random.sample(train_smiles, 100)
    rand_mols = [MolFromSmiles(s) for s in rand_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in rand_mols]
    vals = [apply_to_valid(s, diversity, fps=fps) for s in smiles]
    return vals


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

#
# 2.3. Variety
#

"""
This metric compares the Tanimoto distance of a given molecule
with a random sample of the other generated smiles.
"""


def batch_variety(smiles, train_smiles=None):
    filtered = filter_smiles(smiles)
    mols = [Chem.MolFromSmiles(smile) for smile in np.random.choice(
        filtered, int(len(filtered) / 10))]
    setfps = [Chem.GetMorganFingerprintAsBitVect(
        mol, 4, nBits=2048) for mol in mols]
    vals = [apply_to_valid(s, variety, setfps=setfps) for s in smiles]
    return vals


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

#
# 2.4. Novelty
#

"""
These metrics check whether a given smile is in the provided
training set or not.
"""


def batch_novelty(smiles, train_smiles):
    vals = [novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals


def novelty(smile, train_smiles):
    newness = 1.0 if smile not in train_smiles else 0.0
    return newness


def batch_hardnovelty(smiles, train_smiles):
    vals = [hard_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals


def hard_novelty(smile, train_smiles):
    newness = 1.0 if canon_smile(smile) not in train_smiles else 0.0
    return newness


def soft_novelty(smile, train_smiles):
    newness = 1.0 if smile not in train_smiles else 0.3
    return newness


def batch_softnovelty(smiles, train_smiles):
    vals = [soft_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

#
# 2.5. Creativity
#

"""
This metric computes the Tanimoto distance of a smile to the training set,
as a measure of how different these molecules are from the provided ones.
"""


def batch_creativity(smiles, train_smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in filter_smiles(train_smiles)]
    setfps = [Chem.GetMorganFingerprintAsBitVect(
        mol, 4, nBits=2048) for mol in mols]
    vals = [apply_to_valid(s, creativity, setfps=setfps) for s in smiles]
    return vals


def creativity(mol, setfps):
    return np.mean(DataStructs.BulkTanimotoSimilarity(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048), setfps))

#
# 2.6. Symmetry
#

"""
This metric yields 1.0 if the generated molecule has any element of symmetry, and 0.0
if the point group is C1.
"""


def batch_symmetry(smiles, train_smiles=None):
    vals = [apply_to_valid(s, symmetry) for s in smiles]
    return vals


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

#
# 2.7. Solubility
#

"""
This metric computes the logarithm of the water-octanol partition
coefficient, using RDkit's implementation of Wildman-Crippen method, 
and then remaps it to the 0.0-1.0 range.

Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions. 
Journal of chemical information and computer sciences, 39(5), 868-873.
"""


def batch_solubility(smiles, train_smiles=None):
    vals = [apply_to_valid(s, logP) for s in smiles]
    return vals


def logP(mol, train_smiles=None):
    val = Crippen.MolLogP(mol)
    if NORMALIZE:
        low_logp = -2.12178879609
        high_logp = 6.0429063424
        val = remap(val, low_logp, high_logp)
        val = np.clip(val, 0.0, 1.0)
    return val

#
# 2.8. Conciseness
#

"""
This metric penalizes smiles strings that are too long, assuming that the
canonic smile is the shortest representation.
"""


def batch_conciseness(smiles, train_smiles=None):
    vals = [conciseness(s) if verify_sequence(s) else 0 for s in smiles]
    return vals


def conciseness(smile, train_smiles=None):
    canon = canon_smile(smile)
    diff_len = len(smile) - len(canon)
    val = np.clip(diff_len, 0.0, 20)
    val = 1 - 1.0 / 20.0 * val
    return val

#
# 2.9. Synthetic accesibility
#

"""
This metric checks whether a given molecule is easy to synthesize or not.
It is based on (although not completely equivalent to) the work of Ertl
and Schuffenhauer.

Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility 
score of drug-like molecules based on molecular complexity and fragment contributions. 
Journal of cheminformatics, 1(1), 8.
"""


def batch_SA(smiles, train_smiles=None):
    vals = [apply_to_valid(s, SA_score) for s in smiles]
    return vals


def SA_score(mol):
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

#
# 2.10. Drug-likeness
#

"""
This metric computes the 'drug-likeness' of a given molecule through
an equally ponderated mean of the following factors:

    - Logarithm of the water-octanol partition coefficient
    - Synthetic accesibility
    - Conciseness of the molecule
    - Soft novelty
"""


def batch_drugcandidate(smiles, train_smiles=None):
    vals = [drug_candidate(s, train_smiles)
            if verify_sequence(s) else 0 for s in smiles]
    return vals


def drug_candidate(smile, train_smiles):
    mol = Chem.MolFromSmiles(smile)
    good_logp = constant_bump(logP(mol), 0.210, 0.945)
    sa = SA_score(mol)
    novel = soft_novelty(smile, train_smiles)
    compact = conciseness(smile)
    val = (compact + good_logp + sa + novel) / 4.0
    return val

#
# 2.11. Lipinski's rule of five
#

"""
This metric assigns 1.0 if the molecule follows Lipinski's rule of
five and 0.0 if not.
"""


def batch_lipinski(smiles, train_smiles):
    vals = [apply_to_valid(s, Lipinski) for s in smiles]
    return vals


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

#
# 2.12. NP-likeness
#

"""
This metric computes the likelihood that a given molecule is
a natural product.
"""


def batch_NPLikeliness(smiles, train_smiles=None):
    vals = [apply_to_valid(s, NP_score) for s in smiles]
    return vals


def NP_score(mol):
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

#
# 2.13. PCE
#

"""
This metric computes the Power Conversion Efficiency of a organic
solar cell built with a given molecule, using a CNN on a 64x64/32x32
bit array containing Morgan fingerprints.
"""


def batch_PCE(smiles, train_smiles=None):
    cnn = cnn_pce(lbit=32)
    vals = [cnn.predict(smile) if verify_sequence(smile)
            else 0.0 for smile in smiles]
    return remap(vals, np.amin(vals), np.amax(vals))

#
# 2.14. Bandgap
#

"""
This metric computes the HOMO-LUMO energy difference of a given 
molecule,using a CNN on a 64x64/32x32 bit array containing Morgan fingerprints.
"""


def batch_bandgap(smiles, train_smiles=None):
    cnn = cnn_homolumo(lbit=32)
    vals = [cnn.predict(smile) if verify_sequence(smile)
            else 0.0 for smile in smiles]
    return remap(vals, np.amin(vals), np.amax(vals))

#
# 2.15. Substructure match
#

"""
This metric assigns 1.0 if a previously defined substructure (through the global 
variable obj_substructure) is present in a given molecule, and 0.0 if not.
"""


def batch_substructure_match_all(smiles, train_smiles=None):
    if ALL_POS_PATTS == None:
        print('No substructures has been specified')
        raise

    vals = [apply_to_valid(s, substructure_match_all) for s in smiles]
    return vals


def substructure_match_all(mol, train_smiles=None):
    val = all([mol.HasSubstructMatch(patt) for patt in ALL_POS_PATTS])
    return int(val)


def batch_substructure_match_any(smiles, train_smiles=None):
    if ANY_POS_PATTS == None:
        print('No substructures has been specified')
        raise

    vals = [apply_to_valid(s, substructure_match_any) for s in smiles]
    return vals


def substructure_match_any(mol, train_smiles=None):
    val = any([mol.HasSubstructMatch(patt) for patt in ANY_POS_PATTS])
    return int(val)


def batch_substructure_absence(smiles, train_smiles=None):
    if ALL_NEG_PATTS == None:
        print('No substructures has been specified')
        raise

    vals = [apply_to_valid(s, substructure_match_any) for s in smiles]
    return vals


def substructure_absence(mol, train_smiles=None):
    val = all([not mol.HasSubstructMatch(patt) for patt in ANY_NEG_PATTS])
    return int(val)

#
# 2.16. Chemical beauty
#

"""
Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). Quantifying the chemical beauty of drugs. Nature chemistry, 4(2), 90-98.
"""


def batch_beauty(smiles, train_smiles=None):
    vals = [apply_to_valid(s, chemical_beauty) for s in smiles]
    return vals


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

#
# 2.17. Density
#

"""
This metric computes the density of a given molecule, using a CNN 
on a 64x64/32x32 bit array containing Morgan fingerprints.
"""


def batch_density(smiles, train_smiles=None):
    cnn = cnn_density(lbit=32)
    vals = [cnn.predict(smile) if verify_sequence(smile)
            else 0.0 for smile in smiles]
    return remap(vals, np.amin(vals), np.amax(vals))

#
# 2.18. Melting point
#

"""
This metric computes the melting point of a given  molecule, using a 
CNN on a 64x64/32x32 bit array containing Morgan fingerprints.
"""


def batch_mp(smiles, train_smiles=None):
    cnn = cnn_mp(lbit=32)
    vals = [cnn.predict(smile) if verify_sequence(smile)
            else 0.0 for smile in smiles]
    return remap(vals, np.amin(vals), np.amax(vals))

#
# 2.19. Melting point
#

"""
This metric computes the mutagenicity of a given  molecule, using a 
CNN on a 64x64/32x32 bit array containing Morgan fingerprints.
"""


def batch_mp(smiles, train_smiles=None):
    cnn = cnn_mp(lbit=32)
    vals = [cnn.predict(smile) if verify_sequence(smile)
            else 0.0 for smile in smiles]
    return remap(vals, np.amin(vals), np.amax(vals))

############################################
#
#   3. LOAD ALL REWARDS
#
#############################################
#
#
#

def get_metrics():
    metrics = OrderedDict()
    metrics['validity'] = batch_validity
    metrics['novelty'] = batch_novelty
    metrics['creativity'] = batch_creativity
    metrics['hard_novelty'] = batch_hardnovelty
    metrics['soft_novelty'] = batch_softnovelty
    metrics['diversity'] = batch_diversity
    metrics['variety'] = batch_variety
    metrics['symmetry'] = batch_symmetry
    metrics['conciseness'] = batch_conciseness
    metrics['solubility'] = batch_solubility
    metrics['naturalness'] = batch_NPLikeliness
    metrics['synthesizability'] = batch_SA
    metrics['lipinski'] = batch_lipinski
    metrics['drug_candidate'] = batch_drugcandidate
   # metrics['pce'] = batch_PCE
   # metrics['bandgap'] = batch_bandgap
   # metrics['substructure_match'] = batch_substructure_match
    metrics['chemical_beauty'] = batch_beauty
    return metrics
