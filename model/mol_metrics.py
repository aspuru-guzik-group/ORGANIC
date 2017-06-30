from __future__ import absolute_import, division, print_function
from builtins import range
import os
import numpy as np
import pubchempy as pcp
import tensorflow as tf
import csv
import time
import pickle
import gzip
import math
import random
import cirpy
import pymatgen as mg
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, MolToSmiles
from rdkit.Chem.Fingerprints import FingerprintMols
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import rdkit.Chem.inchi as inchi
from chemspipy import ChemSpider
from chembl_webresource_client import *
from cnn_metrics import cnn_pce, cnn_homolumo

# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

############################################
#
#   1. GLOBAL FUNCTIONS AND UTILITIES
#
#       1.1. Loading models
#       1.2. Loading utilities
#       1.3. Math utilities
#       1.4. Encoding/decoding utilities
#       1.5. Results utilities
#
#############################################
#
#
#

#
# 1.1. Loading models
#

global NP_LOADED, SA_LOADED
NP_LOADED = False
SA_LOADED = False

global BITSIDE
BITSIDE = 64

#
# 1.2. Loading utilities
#

def load_train_data(filename):
    ext = filename.split(".")[-1]
    if ext == 'csv':
        return read_smiles_csv(filename)
    if ext == 'smi':
        return read_smi(filename)
    else:
        raise ValueError('data is not smi or csv!')
    return

def read_smiles_csv(filename):
    # Assumes smiles is in column 0
    with open(filename) as file:
        reader = csv.reader(file)
        smiles_idx = next(reader).index("smiles")
        data = [row[smiles_idx] for row in reader]
    return data

def save_smi(name, smiles):
    if not os.path.exists('epoch_data'):
        os.makedirs('epoch_data')
    smi_file = os.path.join('epoch_data', "{}.smi".format(name))
    with open(smi_file, 'w') as afile:
        afile.write('\n'.join(smiles))
    return

def read_smi(filename):
    with open(filename) as file:
        smiles = file.readlines()
    smiles = [i.strip() for i in smiles]
    return smiles

#
# 1.3. Math utilities
#

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def constant_bump(x, x_low, x_high, decay=0.025):
    if x <= x_low:
        return np.exp(-(x - x_low)**2 / decay)
    elif x >= x_high:
        return np.exp(-(x - x_high)**2 / decay)
    else:
        return 1
    return

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

#
# 1.4. Encoding/decoding utilities
#

def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile))

def verified_and_below(smile, max_len):
    return len(smile) < max_len and verify_sequence(smile)

def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1

def restrict_to_valid(smile):
    mol = Chem.MolFromSmiles(smile)
    if (smile != '' and mol is not None and mol.GetNumAtoms() > 1):
        return smile
    return None

def filter_smiles(smiles):
    mols = []
    for smile in smiles:
        if verify_sequence(smile) == True:
            mols.append(smile)
    return mols

def build_vocab(smiles, pad_char='_', start_char='^'):
    i = 1
    char_dict, ord_dict = {start_char: 0}, {0: start_char}
    for smile in smiles:
        for c in smile:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    char_dict[pad_char], ord_dict[i] = i, pad_char
    return char_dict, ord_dict

def pad(smile, n, pad_char='_'):
    if n < len(smile):
        return smile
    return smile + pad_char * (n - len(smile))

def unpad(smile, pad_char='_'): return smile.rstrip(pad_char)

def encode(smile, max_len, char_dict): return [
    char_dict[c] for c in pad(smile, max_len)]

def decode(ords, ord_dict): return unpad(
    ''.join([ord_dict[o] for o in ords]))

#
# 1.5. Results utilities
#

def print_params(p):
    print('Using parameters:')
    if (type(p['TOTAL_BATCH']) is list) or (type(p['OBJECTIVE']) is list):
        for key, value in p.items():
            if key == 'OBJECTIVE' or key == 'TOTAL_BATCH':
                print('{:20s}'.format(key))
                for each in value:
                    print('     {}'.format(each))
            else:
                print('{:20s} - {:12}'.format(key, value))
    else:
        for key, value in p.items():
            print('{:20s} - {:12}'.format(key, value))
    print('rest of parameters are set as default\n')
    return

def compute_results(model_samples, train_data, ord_dict, results={}, verbose=True):
    samples = [decode(s, ord_dict) for s in model_samples]
    results['mean_length'] = np.mean([len(sample) for sample in samples])
    results['n_samples'] = len(samples)
    results['uniq_samples'] = len(set(samples))
    verified_samples = []
    unverified_samples = []
    for sample in samples:
        if verify_sequence(sample):
            verified_samples.append(sample)
        else:
            unverified_samples.append(sample)
    results['good_samples'] = len(verified_samples)
    results['bad_samples'] = len(unverified_samples)
    # # collect metrics
    # metrics = ['novelty', 'hard_novelty', 'soft_novelty',
    #            'diversity', 'conciseness', 'solubility',
    #            'naturalness', 'synthesizability']

    # for objective in metrics:
    #     func = load_reward(objective)
    #     results[objective] = np.mean(func(verified_samples, train_data))

    # save smiles
    if 'Batch' in results.keys():
        smi_name = '{}_{}'.format(results['exp_name'], results['Batch'])
        save_smi(smi_name, samples)
        results['model_samples'] = smi_name
    # print results
    if verbose:
        # print_results(verified_samples, unverified_samples, metrics, results)
        print_results(verified_samples, unverified_samples, results)
    return

# def print_results(verified_samples, unverified_samples, metrics, results={}):
def print_results(verified_samples, unverified_samples, results={}):
    print('~~~ Summary Results ~~~')
    print('{:15s} : {:6d}'.format("Total samples", results['n_samples']))
    percent = results['uniq_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format(
        'Unique', results['uniq_samples'], percent))
    percent = results['bad_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format('Unverified',
                                             results['bad_samples'], percent))
    percent = results['good_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format(
        'Verified', results['good_samples'], percent))
    # print('\tmetrics...')
    # for i in metrics:
    #     print('{:20s} : {:1.4f}'.format(i, results[i]))

    if len(verified_samples) > 10:
        print('\nExample of good samples:')

        for s in verified_samples[0:10]:
            print('' + s)
    else:
        print('\nno good samples found :(')

    if len(unverified_samples) > 10:
        print('\nExample of bad samples:')
        for s in unverified_samples[0:10]:
            print('' + s)
    else:
        print('\nno bad samples found :(')

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    return


############################################
#
#   2.MOLECULAR METRICS
#
#       2.1. Diversity
#       2.2. Variety
#       2.3. Novelty, hardnovelty and softnovelty
#       2.4. Creativity
#       2.5. Symmetry
#       2.6. Solubility
#       2.7. Conciseness
#       2.8. Synthetic accesibility
#       2.9. Druglikeness
#       2.10. Lipinski's rule of five
#       2.11. NP-likeness
#       2.12. PCE
#       2.13. Bandgap
#       2.14. Substructure match
#
#############################################
#
#
#

#
# 2.1. Diversity
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
    vals = [diversity(s, fps) if verify_sequence(s)
            else 0.0 for s in smiles]
    return vals

def diversity(smile, fps):
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    ref_mol = Chem.MolFromSmiles(smile)
    ref_fps = Chem.GetMorganFingerprintAsBitVect(ref_mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(
        ref_fps, fps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    val = remap(mean_dist, low_rand_dst, mean_div_dst)
    val = np.clip(val, 0.0, 1.0)
    return val

#
# 2.2. Variety
#

"""
This metric compares the Tanimoto distance of a given molecule
with a random sample of the other generated smiles.
"""

def batch_variety(smiles, train_smiles=None):
    mols = [Chem.MolFromSmiles(smile) for smile in np.random.choice(filter_smiles(smiles), len(smiles)/10)]
    setfps = [Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in mols]
    vals = [variety(smile, setfps) if verify_sequence(smile) else 0.0 for smile in smiles]
    return vals

def variety(smile, setfps):
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    mol = Chem.MolFromSmiles(smile)
    fp = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(fp, setfps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    val = remap(mean_dist, low_rand_dst, mean_div_dst)
    val = np.clip(val, 0.0, 1.0)
    return val

#
# 2.3. Novelty
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
# 2.4. Creativity
#

"""
This metric computes the Tanimoto distance of a smile to the training set,
as a measure of how different these molecules are from the provided ones.
"""

def batch_creativity(smiles, train_smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in filter_smiles(train_smiles)]
    setfps = [Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in mols]
    vals = [creativity(smile, setfps) if verify_sequence(smile) else 0.0 for smile in smiles]
    return vals

def creativity(smile, setfps):
    mol = Chem.MolFromSmiles(smile)
    return np.mean(DataStructs.BulkTanimotoSimilarity(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048), setfps))

#
# 2.5. Symmetry
#

"""
This metric yields 1.0 if the generated molecule has any element of symmetry, and 0.0
if the point group is C1.
"""

def batch_symmetry(smiles, train_smiles=None):
    vals = [symmetry(smile) if verify_sequence(smile) else 0 for smile in smiles]
    return vals

def symmetry(smile):
    ids, xyz = get3DCoords(smile)
    sch_symbol = getSymmetry(ids, xyz)
    return 1.0 if sch_symbol != 'C1' else 0.0

def get3DCoords(smile):
    molec = Chem.MolFromSmiles(smile)
    mwh = Chem.AddHs(molec)
    Chem.EmbedMolecule(mwh)
    Chem.UFFOptimizeMolecule(mwh)
    ids = []
    xyz = []
    for i in range(mwh.GetNumAtoms()):
        pos = mwh.GetConformer().GetAtomPosition(i)
        ids.append(mwh.GetAtomWithIdx(i).GetSymbol())
        xyz.append([pos.x, pos.y, pos.z])
    return ids, xyz

def getSymmetry(ids, xyz):
    mol = PointGroupAnalyzer(mg.Molecule(ids, xyz))
    return mol.sch_symbol

#
# 2.6. Solubility
#

"""
This metric computes the logarithm of the water-octanol partition
coefficient, using RDkit's implementation of Wildman-Crippen method, 
and then remaps it to the 0.0-1.0 range.

Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions. 
Journal of chemical information and computer sciences, 39(5), 868-873.
"""

def batch_solubility(smiles, train_smiles=None):
    vals = [logP(s, train_smiles) if verify_sequence(s) else 0 for s in smiles]
    return vals


def logP(smile, train_smiles=None):
    low_logp = -2.12178879609
    high_logp = 6.0429063424
    logp = Crippen.MolLogP(Chem.MolFromSmiles(smile))
    val = remap(logp, low_logp, high_logp)
    val = np.clip(val, 0.0, 1.0)
    return val

#
# 2.7. Conciseness
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
    diff_len = len(smile) -len(canon)
    val = np.clip(diff_len, 0.0, 20)
    val = 1 - 1.0 / 20.0 * val
    return val

#
# 2.8. Synthetic accesibility
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
    if SA_LOADED == False:
        global SA_model
        SA_model = readSAModel()
        SA_LOADED = True
    scores = [SA_score(s) for s in smiles]
    return scores

def readSAModel(filename='../data/SA_score.pkl.gz'):
    print("mol_metrics: reading SA model ...")
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

def SA_score(smile):
    mol = Chem.MolFromSmiles(smile)

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
# 2.9. Drug-likeness
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
    good_logp = constant_bump(logP(smile), 0.210, 0.945)
    sa = SA_score(smile)
    novel = soft_novelty(smile, train_smiles)
    compact = conciseness(smile)
    val = (compact + good_logp + sa + novel) / 4.0
    return val

#
# 2.10. Lipinski's rule of five
#

"""
This metric assigns 1.0 if the molecule follows Lipinski's rule of
five and 0.0 if not.
"""

def batch_lipinski(smile, train_smiles):
    vals = [Lipinski(smile) if verify_sequence(smile) else 0.0 for smile in smiles]
    return vals

def Lipinski(smile):
    mol = Chem.MolFromSmiles(smile)
    druglikeness = 0.0
    druglikeness += 0.25 if logP(mol) == 0 else 0.0
    druglikeness += 0.25 if Chem.Descriptors.MolWt(mol) <= 500 else 0.0
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
        donors += 1 if ((a1, a2) == (1, 8)) or ((a1, a2) == (8,1)) else 0.0
        donors += 1 if ((a1, a2) == (1, 7)) or ((a1, a2) == (7,1)) else 0.0
    druglikeness += 0.25 if donors <= 5 else 0.0
    return druglikeness

#
# 2.11. NP-likeness
#

"""
This metric computes the likelihood that a given molecule is
a natural product.
"""


def batch_NPLikeliness(smiles, train_smiles=None):
    if NP_LOADED == False:
        global NP_model
        NP_model = readNPModel()
        NP_LOADED = True
    scores = [NP_score(s) if verify_sequence(s) else 0 for s in smiles]
    return scores

def readNPModel(filename='../data/NP_score.pkl.gz'):
    print("mol_metrics: reading NP model ...")
    start = time.time()
    NP_model = pickle.load(gzip.open(filename))
    end = time.time()
    print("loaded in {}".format(end - start))
    return NP_model

def NP_score(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = Chem.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()

    # calculating the score
    score = 0.
    for bit in bits:
        score += NP_model.get(bit, 0)
    score /= float(mol.GetNumAtoms())

    # preventing score explosion for exotic molecules
    if score > 4:
        score = 4. + math.log10(score - 4. + 1.)
    if score < -4:
        score = -4. - math.log10(-4. - score + 1.)
    val = np.clip(remap(score, -3, 1), 0.0, 1.0)
    return val

#
# 2.12. PCE
#

"""
This metric computes the Power Conversion Efficiency of a organic
solar cell built with a given molecule, using a CNN on a 64x64
bit array containing Morgan fingerprints.
"""

def batch_PCE(smiles, train_smiles=None):
    cnn = cnn_pce(lbit=BITSIDE)
    vals = [cnn.predict(smile) if verify_sequence(smile) else 0.0 for smile in smiles]
    return vals

#
# 2.13. Bandgap
#

"""
This metric computes the HOMO-LUMO energy difference of a given 
molecule,using a CNN on a 64x64 bit array containing Morgan fingerprints.
"""

def batch_bandgap(smiles, train_smiles=None):
    cnn = cnn_homolumo(lbit=BITSIDE)
    vals = [cnn.predict(smile) if verify_sequence(smile) else 0.0 for smile in smiles]
    return vals

#
# 2.14. Substructure match
#

"""
This metric assigns 1.0 if a previously defined substructure (through the global 
variable obj_substructure) is present in a given molecule, and 0.0 if not.
"""

def batch_substructure_match(smiles, train_smiles=None):
    if substructure_match == None:
        print('No substructure has been specified')
        raise
    vals = [substructure_match(smile, sub_mol=obj_substructure) if verify_sequence(smile) else 0.0 for smile in smiles]

def substructure_match(smile, train_smiles=None, sub_mol=None):
    mol = Chem.MolFromSmiles(smile)
    val = mol.HasSubstructMatch(sub_mol)
    return int(val)

############################################
#
#   3. LOAD ALL REWARDS
#
#############################################
#
#
#

def load_reward(objective):

    metrics = {}
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
    metrics['pce'] = batch_PCE
    metrics['bandgap'] = batch_bandgap
    metrics['substructure_match'] = batch_substructure_match
    if objective in metrics.keys():
        return metrics[objective]
    else:
        raise ValueError('objective {} not found!'.format(objective))
    return
