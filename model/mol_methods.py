from __future__ import absolute_import, division, print_function
import os
import csv
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles

def read_smi(filename):
    """
    Function to read a .smi file.

    """

    with open(filename) as file:
        smiles = file.readlines()
    smiles = [i.strip() for i in smiles]
    return smiles


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


#
# 1.3. Math utilities
#


def gauss_remap(x, x_mean, x_std):
    return np.exp(-(x - x_mean)**2 / (x_std**2))


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def constant_range(x, x_low, x_high):
    if hasattr(x, "__len__"):
        return np.array([constant_range_func(xi, x_low, x_high) for xi in x])
    else:
        return constant_range_func(x, x_low, x_high)


def constant_range_func(x, x_low, x_high):
    if x <= x_low or x >= x_high:
        return 0
    else:
        return 1


def constant_bump_func(x, x_low, x_high, decay=0.025):
    if x <= x_low:
        return np.exp(-(x - x_low)**2 / decay)
    elif x >= x_high:
        return np.exp(-(x - x_high)**2 / decay)
    else:
        return 1


def constant_bump(x, x_low, x_high, decay=0.025):
    if hasattr(x, "__len__"):
        return np.array([constant_bump_func(xi, x_low, x_high, decay) for xi in x])
    else:
        return constant_bump_func(x, x_low, x_high, decay)


def smooth_plateau(x, x_point, decay=0.025, increase=True):
    if hasattr(x, "__len__"):
        return np.array([smooth_plateau_func(xi, x_point, decay, increase) for xi in x])
    else:
        return smooth_plateau_func(x, x_point, decay, increase)


def smooth_plateau_func(x, x_point, decay=0.025, increase=True):
    if increase:
        if x <= x_point:
            return np.exp(-(x - x_point)**2 / decay)
        else:
            return 1
    else:
        if x >= x_point:
            return np.exp(-(x - x_point)**2 / decay)
        else:
            return 1


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


def apply_to_valid(smile, fun, **kwargs):
    mol = Chem.MolFromSmiles(smile)
    return fun(mol, **kwargs) if smile != '' and mol is not None and mol.GetNumAtoms() > 1 else 0.0


def filter_smiles(smiles):
    return [smile for smile in smiles if verify_sequence(smile)]


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

