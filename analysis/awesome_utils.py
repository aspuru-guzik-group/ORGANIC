import scipy as sp
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import cairosvg
import scipy.stats as stats
import pandas as pd
from PIL import Image
from scipy.stats import hypergeom
from collections import defaultdict
from rdkit.Chem import rdMolDescriptors as rdmd


def error_stats(x, y, w=None, verbose=False):
    if w is None:
        w = np.ones(len(y))
    mae = metrics.mean_absolute_error(x, y, w)
    mae_std = np.std(np.abs(x - y))
    rmse = np.sqrt(metrics.mean_squared_error(x, y, w))
    r2 = sp.stats.pearsonr(x, y)[0]
    R2 = metrics.r2_score(x, y)
    if verbose:
        print('MAE  = %.4f +/- %3.4f' % (mae, mae_std))
        print('RMSD = %.4f ' % (rmse))
        print('r^2  = %.3f , R^2  = %.3f ' % (r2, R2))

    return mae, mae_std, rmse, r2, R2



def data_scatter(data, prop, label_x, label_y, c, title, cmap=None):
    x = data[label_x]
    y = data[label_y]
    if cmap is None:
        plt.scatter(x, y, c=c, s=50, alpha=0.75, label='')
    else:
        z = data[c]
        plt.scatter(x, y, c=z, s=50, alpha=0.75, label='', cmap=cmap)

    xmin, xmax = np.min(x), np.max(x)
    # ideal fit
    mae, mae_std, rmse, r2, R2 = error_stats(x, y, verbose=False)

    info_str = '\n'.join(['ideal fit',
                          '$R^2=%.3f$' % R2,
                          '$r^2=%.3f$' % r2,
                          'MAE =%2.3f (%3.2f)' % (mae, mae_std),
                          'RMSE  =%2.3f' % rmse])
    plt.plot([xmin, xmax], [xmin, xmax], ls='--',
             c='k', alpha=0.5, lw=3, label=info_str)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    sns.despine()
    plt.title(title)
    plt.legend(loc='upper left')
    if cmap is not None:
        plt.colorbar()
    plt.savefig('results/%s_%s_%s.png' % (prop, label_x, label_y), dpi=300)
    plt.savefig('results/%s_%s_%s.svg' % (prop, label_x, label_y), dpi=300)
    plt.show()
    return


def basic_stats(y, verbose=False):
    mean_y, std_y = np.mean(y), np.std(y)
    min_y, max_y = np.min(y), np.max(y)
    low_y, high_y = mean_y - std_y * 2, mean_y + std_y * 2
    if verbose:
        print('Mean, std : %.3f +/- %.3f' % (mean_y, std_y))
        print('Min/Max   : [%.3f , %.3f] ' % (min_y, max_y))
        print('95%% range : [%.3f , %.3f] ' % (low_y, high_y))

    return mean_y, std_y, min_y, max_y, low_y, high_y


def linear_fit(x, y):
    alpha, beta, r1, p_value, std_err = sp.stats.linregress(x, y)
    polynomial = np.poly1d([alpha, beta])
    fit_y = polynomial(x)
    fit_x = (y - beta) / alpha
    mean, std, r1, r2 = error_stats(fit_y, y)

    return fit_x, fit_y, mean, std, r1, r2


def awesome_settings():
    # awesome plot options
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=2)
    sns.set_palette(sns.color_palette('bright'))
    # image stuff
    plt.rcParams['figure.figsize'] = (8.0, 4.0)
    plt.rcParams['savefig.dpi'] = 60
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.shadow'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.autolayout'] = True

    return


def color_properties(props):
    random_c = random.sample(sns.color_palette("Set2", 11), k=len(props))
    cmap_dict = {p: sns.light_palette(c, reverse=True, as_cmap=True,
                                      input="rgb") for p, c in zip(props, random_c)}
    col_dict = {p: c(0) for p, c in cmap_dict.items()}

    return col_dict, cmap_dict


def stats_box(y):
    result = stats.describe(y)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ex_labels = []
    ex_labels.append('$n=%d$' % (result.nobs))
    ex_labels.append('$\mu=%2.2f$' % (result.mean))
    ex_labels.append('$med=%2.2f$' % (np.median(y)))
    ex_labels.append('$\sigma=%2.2f$' % (np.sqrt(result.variance)))
    ex_labels.append('$\min=%2.2f$' % (result.minmax[0]))
    ex_labels.append('$\max=%2.2f$' % (result.minmax[1]))
    ex_handles = [mpatches.Patch(
        color='white', alpha=0.0, visible=False) for i in ex_labels]
    plt.legend(handles=handles + ex_handles, labels=labels + ex_labels,
               loc='best', frameon=False)

    return


def smile2Mol(row):
    return Chem.MolFromSmiles(row['smiles'])


def save_svg(svg, svg_file, dpi=150):
    png_file = svg_file.replace('.svg', '.png')
    with open(svg_file, 'w') as afile:
        afile.write(svg)
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_file, dpi=dpi)
    return


def save_result(name):
    plt.savefig('results/%s.png' % name, dpi=300)
    plt.savefig('results/%s.svg' % name, dpi=300)
    return

# Start by importing some code to allow the depiction to be used:
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
from rdkit.Chem import AllChem as Chem

# a function to make it a bit easier. This should probably move to somewhere in
# rdkit.Chem.Draw


def _prepareMol(mol, kekulize):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc


def moltosvg(mol, molSize=(450, 200), kekulize=True, drawer=None, **kwargs):
    mc = _prepareMol(mol, kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc, **kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg.replace('svg:', '')


def getSubstructDepiction(mol, atomID, radius):
    if radius > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
        atomsToUse = []
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env = None
    return atomsToUse


def bitImage(bitId, mol, rad=4, nBits=1024):
    info = {}
    fp = Chem.GetMorganFingerprintAsBitVect(mol, rad, nBits, bitInfo=info)
    aid, rad = info[bitId][0]
    highlight = getSubstructDepiction(mol, aid, rad)
    img = moltosvg(mol, legend='Bit %d' % bitId,
                   highlightAtoms=highlight,
                   highlightAtomColors={aid: (0.3, 0.3, 1)})
    return img


def count_radius(bit_df, data_df):
    rads = {i: [] for i in range(len(bit_df))}
    for indx, row in data_df.iterrows():
        for bitId, value in row['FP_info'].items():
            rads[bitId].append(value[0][1])
    return rads


def bit_radius(row, rads):
    bitId = row['Bit']
    return pd.Series([np.mean(rads[bitId]), np.std(rads[bitId])])


def bit_example(row, data_df):
    bitId = int(row['Bit'])
    for indx, drow in data_df.sample(frac=1).iterrows():
        if drow['FP'][bitId] == 1:
            return pd.Series([drow['Mol'], drow['FP_info']])

    print('Did not find example for %d' % row['Bit'])

    return


def bitGrid(bit_df, svg=True, nrows=6,ex_info=''):

    mols, labels, highlights, atomlights = [], [], [], []

    for bitId, row in bit_df.iterrows():
        mol = row['Example']
        bitId = row['Bit']
        aid, rad = row['Example_info'][bitId][0]
        highlights.append(getSubstructDepiction(mol, aid, rad))
        atomlights.append({aid: (0.3, 0.3, 1)})
        if ex_info != '':
            labels.append('Bit %d (%2.2f)' % (bitId, row[ex_info]))
        else:
            labels.append('Bit %d' % bitId)
        mols.append(mol)

    img_grid = Draw.MolsToGridImage(mols, legends=labels,
                                    highlightAtomLists=highlights,
                                    highlightAtomColorsLists=atomlights,
                                    molsPerRow=nrows, useSVG=svg)
    return img_grid


def mexi_colormap(values, n=2):
    minv, maxv = np.min(values), np.max(values)
    cmap = sns.diverging_palette(10, 150, as_cmap=True)
    norm = MidpointNormalize(vmin=minv, vmax=maxv, midpoint=0.)
    pal = sns.diverging_palette(10, 150, n=n)

    def color_map(v):
        return cmap(norm(v))
    return color_map, pal


class MidpointNormalize(mpl.colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def smart_sample(df, col, how='log'):
    vals = df[col].dropna()
    # how to not plot?
    hist, bins, = np.histogram(vals.values, bins='scott')
    samples = []
    last_bin = bins[0] - 0.1
    for b in bins:
        sub_df = vals[np.logical_and(vals > last_bin, vals <= b)]
        if len(sub_df > 0):
            if how == 'log':
                sub_n = np.floor(np.log10(len(sub_df))) + 1
            else:
                sub_n = 1
            samples += [i for i in sub_df.sample(n=sub_n).index]
        last_bin = b

    return df.ix[samples, :]


def hypergeomZ_bit(row, df, metric, tol):
    if row['Bit'] % 1000 == 0:
        print(int(row['Bit']), end=' ')
    # Hypergeometric parameters consistent with wikipedia
    N = len(df)
    K = row['All_count']

    df_top = df[df[metric] >= tol]
    n = len(df_top)
    k = row['Top_count']

    frac_top = float(k) / float(n)
    if frac_top > 0:
        mean_dist = hypergeom.mean(N, K, n) / float(n)
        std_dist = hypergeom.std(N, K, n) / float(n)
        z_score = (frac_top - mean_dist) / std_dist
    else:
        z_score = np.nan
    return z_score


def hypergeomZ(df, metric, top, col, col_value=True):

    df_top = df[df[metric] > top]
    # Hypergeometric parameters consistent with wikipedia
    N = len(df)
    K = len(df[df[col] == col_value])
    n = len(df_top)
    k = len(df_top[df_top[col] == col_value])
    frac_top = float(k) / float(n)
    if frac_top > 0:
        mean_dist = hypergeom.mean(N, K, n) / float(n)
        std_dist = hypergeom.std(N, K, n) / float(n)
        z_score = (frac_top - mean_dist) / std_dist
    else:
        z_score = np.nan
    return z_score


def count_collision(df):
    radii = [1, 2, 3, 4, 5]
    lengths = [512, 1024, 2048, 4096, 8192]
    counts = defaultdict(list)
    for indx, row in df.iterrows():
        m = row['Mol']
        for rad in radii:
            counts[
                (rad, -1)].append(len(rdmd.GetMorganFingerprint(m, rad).GetNonzeroElements()))
            for l in lengths:
                counts[(rad, l)].append(
                    rdmd.GetMorganFingerprintAsBitVect(m, rad, l).GetNumOnBits())
    return counts


def plot_collisions(counts):
    radii = [1, 2, 3, 4, 5]
    lengths = [512, 1024, 2048, 4096, 8192]
    cols = sns.color_palette("Set2", len(radii))
    for pidx, nbits in enumerate(reversed(lengths)):
        colls = []
        labels = ['$r={}$'.format(rad) for rad in radii]
        for rad in radii:
            v1 = np.array(counts[1, -1])
            v2 = np.array(counts[1, nbits])
            colls.append(v1 - v2)

        plt.hist(colls, log=True, label=labels, color=cols)
        plt.title('%d bits' % nbits)
        plt.xlabel('# Collisions')
        plt.ylabel('# Molecules')
        plt.legend()
        sns.despine(trim=True)
        plt.show()
    return


def recolor_image(img_path, col):
    img = Image.open(img_path)
    img = img.convert('L')
    recolor = Image.new('RGBA', (img.size))
    rgb = img.load()
    for x in range(img.width):
        for y in range(img.height):
            L = 255 - img.getpixel((x, y))
            recolor.putpixel((x, y), (col[0], col[1], col[2], L))
    return recolor
