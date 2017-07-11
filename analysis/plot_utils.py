import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import cairosvg
import scipy.stats as stats
from IPython.core.display import display, HTML
import markdown2

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


def save_svg(svg, svg_file, dpi=150):
    png_file = svg_file.replace('.svg', '.png')
    with open(svg_file, 'w') as afile:
        afile.write(svg)
    cairosvg.svg2png(bytestring=svg.encode(
        'utf-8'), write_to=png_file, dpi=dpi)
    return


def save_result(name):
    plt.savefig('results/%s.png' % name, dpi=300)
    plt.savefig('results/%s.svg' % name, dpi=300)
    return


def plot_distributions_compare(prop, func, smiles, train_smiles, mm, both=True):
    mm.NORMALIZE = False
    y = func(smiles, train_smiles)
    sns.distplot(y)
    stats_box(y)
    plt.xlabel(prop)
    sns.despine(offset=True)
    plt.show()
    if both:
        mm.NORMALIZE = True
        y = func(smiles, train_smiles)
        sns.distplot(y)
        stats_box(y)
        plt.xlabel(prop + ' normalized')
        sns.despine(offset=True)
        plt.show()
    return

def html_header(txt):
    return display(HTML(markdown2.markdown(txt)))
