from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range, str
import re
import numpy as np
import matplotlib.pyplot as plt
from nlputils.dict_utils import invert_dict0
from nlputils.visualize import get_colors


def check_and(*args):
    return lambda x: all(q in x for q in args), "and:"+str(args)


def check_or(*args):
    return lambda x: any(q in x for q in args), "or:"+str(args)


def check_in(q):
    return lambda x: q in x, q


def check_occurrences(textdict, doccats, queries):
    """
    For all queries, check how often they occur in documents of a specific class

    Inputs:
        textdict: dict with {doc_id: text}
        doccats: dict with {doc_id: category}
        queries: some queries to check for; either strings or using check_and and check_or, e.g.
                 ['hello', check_and('italy', 'earthquake'), check_or('trump', 'obama')]
                 - due to preprocessing constraints, all query words have to be single words!
    Returns:
        results: a dict with {query: {category: frequency}}, e.g.
                 {'hello': {'politics': 0., 'world': 0.01},
                  'and:(italy, earthquake)': {'politics': 0.1, 'world': 0.2},
                  'or:(trump, obama)': {'politics': 0.9, 'world': 0.1}}
    """
    # invert doccats to get for every category the list of documents in it
    catdocs = invert_dict0(doccats)
    # do some preprocessing
    textdict = {did: set(re.findall(r"[a-z0-9-]+", textdict[did].lower())) for did in textdict}
    # check for all queries
    results = {}
    for q in queries:
        # convert regular string queries into functions as well
        if isinstance(q, str):
            q = check_in(q)
        # split in name to store results and query itself
        q, str_q = q
        results[str_q] = {}
        for cat in catdocs:
            results[str_q][cat] = len([1 for did in catdocs[cat] if q(textdict[did])]) / float(len(catdocs[cat]))
    return results


def vis_occurrences(results, bars=False, queries=[]):
    """
    Visualize the results from check_occurrences.

    Inputs:
        results: a dict with {query: {category: frequency}}
        bars: whether to create a line plot (default; good if categories represent timestamps
              or there are a lot of categories and queries)
              or a bar chart (if bars=True; better for actual categories)
        queries: optional list with queries, has to be keys to results
    """
    if not queries:
        queries = sorted(results.keys())
        print(queries)
    categories = sorted(results[queries[0]].keys())
    cat_names = [str(c).replace('_', '\n') for c in categories]
    r = 90 if max([len(c) for c in cat_names]) > 6 else 0
    colors = get_colors(len(queries))
    plt.figure()
    if not bars:
        for i, q in enumerate(queries):
            plt.plot([results[q][cat] for cat in categories], color=colors[i], label=q)
        plt.xticks(list(range(len(categories)))[::max(1, len(categories)//10)], cat_names[::max(1, len(categories)//10)], rotation=r)
        plt.xlim([0, len(categories)-1])
    else:
        ind = np.arange(len(categories))
        width = 0.9/len(queries)
        for i, q in enumerate(queries):
            plt.bar(ind + i*width, [results[q][cat] for cat in categories], width, color=colors[i], label=q)
        plt.xticks(ind + 0.45, cat_names, rotation=r)
        plt.xlim([-0.1, len(categories)])
    plt.ylabel('frequency')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
