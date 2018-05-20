from __future__ import unicode_literals, division, print_function, absolute_import
import sys
import os
import matplotlib.pyplot as plt
from textcatvis.data_utils import load_data
from textcatvis.visualize_relevantwords import visualize_tfidf, visualize_distinctive, visualize_clf
from textcatvis.cluster import cluster_texts
from textcatvis.check_query import *

if __name__ == '__main__':
    # call function with path to dataset
    if len(sys.argv) != 2:
        print("Call this script with the absolute path to a dataset, i.e. '$ python analyze_relevantwords.py /absolute/path/to/dataset'")
        sys.exit()
    path_to_data = sys.argv[1]
    # create folders for results if they don't already exist
    dataset = os.path.basename(os.path.normpath(path_to_data))
    if not os.path.isdir('results'):
        os.mkdir('results')
    for method in ['tfidf', 'distinctive', 'clf']:
        if not os.path.isdir(os.path.join('results', '%s_wc_%s' % (dataset, method))):
            os.mkdir(os.path.join('results', '%s_wc_%s' % (dataset, method)))
    # load data
    print("loading data for dataset %s" % dataset)
    textdict, doccats = load_data(path_to_data)
    # if we only have a single category, we need to cluster the texts
    use_clf = True
    if len(set(doccats.values())) == 1:
        print("clustering", end=' ')
        doccats = cluster_texts(textdict)
        print(" - got %i clusters + %i samples considered noise" % (len(set(doccats.values()))-1, len([1 for i in doccats.values() if i == -1])))
        use_clf = False
    print("creating word clouds")
    # visualize w/o html for tfidf (to get bigram word clouds)
    scores_tfidf = visualize_tfidf(textdict, doccats, create_html=False, subdir_wc=os.path.join('results', '%s_wc_tfidf' % dataset))
    # visualize distinctive
    scores_distinctive = visualize_distinctive(textdict, doccats, subdir_wc=os.path.join('results', '%s_wc_distinctive' % dataset))
    # visualize w/o html using clf (to get bigram word clouds)
    if use_clf:
        # only classify if we had actual classes
        scores_clf = visualize_clf(textdict, doccats, create_html=False, subdir_wc=os.path.join('results', '%s_wc_clf' % dataset))
    print("checking example queries")
    # identify fraction of articles per category containing...
    # any stop words; mentioning the current AND former president; containing the word brain
    queries = [check_or('and', 'or', 'the'), check_and('trump', 'obama'), 'brain']
    vis_occurrences(check_occurrences(textdict, doccats, queries), len(set(doccats.values())) <= 10)
    plt.show()
