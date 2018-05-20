from __future__ import unicode_literals, division, print_function, absolute_import
import os
import matplotlib.pyplot as plt
from datasets.cancer_papers.load_cancer import articles2dict
from textcatvis.visualize_relevantwords import visualize_tfidf, visualize_distinctive, visualize_clf
from textcatvis.check_query import *

if __name__ == '__main__':
    cwd = os.getcwd()
    resdir = os.path.join(cwd, 'results')
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    for dname in ['cancer_html_tfidf', 'cancer_wc_tfidf', 'cancer_wc_distinctive', 'cancer_html_clf', 'cancer_wc_clf',
                  'partype_html_tfidf', 'partype_wc_tfidf', 'partype_wc_distinctive', 'partype_html_clf', 'partype_wc_clf']:
        if not os.path.isdir(os.path.join(resdir, dname)):
            os.mkdir(os.path.join(resdir, dname))
    # do experiments for cancer type
    textdict, doccats, _ = articles2dict(label='keyword', reduced_labels=False, combine_paragraphs=False, ignore_types=['Mixed'])
    # visualize with html for tfidf
    _ = visualize_tfidf(textdict, doccats, create_html=True, subdir_html=os.path.join(resdir, 'cancer_html_tfidf'), subdir_wc=os.path.join(resdir, 'cancer_wc_tfidf'))
    # visualize w/o html for tfidf (to get bigram word clouds)
    _ = visualize_tfidf(textdict, doccats, create_html=False, subdir_wc=os.path.join(resdir, 'cancer_wc_tfidf'))
    # visualize distinctive
    _ = visualize_distinctive(textdict, doccats, subdir_wc=os.path.join(resdir, 'cancer_wc_distinctive'))
    # visualize with html using clf
    _ = visualize_clf(textdict, doccats, create_html=True, subdir_html=os.path.join(resdir, 'cancer_html_clf'), subdir_wc=os.path.join(resdir, 'cancer_wc_clf'))
    # F1 micro-avg: 0.950, F1 macro-avg: 0.949
    # Accuracy: 0.950
    # visualize w/o html using clf (to get bigram word clouds)
    _ = visualize_clf(textdict, doccats, create_html=False, subdir_wc=os.path.join(resdir, 'cancer_wc_clf'))
    # F1 micro-avg: 0.950, F1 macro-avg: 0.949
    # Accuracy: 0.950
    # check occurrences of different words per cancer type
    queries = ['brain', check_or('man', 'men'), check_or('woman', 'women')]
    vis_occurrences(check_occurrences(textdict, doccats, queries), True)
    # do experiments for partype
    textdict, doccats, _ = articles2dict(label='partype', reduced_labels=False, combine_paragraphs=False, ignore_types=['Mixed'])
    # visualize with html for tfidf
    _ = visualize_tfidf(textdict, doccats, create_html=True, subdir_html=os.path.join(resdir, 'partype_html_tfidf'), subdir_wc=os.path.join(resdir, 'partype_wc_tfidf'))
    # visualize w/o html for tfidf (to get bigram word clouds)
    _ = visualize_tfidf(textdict, doccats, create_html=False, subdir_wc=os.path.join(resdir, 'partype_wc_tfidf'))
    # visualize distinctive
    _ = visualize_distinctive(textdict, doccats, subdir_wc=os.path.join(resdir, 'partype_wc_distinctive'))
    # visualize with html using clf
    _ = visualize_clf(textdict, doccats, create_html=True, subdir_html=os.path.join(resdir, 'partype_html_clf'), subdir_wc=os.path.join(resdir, 'partype_wc_clf'))
    # F1 micro-avg: 0.898, F1 macro-avg: 0.901
    # Accuracy: 0.898
    # visualize w/o html using clf (to get bigram word clouds) - here we use optional maskimages to create the word clouds
    _ = visualize_clf(textdict, doccats, create_html=False, subdir_wc=os.path.join(resdir, 'partype_wc_clf'),
                      maskfiles={'Abstract': os.path.join('maskimgs','A.png'), 'Discussion': os.path.join('maskimgs','D.png'), 'Introduction': os.path.join('maskimgs','I.png'),
                                 'Methods': os.path.join('maskimgs','M.png'), 'Results': os.path.join('maskimgs','R.png')})
    # F1 micro-avg: 0.895, F1 macro-avg: 0.898
    # Accuracy: 0.895
    plt.show()
