from __future__ import unicode_literals, division, print_function, absolute_import
import sys
import numpy as np
from nlputils.dict_utils import invert_dict0, invert_dict2
from nlputils.features import FeatureTransform


def distinctive_fun_tpr(tpr, fpr):
    # to get the development of word occurrences
    return tpr


def distinctive_fun_diff(tpr, fpr):
    # computes the distinctive score as the difference between tpr and fpr rate (not below 0 though)
    return np.maximum(tpr - fpr, 0.)


def distinctive_fun_tprmean(tpr, fpr):
    # computes the distinctive score as the mean between the tpr and the difference between tpr and fpr rate
    return 0.5 * (tpr + np.maximum(tpr - fpr, 0.))


def distinctive_fun_tprmult(tpr, fpr):
    return tpr * np.maximum(tpr - fpr, 0.)


def distinctive_fun_quot(tpr, fpr):
    # return 1./(1.+np.exp(-tpr/np.maximum(fpr,sys.float_info.epsilon)))
    return (np.minimum(np.maximum(tpr / np.maximum(fpr, sys.float_info.epsilon), 1.), 4.) - 1) / 3.


def distinctive_fun_quotdiff(tpr, fpr):
    # return 1./(1.+np.exp(-tpr/np.maximum(fpr,sys.float_info.epsilon)))
    return 0.5 * (distinctive_fun_quot(tpr, fpr) + distinctive_fun_diff(tpr, fpr))


def get_distinctive_words(textdict, doccats, distinctive_fun=distinctive_fun_quotdiff):
    """
    For every category, find distinctive (i.e. `distinguishing') words by comparing how often the word each word
    occurs in this target category compared to all other categories.

    Input:
        - textdict: a dict with {docid: text}
        - doccats: a dict with {docid: cat} (to get trends in time, cat could also be a year/day/week)
        - distinctive_fun: which formula should be used when computing the score (default: distinctive_fun_quotdiff)
    Returns:
        - distinctive_words: a dict with {cat: {word: score}},
          i.e. for every category the words and a score indicating
          how relevant the word is for this category (the higher the better)
          you could then do sorted(distinctive_words[cat], key=distinctive_words[cat].get, reverse=True)[:10]
          to get the 10 most distinguishing words for that category
    """
    # transform all texts into sets of preprocessed words and bigrams
    print("computing features")
    ft = FeatureTransform(norm='max', weight=False, renorm=False, identify_bigrams=True, norm_num=False)
    docfeats = ft.texts2features(textdict)
    #docfeats = {did: set(docfeats[did].keys()) for did in docfeats}
    # invert this dict to get for every word the documents it occurs in
    # word_dids = {word: set(dids) for word, dids in invert_dict1(docfeats).items()}
    # invert the doccats dict to get for every category a list of documents belonging to it
    cats_dids = {cat: set(dids) for cat, dids in invert_dict0(doccats).items()}
    # get a list of all words
    word_list = list(invert_dict2(docfeats).keys())
    # count the true positives for every word and category
    print("computing tpr for all words and categories")
    tpc_words = {}
    for word in word_list:
        tpc_words[word] = {}
        for cat in cats_dids:
            # out of all docs in this category, in how many did the word occur?
            #tpc_words[word][cat] = len(cats_dids[cat].intersection(word_dids[word])) / len(cats_dids[cat])
            # average tf score in the category
            # (don't just take mean of the list comprehension otherwise you're missing zero counts)
            tpc_words[word][cat] = sum([docfeats[did][word] for did in cats_dids[cat] if word in docfeats[did]]) / len(cats_dids[cat])
    # for every category, compute a score for every word
    distinctive_words = {}
    for cat in cats_dids:
        print("computing distinctive words for category %r" % cat)
        distinctive_words[cat] = {}
        # compute a score for every word
        for word in word_list:
            # in how many of the target category documents the word occurs
            tpr = tpc_words[word][cat]
            if tpr:
                # in how many of the non-target category documents the word occurs (mean+std)
                fprs = [tpc_words[word][c] for c in cats_dids if not c == cat]
                fpr = np.mean(fprs) + np.std(fprs)
                # compute score
                distinctive_words[cat][word] = distinctive_fun(tpr, fpr)
    return distinctive_words


def test_distinctive_computations(distinctive_fun=distinctive_fun_diff, fun_name='Rate difference'):
    """
    given a function to compute the "distinctive score" of a word given its true and false positive rate,
    plot the distribution of scores (2D) corresponding to the different tpr and fpr
    """
    # make a grid of possible tpr and fpr combinations
    import matplotlib.pyplot as plt
    x, y = np.linspace(0, 1, 101), np.linspace(1, 0, 101)
    fpr, tpr = np.meshgrid(x, y)
    score = distinctive_fun(tpr, fpr)
    plt.figure()
    plt.imshow(score, cmap=plt.get_cmap('viridis'))
    plt.xlabel('FPR$_c(t_i)$')
    plt.ylabel('TPR$_c(t_i)$')
    plt.xticks(np.linspace(0, 101, 11), np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 101, 11), np.linspace(1, 0, 11))
    plt.title('Score using %s' % fun_name)
    plt.colorbar()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_distinctive_computations(distinctive_fun_tpr, 'TPR')
    test_distinctive_computations(distinctive_fun_diff, 'Rate Difference')
    test_distinctive_computations(distinctive_fun_tprmean, 'Mean of TPR and Rate Difference')
    test_distinctive_computations(distinctive_fun_tprmult, 'TPR weighted Rate Difference')
    test_distinctive_computations(distinctive_fun_quot, 'Rate Quotient')
    test_distinctive_computations(distinctive_fun_quotdiff, 'Mean of Rate Quotient and Difference')
    plt.show()
