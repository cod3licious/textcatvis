from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import zip
import os
import random
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as logreg
import sklearn.metrics as skmet
from nlputils.features import FeatureTransform, features2mat
from nlputils.dict_utils import invert_dict0, combine_dicts
from .vis_utils import create_wordcloud, scores2html
from .distinctive_words import get_distinctive_words


def select_subset(textdict, doccats, visids=[]):
    """
    select a random subset of the dataset if it contains more than 10000 examples

    Input and Returns:
        textdict: dict with {doc_id: text}
        doccats: dict with {doc_id: category}
        visids: a subset of docids for which the html visualization should be created
    """
    docids = sorted(textdict.keys())  # sort for consistency across OS
    random.seed(42)
    random.shuffle(docids)
    # visualize up to 1000 documents
    if not len(visids):
        visids = docids[:1000]
    elif len(visids) > 1000:
        print("WARNING: creating visualizations for %i, i.e. more than 1000 documents can be slow!" % len(visids))
        if len(visids) > 10000:
            print("You don't know what you're doing....Truncating visids to 5000 examples.")
            visids = visids[:5000]
    # select subsets of examples to speed up the computations
    if len(docids) > 10000:
        # always make sure you end up with exactly 10k random examples (incl visids) but also don't shuffle a lot more than 10k ids
        docids = list(set(docids[:10000+len(visids)]).difference(set(visids)))
        random.shuffle(docids)
        docids = docids[:10000-len(visids)] + visids
        textdict = {d: textdict[d] for d in docids}
        doccats = {d: doccats[d] for d in docids}
    return textdict, doccats, visids


def visualize_tfidf(textdict, doccats, create_html=True, visids=[], subdir_html='', subdir_wc='', maskfiles={}):
    """
    visualize a text categorization dataset w.r.t. tf-idf features (create htmls with highlighted words and word clouds)

    Input:
        textdict: dict with {doc_id: text}
        doccats: dict with {doc_id: category}
        create_html: whether to create the html files with scores highlighted for individual documents (default: True)
        visids: a subset of docids for which the html visualization should be created (optional)
                (if create_html=True but visids=[], select up to 1000 random ids)
        subdir_html: subdirectory to save the created html files in (has to exist)
        subdir_wc: subdirectory to save the created word cloud images in (has to exist)
        maskfiles: dict with {category: path_to_maskfile} for creating the word clouds in a specific form
    Returns:
        relevant_words: dict with {category: {word: relevancy score}}
    """
    print("possibly selecting subset of 10000 examples")
    textdict, doccats, visids = select_subset(textdict, doccats, visids)
    print("transforming text into features")
    # we can identify bigrams if we don't have to create htmls
    ft = FeatureTransform(norm='max', weight=True, renorm='max', identify_bigrams=not create_html, norm_num=False)
    docfeats = ft.texts2features(textdict)
    # maybe highlight the tf-idf scores in the documents
    if create_html:
        print("creating htmls for %i of %i documents" % (len(visids), len(docfeats)))
        for i, did in enumerate(visids):
            if not i % 100:
                print("progress: at %i of %i documents" % (i, len(visids)))
            metainf = did + '\n' + 'True Class: %s\n' % doccats[did]
            name = did + '_' + doccats[did]
            scores2html(textdict[did], docfeats[did], os.path.join(subdir_html, name.replace(' ', '_').replace('/', '_')), metainf)
    # get a map for each category to the documents belonging to it
    catdocs = invert_dict0(doccats)
    # create word clouds for each category by summing up tfidf scores
    scores_collected = {}
    for cat in catdocs:
        print("creating word cloud for category %r with %i samples" % (cat, len(catdocs[cat])))
        scores_collected[cat] = {}
        for did in catdocs[cat]:
            scores_collected[cat] = combine_dicts(scores_collected[cat], docfeats[did], sum)
        # create word cloud
        create_wordcloud(scores_collected[cat], os.path.join(subdir_wc, "%s.png" % cat), maskfiles[cat] if cat in maskfiles else None)
    return scores_collected


def visualize_clf(textdict, doccats, create_html=True, visids=[], subdir_html='', subdir_wc='', maskfiles={}, use_logreg=False):
    """
    visualize a text categorization dataset w.r.t. classification scores (create htmls with highlighted words and word clouds)

    Input:
        textdict: dict with {doc_id: text}
        doccats: dict with {doc_id: category}
        create_html: whether to create the html files with scores highlighted for individual documents (default: True)
        visids: a subset of docids for which the html visualization should be created (optional)
                (if create_html=True but visids=[], select up to 1000 random ids)
        subdir_html: subdirectory to save the created html files in (has to exist)
        subdir_wc: subdirectory to save the created word cloud images in (has to exist)
        maskfiles: dict with {category: path_to_maskfile} for creating the word clouds in a specific form
        use_logreg: default False; whether to use logistic regression instead of linear SVM
    Returns:
        relevant_words: dict with {category: {word: relevancy score}}
    """
    print("possibly selecting subset of 10000 examples")
    textdict, doccats, visids = select_subset(textdict, doccats, visids)
    # training examples are all but visids
    trainids = list(set(textdict.keys()).difference(set(visids)))
    # train a classifier and predict
    if use_logreg:
        renorm = 'max'
        clf = logreg(class_weight='balanced', random_state=1)
    else:
        renorm = 'length'
        clf = LinearSVC(C=10., class_weight='balanced', random_state=1)
    print("transforming text into features")
    # make features (we can use bigrams if we don't have to create htmls)
    ft = FeatureTransform(norm='max', weight=True, renorm=renorm, identify_bigrams=not create_html, norm_num=False)
    docfeats = ft.texts2features(textdict, fit_ids=trainids)
    # convert training data to feature matrix
    featmat_train, featurenames = features2mat(docfeats, trainids)
    y_train = [doccats[tid] for tid in trainids]
    # fit classifier
    print("training classifier")
    clf.fit(featmat_train, y_train)
    del featmat_train
    # make test featmat and label vector
    print("making predictions")
    featmat_test, featurenames = features2mat(docfeats, visids, featurenames)
    # get actual classification results for all test samples
    predictions = clf.decision_function(featmat_test)
    predictions_labels = clf.predict(featmat_test)
    y_true, y_pred = [doccats[tid] for tid in visids], list(predictions_labels)
    # report classification accuracy
    if len(clf.classes_) > 2:
        f1_micro, f1_macro = skmet.f1_score(y_true, y_pred, average='micro'), skmet.f1_score(y_true, y_pred, average='macro')
        print("F1 micro-avg: %.3f, F1 macro-avg: %.3f" % (f1_micro, f1_macro))
    print("Accuracy: %.3f" % skmet.accuracy_score(y_true, y_pred))
    # create the visualizations
    print("creating the visualization for %i test examples" % len(visids))
    # collect all the accumulated scores to later create a wordcloud
    scores_collected = np.zeros((len(featurenames), len(clf.classes_)))
    # run through all test documents
    for i, tid in enumerate(visids):
        if not i % 100:
            print("progress: at %i of %i test examples" % (i, len(visids)))
        # transform the feature vector into a diagonal matrix
        feat_vec = lil_matrix((len(featurenames), len(featurenames)), dtype=float)
        feat_vec.setdiag(featmat_test[i, :].toarray().flatten())
        feat_vec = csr_matrix(feat_vec)
        # get the scores (i.e. before summing up)
        scores = clf.decision_function(feat_vec)
        # adapt for the intercept
        scores -= (1. - 1./len(featurenames)) * clf.intercept_
        # when creating the html visualization we want the words speaking for the prediction
        # but when creating the word cloud, we want the words speaking for the actual class
        metainf = tid + '\n'
        # binary or multi class?
        if len(scores.shape) == 1:
            if clf.classes_[0] == predictions_labels[i]:
                # we want the scores which speak for the class - for the negative class,
                # the sign needs to be reversed
                scores *= -1.
            scores_dict = dict(zip(featurenames, scores))
            metainf += 'True Class: %s\n' % doccats[tid]
            metainf += 'Predicted Class: %s  (Score: %.4f)' % (predictions_labels[i], predictions[i])
            scores_collected[:, clf.classes_ == doccats[tid]] += np.array([scores]).T
        else:
            scores_dict = dict(zip(featurenames, scores[:, clf.classes_ == predictions_labels[i]][:, 0]))
            metainf += 'True Class: %s  (Score: %.4f)\n' % (doccats[tid], predictions[i, clf.classes_ == doccats[tid]][0])
            metainf += 'Predicted Class: %s  (Score: %.4f)' % (predictions_labels[i], predictions[i, clf.classes_ == predictions_labels[i]][0])
            scores_collected[:, clf.classes_ == doccats[tid]] += scores[:, clf.classes_ == doccats[tid]]
        # use the vector with scores together with the corresponding feature names and the original text
        # to create the pretty visualization
        if create_html:
            if y_true[i] == y_pred[i]:
                name = 'correct_'
            else:
                name = 'error_'
            name += tid + '_' + doccats[tid]
            scores2html(textdict[tid], scores_dict, os.path.join(subdir_html, name.replace(' ', '_').replace('/', '_')), metainf)
    print("creating word clouds")
    # normalize the scores for each class
    scores_collected /= np.max(np.abs(scores_collected), axis=0)
    # transform the collected scores into a dictionary and create word clouds
    scores_collected_dict = {cat: dict(zip(featurenames, scores_collected[:, clf.classes_ == cat][:, 0])) for cat in clf.classes_}
    for cat in scores_collected_dict:
        create_wordcloud(scores_collected_dict[cat], os.path.join(subdir_wc, "%s.png" % cat), maskfiles[cat] if cat in maskfiles else None)
    return scores_collected_dict


def visualize_distinctive(textdict, doccats, subdir_wc='', maskfiles={}):
    """
    visualize a text categorization dataset by creating word clouds of `distinctive' words

    Input:
        textdict: dict with {doc_id: text}
        doccats: dict with {doc_id: category}
        subdir_wc: subdirectory to save the created word cloud images in (has to exist)
        maskfiles: dict with {category: path_to_maskfile} for creating the word clouds in a specific form
    Returns:
        relevant_words: dict with {category: {word: relevancy score}}
    """
    print("possibly selecting subset of 10000 examples")
    textdict, doccats, _ = select_subset(textdict, doccats, {})
    print("get 'distinctive' words")
    # this contains a dict for every category with {word: trend_score_for_this_category}
    distinctive_words = get_distinctive_words(textdict, doccats)
    # create the corresponding word clouds
    print("creating word clouds")
    for cat in distinctive_words:
        create_wordcloud(distinctive_words[cat], os.path.join(subdir_wc, "%s.png" % cat), maskfiles[cat] if cat in maskfiles else None)
    return distinctive_words
