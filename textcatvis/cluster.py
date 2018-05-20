from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import KernelPCA
from sklearn.cluster import DBSCAN
from nlputils.features import FeatureTransform, features2mat


def cluster_texts(textdict, eps=0.45, min_samples=3):
    """
    cluster the given texts

    Input:
        textdict: dictionary with {docid: text}
    Returns:
        doccats: dictionary with {docid: cluster_id}
    """
    doc_ids = list(textdict.keys())
    # transform texts into length normalized kpca features
    ft = FeatureTransform(norm='max', weight=True, renorm='length', norm_num=False)
    docfeats = ft.texts2features(textdict)
    X, featurenames = features2mat(docfeats, doc_ids)
    e_lkpca = KernelPCA(n_components=250, kernel='linear')
    X = e_lkpca.fit_transform(X)
    xnorm = np.linalg.norm(X, axis=1)
    X = X/xnorm.reshape(X.shape[0], 1)
    # compute cosine similarity
    D = 1. - linear_kernel(X)
    # and cluster with dbscan
    clst = DBSCAN(eps=eps, metric='precomputed', min_samples=min_samples)
    y_pred = clst.fit_predict(D)
    return {did: y_pred[i] for i, did in enumerate(doc_ids)}
