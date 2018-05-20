from __future__ import unicode_literals, division, print_function, absolute_import
import os
import requests
import matplotlib.pyplot as plt
from nlputils.dict_utils import invert_dict0
from textcatvis.visualize_relevantwords import visualize_tfidf, visualize_distinctive, visualize_clf
from textcatvis.cluster import cluster_texts
from textcatvis.check_query import *


def download_nytimes_archive(year, month):
    """
    Download articles from NYTimes for the given year and month

    Inputs:
        year, month: integers indicating the year and month for which to download articles
    Returns:
        textdict: a dict with {articleid: article text}
        doccats: a dict with {articleid: publication date}
    """
    # request an API key for the NYTimes Archive API: https://developer.nytimes.com/signup
    # and save it in a file called 'nytimes_apikey.txt'
    try:
        with open('nytimes_apikey.txt') as f:
            api_key = f.read().strip()
    except:
        print("Please request an API key for the NYTimes Archive API from https://developer.nytimes.com/signup ", end=' ')
        print("and save it in a file called 'nytimes_apikey.txt'")
    # download articles for given month and year
    url = "https://api.nytimes.com/svc/archive/v1/%i/%i.json" % (year, month)
    response = requests.get(url, params={"api-key": api_key}).json()['response']
    assert len(response['docs']) == response['meta']['hits'], "did not receive all articles..."
    textdict, doccats = {}, {}
    for i, article in enumerate(response['docs']):
        docid = "%i %s" % (i, article['pub_date'])
        textdict[docid] = "%s\n%s" % (article['headline']['main'], article['snippet'])
        doccats[docid] = article['pub_date'].split('T')[0]
    return textdict, doccats


def get_articles(date_begin, date_end):
    """
    download articles from NYTimes between date_begin and date_end (both inclusive)

    Inputs:
        date_begin, date_end: strings with dates in the format "%YYYY-%MM-%DD",
            e.g. '2017-01-22'
    Returns:
        textdict: a dict with {articleid: article text}
        doccats: a dict with {articleid: publication date}
    """
    year_begin, month_begin = int(date_begin.split('-')[0]), int(date_begin.split('-')[1])
    year_end, month_end = int(date_end.split('-')[0]), int(date_end.split('-')[1])
    # get articles from month_begin/year_begin until month_end/year_end
    textdict, doccats = {}, {}
    year, month = year_begin, month_begin
    while (year < year_end) or (year == year_end and month <= month_end):
        textdict_temp, doccats_temp = download_nytimes_archive(year, month)
        textdict.update(textdict_temp)
        doccats.update(doccats_temp)
        if month < 12:
            month += 1
        else:
            month = 1
            year += 1
    # select only articles in the given interval
    article_ids = [aid for aid in doccats if doccats[aid] >= date_begin and doccats[aid] <= date_end]
    textdict = {aid: textdict[aid] for aid in article_ids}
    doccats = {aid: doccats[aid] for aid in article_ids}
    return textdict, doccats


def split_articles(textdict, doccats, date_cut):
    """
    split given articles into "current" (after date_cut (inclusive)) and "old" articles (before date_cut)

    Inputs:
        textdict: a dict with {articleid: article text}
        doccats: a dict with {articleid: publication date}
        date_cut: strings with dates in the format "%YYYY-%MM-%DD", e.g. '2017-01-22'
    Returns:
        textdict: a dict with {articleid: article text}
        doccats: a dict with {articleid: 'current'/'old' depending on publication date}
    """
    # sort articles based on date_cut
    ids_current = [aid for aid in doccats if doccats[aid] >= date_cut]
    ids_old = [aid for aid in doccats if doccats[aid] < date_cut]
    textdict = {aid: textdict[aid] for aid in ids_old + ids_current}
    doccats = {aid: 'current' for aid in ids_current}
    doccats.update({aid: 'old' for aid in ids_old})
    return textdict, doccats


if __name__ == '__main__':
    cwd = os.getcwd()
    resdir = os.path.join(cwd, 'results')
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    for dname in ['nytimes_wc_tfidf', 'nytimes_wc_distinctive', 'nytimes_wc_clf', 'nytimes_wc_distinctive_clusters']:
        if not os.path.isdir(os.path.join(resdir, dname)):
            os.mkdir(os.path.join(resdir, dname))
    ### experiment 1: compare inauguration week and before
    # the articles we're interested in are from the week of trumps inauguration, i.e. 1-16 until 1-22
    # and the 3 weeks before this, i.e. 2016-12-26 until 2017-1-15
    textdict, doccats = get_articles('2016-12-26', '2017-01-22')
    textdict, doccats = split_articles(textdict, doccats, '2017-01-16')
    # visualize w/o html for tfidf (to get bigram word clouds)
    _ = visualize_tfidf(textdict, doccats, create_html=False, subdir_wc=os.path.join(resdir, 'nytimes_wc_tfidf'),
                        maskfiles={'current': os.path.join('maskimgs','up.png'), 'old': os.path.join('maskimgs','down.png')})
    # visualize distinctive
    _ = visualize_distinctive(textdict, doccats, subdir_wc=os.path.join(resdir, 'nytimes_wc_distinctive'),
                              maskfiles={'current': os.path.join('maskimgs','up.png'), 'old': os.path.join('maskimgs','down.png')})
    # visualize w/o html using clf (to get bigram word clouds)
    _ = visualize_clf(textdict, doccats, create_html=False, subdir_wc=os.path.join(resdir, 'nytimes_wc_clf'),
                      maskfiles={'current': os.path.join('maskimgs','up.png'), 'old': os.path.join('maskimgs','down.png')})
    # Accuracy: 0.713
    ### experiment 2: cluster articles from the inauguration week
    textdict, doccats = get_articles('2017-01-16', '2017-01-22')
    clusters = cluster_texts(textdict)
    cluster_docs = invert_dict0(clusters)
    _ = visualize_distinctive(textdict, clusters, subdir_wc=os.path.join(resdir, 'nytimes_wc_distinctive_clusters'))
    for c in sorted(cluster_docs, key=lambda x: len(cluster_docs[x]), reverse=True):
        print("#### %i documents in cluster %i" % (len(cluster_docs[c]), c))
        if not c == -1:
            for did in cluster_docs[c]:
                print(textdict[did].split("\n")[0])
    ### experiment 3: check the occurrences of some specific words
    textdict, doccats = get_articles('2016-12-26', '2017-01-22')
    queries = [check_or('and', 'or', 'the'), 'tuesday', 'trump', 'obama', check_and('italy', 'avalanche')]
    vis_occurrences(check_occurrences(textdict, doccats, queries), False)
    plt.show()
