from __future__ import unicode_literals, division, print_function, absolute_import
import codecs
import re
from PIL import Image
import numpy as np
import matplotlib
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nlputils.features import preprocess_text

word_scores_dict = None
# colormaps for positive and negative scores
cmap_pos = get_cmap('Greens')
cmap_neg = get_cmap('Reds')
norm_pos = matplotlib.colors.Normalize(0., 1.)
norm_neg = matplotlib.colors.Normalize(0., 1.)


def posneg_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    """
    Generates colors based on the word's positive/negative score

    Parameters
    ----------
    word: the word in question, used to pick the score from the global word_scores_dict
    all other parameters are ignored
    """
    global word_scores_dict
    score = word_scores_dict[word]
    if score < 0:
        rgbc = cmap_neg(norm_neg(-score))
    else:
        rgbc = cmap_pos(norm_pos(score))
    return "rgb(%d, %d, %d)" % (round(255 * rgbc[0]), round(255 * rgbc[1]), round(255 * rgbc[2]))


def create_wordcloud(ws_dict, fname=None, maskfile=None):
    """
    given a dictionary with words and their relevancy scores, visualize the resulting wordcloud

    Inputs:
        - ws_dict: dictionary with {word:score}, where the score can be positive or negative
        - fname: file name where the resulting wordcloud should be saved
        - maskfile: filename where the shape of the wordcloud can be loaded (else it will be a rectangle)
    """
    if maskfile:
        # read the mask image - make sure it's black and white (not black and transparent)
        maskimg = np.array(Image.open(maskfile))
        height, width, _ = maskimg.shape
    else:
        maskimg = None
        width, height = 900, 600

    # check how many positive and negative words we have in the dict
    n_pos = len([1 for w in ws_dict if ws_dict[w] > 0])
    n_neg = len([1 for w in ws_dict if ws_dict[w] < 0])
    # get the 160 most positive words + 40 negative words
    relwords_pos = sorted(ws_dict, key=ws_dict.get, reverse=True)[:min(160, n_pos)]
    relwords_neg = sorted(ws_dict, key=ws_dict.get)[:min(40, n_neg)]
    words_freq = {w: abs(ws_dict[w]) for w in relwords_pos + relwords_neg}
    # set the ws_dict to the global dict - we need this for coloring
    global word_scores_dict
    word_scores_dict = ws_dict

    # normalize the color scales so you actually see some colors ;)
    global norm_pos, norm_neg
    if n_pos:
        norm_pos = matplotlib.colors.Normalize(
            2 * ws_dict[relwords_pos[-1]] - ws_dict[relwords_pos[0]], ws_dict[relwords_pos[0]])
    if n_neg:
        norm_neg = matplotlib.colors.Normalize(
            2 * abs(ws_dict[relwords_neg[-1]]) - abs(ws_dict[relwords_neg[0]]), abs(ws_dict[relwords_neg[0]]))

    # generate wordcloud from the given scores
    wc = WordCloud(background_color="white", max_words=len(words_freq), width=width,
                   height=height, mask=maskimg, color_func=posneg_color_func)
    wc.generate_from_frequencies(words_freq)

    # store to file
    if fname:
        wc.to_file(fname)

    # show
    plt.figure()
    plt.imshow(wc)
    plt.axis("off")


def scores2html(text, scores, fname='testfile', metainf='', highlight_oov=False):
    """
    Based on the original text and relevance scores, generate a html doc highlighting positive / negative words

    Inputs:
        - text: the raw text in which the words should be highlighted
        - scores: a dictionary with {word: score} or a list with tuples [(word, score)]
        - fname: the name (path) of the file
        - metainf: an optional string which will be added at the top of the file (e.g. true class of the document)
        - highlight_oov: if True, out-of-vocabulary words will be highlighted in yellow (default False)
    Saves the visualization in 'fname.html' (you probably want to make this a whole path to not clutter your main directory...)
    """
    # colormaps
    cmap_pos = get_cmap('Greens')
    cmap_neg = get_cmap('Reds')
    norm = matplotlib.colors.Normalize(0., 1.)

    # if not isinstance(text, unicode):
    #     text = text.decode("utf-8")

    # normalize score by absolute max value
    if isinstance(scores, dict):
        N = np.max(np.abs(list(scores.values())))
        scores_dict = {word: scores[word] / N for word in scores}
        # transform dict into word list with scores
        scores = []
        for word in re.findall(r'[\w-]+', text, re.UNICODE):
            word_pp = preprocess_text(word, norm_num=False)
            if word_pp in scores_dict:
                scores.append((word, scores_dict[word_pp]))
            else:
                scores.append((word, None))
    else:
        N = np.max(np.abs([t[1] for t in scores if t[1] is not None]))
        scores = [(w, s / N) if s is not None else (w, None) for w, s in scores]

    htmlstr = u'<body><div style="white-space: pre-wrap; font-family: monospace;">'
    if metainf:
        htmlstr += '%s\n\n' % metainf
    resttext = text
    for word, score in scores:
        # was anything before the identified word? add it unchanged to the html
        htmlstr += resttext[:resttext.find(word)]
        # cut off the identified word
        resttext = resttext[resttext.find(word) + len(word):]
        # get the colorcode of the word
        rgbac = (1., 1., 0.)  # for unknown words
        if highlight_oov:
            alpha = 0.3
        else:
            alpha = 0.
        if score is not None:
            if score < 0:
                rgbac = cmap_neg(norm(-score))
            else:
                rgbac = cmap_pos(norm(score))
            alpha = 0.5
        htmlstr += u'<span style="background-color: rgba(%i, %i, %i, %.1f)">%s</span>'\
            % (round(255 * rgbac[0]), round(255 * rgbac[1]), round(255 * rgbac[2]), alpha, word)
    # after the last word, add the rest of the text
    htmlstr += resttext
    htmlstr += u'</div></body>'
    with codecs.open('%s.html' % fname, 'w', encoding='utf8') as f:
        f.write(htmlstr)
