textcatvis
==========

Faced with a collection of texts, sorted into the categories "C1"-"C23" and no idea what those could be? Got a dump of text documents and need to figure out what they are about and which of those you should have a closer look at?
Code is here to help!

This repository contains tools, which help in getting a quick overview of a text dataset by creating word clouds of the relevant words for each class or identified cluster as well as code to highlight these words in the individual texts, e.g. to better understand classifier decisions. Further details can be found in the corresponding paper (short_ and long_).

If any of this code was helpful for your research, please consider citing it: ::

    @article{horn2017exploring,
      title     = {Exploring text datasets by visualizing relevant words},
      author    = {Horn, Franziska and Arras, Leila and Montavon, Gr{\'e}goire and M{\"u}ller, Klaus-Robert and Samek, Wojciech},
      journal   = {arXiv preprint arXiv:1707.05261},
      year      = {2017}
    }


or ::

    @article{horn2017discovering,
      title     = {Discovering topics in text datasets by visualizing relevant words},
      author    = {Horn, Franziska and Arras, Leila and Montavon, Gr{\'e}goire and M{\"u}ller, Klaus-Robert and Samek, Wojciech},
      journal   = {arXiv preprint arXiv:1707.06100},
      year      = {2017}
    }

.. _short: http://arxiv.org/abs/1707.06100
.. _long: http://arxiv.org/abs/1707.05261


The code is intended for research purposes. It was programmed for Python 2.7, but should theoretically also run on newer Python 3 versions - no guarantees on this though (open an issue if you find a bug, please)!

quick start
-----------
To install, either download the code from here and include the textcatvis folder in your ``$PYTHONPATH`` or install (the library components only) via pip:

    ``$ pip install textcatvis``


If you have text data available as a collection of ``.txt`` files either in a single folder or in multiple folders (in case of texts already sorted in different categories), you can call the script ``analyze_relevantwords.py`` with the path to the folder (or parent directory of multiple folders) to load this data and create word clouds for it.

textcatvis library components
-----------------------------

dependencies: numpy, scipy, matplotlib, sklearn, wordcloud, nlputils_

.. _nlputils: https://github.com/cod3licious/nlputils

- ``data_utils.py``: contains a function to load a text dataset (organized in a folder with subdirectories for each class containing .txt documents) in the form required by the other functions.
- ``cluster.py``: contains a function to cluster a collection of text documents with the DBSCAN algorithm from sklearn.
- ``check_query.py``: contains functions to formulate queries and check how often a term occurs in texts of a given category.
- ``vis_utils.py``: contains functions to create the word clouds and highlight relevant words in individual texts.
- ``distinctive_words.py``: contains code to examine a text dataset and identify "distinctive words" by comparing how often a word occurs in one category compared to all others.
- ``visualize_relevantwords.py``: contains 3 functions to generate word clouds and highlight words in individual documents based on tf-idf features, distinctive words, as well as the classification scores obtained with a linear SVM.

examples
--------

- ``analyze_relevantwords.py``: can be called with a path to a dataset to carry out the analysis for this dataset, i.e. create word clouds for different classes etc.
- in ``experiments_cancer.py``, the above mentioned tools are tested on the `cancer papers dataset`_ to create the results reported in the paper. (You need to download this dataset first.)
- in ``experiments_nytimes.py``, the above mentioned tools are tested on articles downloaded with the NYTimes API. (Make sure you have an API key stored in ``nytimes_apikey.txt``.)

.. _`cancer papers dataset`: https://github.com/cod3licious/cancer_papers

If you have any questions please don't hesitate to send me an `email <mailto:cod3licious@gmail.com>`_ and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
