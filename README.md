## READ ME

Faced with a collection of texts, sorted into the categories "C1"-"C23" and no idea what those could be? Got a dump of text documents and need to figure out what they are about and which of those you should have a closer look at?
Code is here to help!

This repository contains tools, which help in getting a quick overview of a text dataset by creating word clouds of the relevant words for each class or identified cluster as well as code to highlight these words in the individual texts, e.g. to better understand classifier decisions. Further details can be found in the corresponding paper (short and long).

### quick start
To **install**, download this repository and make sure the dependencies listed below are installed as well (or available somewhere in your pythonpath or in the same folder as this code).

If you have text data available as a collection of `.txt` files either in a single folder or in multiple folders in case of texts already sorted in different categories, you can call the script `analyze_relevantwords.py` with the path to the folder (or parent directory of multiple folders) to load this data and create word clouds for it.

### library components

**dependencies** (Python 2.7): numpy, scipy, matplotlib, sklearn, wordcloud, [nlputils](https://github.com/cod3licious/nlputils)

- `data_utils.py`: contains a function to load a text dataset (organized in a folder with subdirectories for each class containing .txt documents) in the form required by the other functions.
- `cluster.py`: contains a function to cluster a collection of text documents with the DBSCAN algorithm from sklearn.
- `check_query.py`: contains functions to formulate queries and check how often a term occurs in texts of a given category.
- `vis_utils.py`: contains functions to create the word clouds and highlight relevant words in individual texts.
- `distinctive_words.py`: code to examine a text dataset and identify "distinctive words" by comparing how often a word occurs in one category compared to all others.
- `visualize_relevantwords.py`: contains 3 functions to generate word clouds and highlight words in individual documents based on tf-idf features, distinctive words, as well as the classification scores obtained with a linear SVM.
- `analyze_relevantwords.py`: can be called with a path to a dataset to carry out the analysis for this dataset, i.e. create word clouds for different classes etc.

### experiments

- in `experiments_cancer.py`, the above mentioned tools are tested on the [cancer papers dataset](https://github.com/cod3licious/cancer_papers) to create the results reported in the paper.
- in `experiments_nytimes.py`, the above mentioned tools are tested on articles downloaded with the NYTimes API (make sure you have an API key stored in `nytimes_apikey.txt`)

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
