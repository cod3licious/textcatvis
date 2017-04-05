## READ ME

Faced with a collection of texts, sorted into the categories "C1"-"C23" and no idea what those could be? Got a dump of text documents and need to figure out what they are about and which of those you should have a closer look at?
Code is here to help!

This repository contains tools, which help in getting a quick overview of a text categorization dataset by creating word clouds of the relevant words for each class as well as code to highlight these words in the individual texts, e.g. to better understand classifier decisions. Further details can be found in the corresponding paper.

### library components

**dependencies:** numpy, scipy, matplotlib, sklearn, wordcloud, [nlputils](https://github.com/cod3licious/nlputils)

- `data_utils.py`: contains a function to load a text dataset (organized in a folder with subdirectories for each class containing .txt documents) in the form required by the other functions.
- `cluster.py`: contains a function to cluster a collection of text documents with the DBSCAN algorithm from sklearn.
- `check_query.py`: contains functions to formulate queries and check how often a term occurs in texts of a given category.
- `vis_utils.py`: contains functions to create the word clouds and highlight relevant words in individual texts.
- `distinctive_words.py`: code to examine a text dataset and identify "distinctive words" by comparing how often a word occurs in one category compared to all others.
- `visualize_relevantwords.py`: contains 3 functions to generate word clouds and highlight words in individual documents based on tf-idf features, distinctive words, as well as the classification scores obtained with a linear SVM.
- `analyze_relevantwords.py`: can be called with a path to a dataset to carry out the analysis for this dataset, i.e. create word clouds for different classes etc.

### experiments

- in `experiments_cancer.py`, the above mentioned tools are tested on the [cancer papers dataset](https://github.com/cod3licious/cancer_papers) to create the results reported in the paper.
- in `experiments_nytimes.py`, the above mentioned tools are tested on articles downloaded with the NYTimes API
- `dbscan_experiments.ipynb` contains some tests on how to best cluster texts documents with the DBSCAN algorithm

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
