from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import next
import os
from glob import iglob


def load_data(path):
    """
    This is a utility function to load a text categorization dataset.
    It assumes the data is organized in the folder supplied in the path argument with different
    folders for each class, where each folder contains individual text documents (.txt).
    Alternatively, unlabeled data can also just be in the current folder and will receive the class label '.'.
    The function returns two dictionaries, one with the raw texts, one with the corresponding classes (= subdirectory names).
    The document ids used to index both dictionaries and match raw texts with categories are constructed as
    classname + name of text file.

    Input:
        path: path to a folder with the data
    Returns:
        textdict: dict with {doc_id: text}
        doccats: dict with {doc_id: category}
    """
    textdict = {}
    doccats = {}
    # if there are unlabeled documents in the current directory
    for fname in iglob(os.path.join(path, '*.txt')):
        # construct unique docid
        docid = os.path.splitext(os.path.basename(fname))[0]
        # save category + text
        doccats[docid] = '.'
        with open(fname) as f:
            textdict[docid] = f.read()
    # go through all category subdirectories
    for cat in next(os.walk(path))[1]:
        if not cat.startswith('.'):
            cat_path = os.path.join(path, cat)
            # go through all txt documents
            for fname in iglob(os.path.join(cat_path, '*.txt')):
                # construct unique docid
                docid = cat + ' ' + os.path.splitext(os.path.basename(fname))[0]
                # save category + text
                doccats[docid] = cat
                with open(fname) as f:
                    textdict[docid] = f.read()
    return textdict, doccats
