import pickle
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

def f_casefold(arg_text: str) -> str:
    """
    @desc: this function is lowering raw text

    @params:
        - arg_text: raw string that need to be casefolded (str)

    @returns:
        - arg_text: casfolded string (str)
    """
    arg_text = arg_text.casefold().strip()
    return arg_text


def f_text_cleansing(arg_str: str) -> str:
    """
    @desc: this function is cleaning raw text that is already tokenized

    @params:
        - arg_text: raw string that need to be cleansed (str)

    @returns:
        - cleaned_text: cleaned string (str)
    """
    if isinstance(arg_str, str):
        # cleanse hashtag and mention
        # cleanse mention hashtag
        arg_str = re.sub(r'(?:#|@)[A-Za-z0-9_]+', '', arg_str)
        arg_str = re.sub(r'(?:https?|www)\S+', '', arg_str)  # cleanse url
        cleaned_text = ''
        for each_letter in arg_str:
            if each_letter.isalpha() or each_letter == ' ':
                cleaned_text += each_letter
        return cleaned_text
    else:
        raise Exception("Only string are Allowed")


def f_tokenizing(arg_text: str) -> list:
    """
    @desc: this function is splitting string into list by it's own space

    @params:
        - arg_text: raw string that need to be splitted (str)

    @returns:
        - out_list: splitted text (list)
    """
    # Transform jadi list, dan split tiap kata berdasarkan space
    if isinstance(arg_text, str):
        out_list = arg_text.split()
        return out_list
    else:
        raise Exception("Only string are Allowed")


words_filter = stopwords.words('indonesian', 'english')
# Pake stopwords dari library nltk buat hilangin kata hubung such as "sangat", " yang", "ketika" dll


def f_stopword(arg_list: list) -> list:
    """
    @desc: this function is filtering string inside a list that not in stopword library

    @params:
        - arg_list: list of string (list)

    @returns:
        - stopped_list: filtered list of string (list)
    """
    if isinstance(arg_list, list):
        stopped_list = list(filter(lambda word: (
            word not in words_filter) and (len(word) > 1), arg_list))
        return stopped_list
    else:
        raise Exception("Only List are Allowed")


# Stemming pake library sastrawi buat menghilangkan imbuhan, such as "mendegar" jadi "dengar"
# "berlari" jadi "lari"
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def f_stemming(arg_list: list) -> str:
    """
    @desc: this function is cleaning string inside a list using sastrawi

    @params:
        - arg_list: list of string (list)

    @returns:
        - d_clean: sastrawi-ed string (string)
    """
    cleaned = []
    for i in arg_list:
        cleaned.append(stemmer.stem(i))
    d_clean = " ".join(cleaned)
    return d_clean


# Load Model
filename = r'{}\finalized_model.sav'.format(os.getcwd())
loaded_model = pickle.load(open(filename, 'rb'))


def get_sentiment(arg_string):
    out_sent = f_casefold(arg_string)
    out_sent = f_text_cleansing(out_sent)
    out_sent = f_tokenizing(out_sent)
    out_sent = f_stopword(out_sent)
    out_sent = f_stemming(out_sent)
    return loaded_model.predict([out_sent])[0]
    