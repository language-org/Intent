# author: Steeve LAQUITAINE

import re

# either ? or ! or .
SENT_TYPE_PATTN = re.compile(r"[\?\!\.]")


def classify_mood(sentences):
    """
    Classify sentence type
    args:
        sentences (pd.DataFrame):
            
    """
    sent_type = []
    for sent in sentences:
        out = SENT_TYPE_PATTN.findall(sent)
        sent_type.append(
            [
                "ask"
                if ix == "?"
                else "wish-or-excl"
                if ix == "!"
                else "state"
                for ix in out
            ]
        )
    return sent_type


def detect_sentence_type(df, sent_type: str):
    """
    Detect sentence types

    parameters
    ----------
    sent_type: str
        'state', 'ask', 'wish-excl' 
    """
    return sent_type in df


def del_null(dictionary):
    """Recursively delete Null keys
    Args:
        dictionary (Dict): dictionary
    """
    for key, value in list(dictionary.items()):
        if value is None:
            del dictionary[key]
        elif isinstance(value, dict):
            del_null(value)
    return dictionary
