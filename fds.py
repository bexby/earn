import Levenshtein

def latex_similarity(latex1, latex2):
    edit_distance = Levenshtein.distance(latex1, latex2)
    max_len = max(len(latex1), len(latex2))
    similarity = 1 - (edit_distance / max_len)
    return similarity
