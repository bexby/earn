import os
import torch
from datasets import Dataset
import json
import ipdb
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from fds import latex_similarity
from tqdm import tqdm

EPS = np.finfo(float).eps
random.seed(27)
kw_model = SentenceTransformer("all-distilroberta-v1", device="cuda")

def read_data(path):
    data_ls = []
    file_ls = os.listdir(path)
    for file in file_ls:
        with open(os.path.join(path, file), "r") as fr:
            lines = fr.readlines()
            for line in lines:
                data = json.loads(line)
                if data["text_keywords"] == "" or len(data["nor_expression"]) > 500:
                    continue
                data_ls.append(data)
    sample_list = random.sample(data_ls, 100000)
    return sample_list


def get_math_sim(item, data_list):
    result = []
    for i in data_list:
        result.append(latex_similarity(item, i))
    return np.array(result)


def get_keywords_sim(index, all_latex_embedding):
    org_emb = all_latex_embedding[index, :]
    sim = kw_model.similarity(org_emb, all_latex_embedding)
    sim = torch.squeeze(sim)
    return sim.numpy()


def entropy_weight(columns):
    d = []
    for column in columns:
        col = column / np.sum(column)
        col = np.where(col == 0, EPS, col)
        e = -1 / np.log(col.shape[0]) * np.sum(col * np.log(col))
        d.append(1 - e)
    res =  d / sum(d)
    res[-1] = max(min(res[-1], 0.2), 0.1)
    res[0] = 1 - res[-1]
    return res


def topk_p(score, k, index):
    topk = np.argpartition(score, -k)[-k:]
    if index in topk:
        topk = np.argpartition(score, -(k + 1))[-(k + 1):]
    wl = list(topk)
    try:
        wl.remove(index)
    except:
        pass

    weight = np.ndarray((k,))
    for i, ind in enumerate(wl):
        weight[i] = score[ind]
    result = random.choices(wl, weights=weight)
    return result[0]


def lowk_p(score, k):
    lowk = np.argpartition(score, k)[:k]
    wl = list(lowk)
    weight = np.ndarray((k,))
    for i, ind in enumerate(wl):
        weight[i] = 1 - score[ind]
    result = random.choices(wl, weights=weight)
    return result[0]


def process(data_list):
    result = {"anchor": [], "positive": [], "negative": []}
    math_list = []
    keywords_list = []
    for item in data_list:
        math_list.append(item["nor_expression"])
        keywords_list.append(item["text_keywords"])
    
    all_latex_embedding = kw_model.encode(data_list)

    for index, item in tqdm(enumerate(data_list), desc="process data"):
        math_similar = get_math_sim(item["nor_expression"], math_list)
        keywords_similar = get_keywords_sim(index, all_latex_embedding)
        keywords_similar = (keywords_similar + 1) / 2
        weight = entropy_weight([math_similar, keywords_similar])
        score = weight[0] * math_similar + weight[1] * keywords_similar
        positive_index = topk_p(score, 3, index)
        negative_index = lowk_p(score, 3)
        org_exp = item["nor_expression"]
        org_kw = item["text_keywords"]
        p_exp = data_list[positive_index]["nor_expression"]
        p_kw = data_list[positive_index]["text_keywords"]
        n_exp = data_list[negative_index]["nor_expression"]
        n_kw = data_list[negative_index]["text_keywords"]
        result["anchor"].append(f"expression: {org_exp} \nkeywords: {org_kw}")
        result["positive"].append(f"expression: {p_exp} \nkeywords: {p_kw}")
        result["negative"].append(f"expression: {n_exp} \nkeywords: {n_kw}")

    ds = Dataset.from_dict(result)
    ds.save_to_disk("training_data")


def main():
    math_file_path = r"C:\vscode_project\retrieval\math_info"
    data_list = read_data(math_file_path)
    process(data_list)


if __name__ == "__main__":
    with torch.no_grad():
        main()