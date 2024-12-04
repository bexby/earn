import os
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import ipdb
import pickle
import re
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"C:\vscode_project\retrieval\models\save"
embedding_path = r"C:\vscode_project\retrieval\embedding_tensor"
kw_model = KeyBERT()
model = SentenceTransformer(model_path, device="cuda")


def get_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2))
    result = []
    for kw in keywords:
        if kw == []:
            result.append("")
        else:
            result.append(kw[0])
    return result[0]    

def initial_selection(kw_exp, topk):
    with open(os.path.join(embedding_path, "math_id_list.pkl"), "rb") as fr:
        math_id_list = pickle.load(fr)
    print(len(math_id_list))
    # ipdb.set_trace()
    org_embedding = model.encode(kw_exp, convert_to_tensor=True, device=device)
    candidates = []
    previous_index = 0
    file_ls = os.listdir(embedding_path)[:-1]
    sorted_file_list = sorted(file_ls, key=lambda x: int(re.search(r'\d+', x).group()))
    for file in tqdm(sorted_file_list, desc="load embeddings"):
        embeddings = torch.load(os.path.join(embedding_path, file))
        embeddings = embeddings.to(device)
        similarities = model.similarity(org_embedding, embeddings)
        topk_values, indices = torch.topk(torch.squeeze(similarities), k=topk, sorted=True)
        for v, i in zip(topk_values, indices):
            candidates.append({"math_id": math_id_list[i + previous_index], "similarity": v})
        previous_index += embeddings.shape[0]
    out_sim = []
    result = []
    for item in candidates:
        out_sim.append(item["similarity"])
    out_topk, out_indices = torch.topk(torch.tensor(out_sim), k=topk, sorted=True)
    for v, i in zip(out_topk, out_indices):
        result.append({"math_id": candidates[i]["math_id"], "similarity": v})
    return result



def main():

    query_expression = r"a^2+b^2=c^2"
    expression_describtion = "this is about math"
    query_text = ""

    exp_keywords = get_keywords(expression_describtion)
    print(exp_keywords)
    kw_exp = "expression:" + query_expression + "\nkeywords:" + exp_keywords
    # ipdb.set_trace()
    res = initial_selection(kw_exp, 20)
    print(res)
    
    

if __name__ == "__main__":
    main()