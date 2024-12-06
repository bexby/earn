from sentence_transformers import SentenceTransformer
import ipdb
import json
import os
from tqdm import tqdm

data_path = r"C:\vscode_project\retrieval\math_info"
model_path = r"C:\vscode_project\retrieval\models\save"
save_path = r"C:\vscode_project\retrieval\embeddings"
model = SentenceTransformer(model_path, device="cuda")

file_ls = os.listdir(data_path)
for file in tqdm(file_ls, desc=""):
    dl = []
    latex = []
    result = []
    with open(os.path.join(data_path, file), "r") as fr:
        lines = fr.readlines()
        for line in lines:
            data = json.loads(line)
            dl.append(data)
            latex.append("expression:" + data["nor_expression"] + "\nkeywords:" + data["text_keywords"])
    embeddings = model.encode(latex)
    ls_emb = embeddings.tolist()
    with open(os.path.join(save_path, file[:10] + ".jsonl"), "a") as fw:
        for i in range(len(dl)):
            fw.write(json.dumps({"math_id": dl[i]["math_id"], "embedding": ls_emb[i]}))
            fw.write("\n")