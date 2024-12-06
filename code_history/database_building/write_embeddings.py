import pymssql
import os
import json
import ipdb
from tqdm import tqdm
import torch
import pickle

conn = pymssql.connect(server="DESKTOP-JC7DEJ3", host='',port=1433, user='sa', password='123', database='retrieval_math_expression')
cursor = conn.cursor()
embeddings_folder = r"C:\Users\86159\Downloads\embeddings"
save_path = r"C:\vscode_project\retrieval\embedding_tensor"

def write_math_info(path):
    embeddings_list = os.listdir(path)
    for file in tqdm(embeddings_list, desc="imported files"):
        data = []
        with open(os.path.join(path, file), "r") as fr:
            lines = fr.readlines()
            for line in lines:
                d = json.loads(line)
                data.append(d)
        for item in data:
            i = repr(item["math_id"])
            e = item["embedding"]
            e = "\'" + repr(e) + "\'"
            # ipdb.set_trace()
            cursor.execute(f"UPDATE math_info SET embedding = {e} WHERE math_id = {i}")
        cursor.connection.commit()

def write_tensor(path, save_path):
    math_id_ls = []
    part_index = 1
    embeddings_list = os.listdir(path)
    embeddings = []
    for file in tqdm(embeddings_list, desc="imported files"):
        with open(os.path.join(path, file), "r") as fr:
            lines = fr.readlines()
            for line in lines:
                d = json.loads(line)
                embeddings.append(d["embedding"])
                math_id_ls.append(d["math_id"])
        if len(embeddings) > 80000:
            tensor_embeddings = torch.tensor(embeddings)
            torch.save(tensor_embeddings, os.path.join(save_path, "embeddings_part" + str(part_index) + ".pt"))
            embeddings = []
            part_index += 1
    
    with open(os.path.join(save_path, "math_id_list.pkl"), "wb") as fw:
        pickle.dump(math_id_ls, fw)

if __name__ == "__main__":
    write_math_info(embeddings_folder)
    conn.close()
    write_tensor(embeddings_folder, save_path)