import pymssql
import os
import json
import ipdb
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time

conn = pymssql.connect(server="DESKTOP-JC7DEJ3", host='',port=1433, user='sa', password='123', database='retrieval_math_expression')
cursor = conn.cursor()
model_path = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_path, device="cuda")

def get_infomation(expression, topk):
    org_embedding = model.encode(expression, convert_to_tensor=True, device="cuda")
    candidates = []
    cursor.execute(f"SELECT math_id, embedding, nor_expression, is_inline FROM math_info")
    ipdb.set_trace()
    while True:
        st = time.time()
        values = cursor.fetchmany(20000)
        mt = time.time()
        if not values:
            break
        temp_data = {"math_id": [], "embedding": [], "nor_expression": [], "is_inline": []}
        for item in values:
            temp_data["math_id"].append(item[0])
            embedding = json.loads(item[1])
            temp_data["embedding"].append(embedding)
            temp_data["nor_expression"].append(item[2])
            temp_data["is_inline"].append(item[3])

        m1t = time.time()
        embeddings = torch.tensor(temp_data["embedding"], device="cuda")
        similarities = model.similarity(org_embedding, embeddings)
        m2t = time.time()
        topk_values, indices = torch.topk(torch.squeeze(similarities), k=topk, sorted=True)
        for v, i in zip(topk_values, indices):
            candidates.append({"math_id": temp_data["math_id"][i], "nor_expression": temp_data["nor_expression"][i], "is_inline": temp_data["is_inline"][i], "similarity": v})
        et = time.time()
        print(f"fatch time:{mt - st}, {(mt - st)/(et - st)}%; for time:{m1t - mt}, {(m1t - mt)/(et - st)}%; similar time:{m2t - m1t}, {(m2t - m1t)/(et - st)}%; sort time:{et - m2t}, {(et - m2t)/(et - st)}%; ")
    
    out_sim = []
    result = []
    for item in candidates:
        out_sim.append(item["similarity"])
    out_topk, out_indices = torch.topk(torch.tensor(out_sim), k=topk, sorted=True)
    for v, i in zip(out_topk, out_indices):
        result.append({"math_id": candidates[i]["math_id"], "nor_expression": candidates[i]["nor_expression"], "is_inline": candidates[i]["is_inline"], "similarity": v})
    return result



if __name__ == "__main__":
    expression = r"a^2+b^2=c^2"
    res = get_infomation(expression, 20)
    for i in res:
        print(res)
    conn.close()