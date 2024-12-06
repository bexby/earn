import os
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import ipdb
import pickle
import re
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import numpy as np
import pymssql
import textwrap


latex_format_symbols = {
    # 格式符
    '\\textbf', '\\textit', '\\texttt', '\\emph', '\\underline', '\\overline', '\\overbrace', '\\underbrace', '\\left', '\\right',
    '\\big', '\\Big', '\\bigg', '\\Bigg', '\\textit', '\\textsf', '\\textrm', '\\mathcal', '\\mathbb', '\\mathfrak', '\\mathscr', '\\displaystyle', '\\textrm',
}

print("正在连接数据库")
conn = pymssql.connect(server="DESKTOP-JC7DEJ3", host='',port=1433, user='sa', password='123', database='retrieval_math_expression')
cursor = conn.cursor()
print("数据库连接成功")

print("正在加载模型")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"C:\vscode_project\retrieval\models\save"
embedding_path = r"C:\vscode_project\retrieval\embedding_tensor"
kw_model = KeyBERT()
model = SentenceTransformer(model_path, device=device)
topic_model = CrossEncoder("jinaai/jina-reranker-v1-tiny-en", trust_remote_code=True)
print("模型加载成功")


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

def AHP():
    items = ["公式相似度", "主题相关度", "文献重要程度", "论文的时效性", "行内行间公式的影响程度"]
    use_cache = input("是否使用上次的权重(yes or no): ")
    if use_cache == "yes":
        weights = np.load("./AHP_cache/weights.npy")
    else:
        element = {"标度": ["1", "3", "5", "7", "9", "2,4,6,8", "倒数"], "含义": ["同样重要", "稍微重要", "明显重要", "强烈重要", "极端重要", "上述俩相邻判断的中值", "如果A比B的标度是3, 则B比A的标度就是1/3"]}
        print(tabulate(element, headers='keys', tablefmt='psql', showindex=False))
        print("请根据上表填写你的判断")
        aline_matrix = np.eye(5)
        for i in range(5):
            for j in range(i + 1, 5):
                w = input(f"请判断\"{items[i]}\"对\"{items[j]}\"的重要性： ")
                if "/" in w:
                    aline_matrix[i][j] = 1 / int(w[-1])
                    aline_matrix[j][i] = int(w[-1])
                else:
                    aline_matrix[i][j] = int(w[0])
                    aline_matrix[j][i] = 1 / int(w[0])
        eigenvalues, eigenvectors = np.linalg.eig(aline_matrix)
        max_eig = np.abs(eigenvalues[0])
        CI = (max_eig - 5) / 4
        RI = 1.12
        CR = CI / RI
        if CR > 0.1:
            print("一致性检验不通过，请重新开始判断")
            return None
        else:
            weights = np.abs(eigenvectors[:, 0]) / np.sum(np.abs(eigenvectors[:, 0]))
            np.save("./AHP_cache/weights.npy", weights)
    
    print("所计算的权重为：")
    output_w = {k: [v] for k, v in zip(items, weights)}
    print(tabulate(output_w, headers='keys', tablefmt='psql', showindex=False))
    return weights


def fatch_infomation(outcome):
    paper_id = []
    math_id = []
    sim = []
    for item in outcome:
        paper_id.append(item["math_id"][:10])
        math_id.append(item["math_id"])
        sim.append(item["similarity"])
    publish_time = []
    for item in paper_id:
        year = int(item[:2])
        month = int(item[2:4])
        time_score = ((year - 17) * 12 + (month - 10)) * (1 / 59)
        publish_time.append(time_score)

    candidates = []
    for pid, mid, tid in zip(paper_id, math_id, range(len(publish_time))):
        cursor.execute(f"SELECT * FROM all_paper_info WHERE id = {repr(pid)}")
        pvalues = cursor.fetchall()
        cursor.execute(f"SELECT math_id, latex_expression, is_inline FROM math_info WHERE math_id = {repr(mid)}")
        mvalues = cursor.fetchall()
        candidates.append({"math_id": mvalues[0][0], "latex_expression": mvalues[0][1], "exp_score": sim[tid], "is_inline": mvalues[0][2], "paper_id": pvalues[0][0], "title": pvalues[0][1], "abstract": pvalues[0][2], "keywords": pvalues[0][3], "num_exp": pvalues[0][4], "impact_factor": pvalues[0][5], "citations": pvalues[0][6], "time_score": publish_time[tid]})
    return candidates

def normalize_format(latex_expr):
    for f in latex_format_symbols:
        latex_expr = re.sub("\\" + f, "", latex_expr)

    variables = sorted(set(re.findall(r'\b[a-zA-Z]\b', latex_expr)))
    replacement_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    var_mapping = {}
    for i, var in enumerate(variables):
        if i < len(replacement_letters):
            var_mapping[var] = replacement_letters[i]
    for var, replacement in var_mapping.items():
        latex_expr = re.sub(rf'\b{var}\b', replacement, latex_expr)

    return latex_expr

def compute_final(info, query_text):
    out_line_weight = float(input("你认为与行内公式相比，行间公式的重要性(介于0~1之间的数值, 如果行间公式重要性是w, 那行内公式的重要性就是1-w):  "))
    citation_weight = float(input("你认为与论文的影响因子相比，论文引用数量的重要性(介于0~1之间的数值, 如果引用数量的重要性是w, 那影响因子的重要性就是1-w):  "))

    used_info = {"math_id": [], "exp_score": [], "time_score": [], "impact_factor": [], "citations": []}
    passages = []
    is_inline = []
    for item in info:
        title = item["title"]
        abstract = item["abstract"]
        keywords = item["keywords"]
        passages.append(f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}")
        is_inline.append(out_line_weight if item["is_inline"] == 0 else 1 - out_line_weight)
        used_info["math_id"].append(item["math_id"])
        used_info["exp_score"].append(item["exp_score"])
        used_info["time_score"].append(item["time_score"])
        used_info["impact_factor"].append(item["impact_factor"])
        used_info["citations"].append(item["citations"])
    
    res = topic_model.predict([(query_text, item) for item in passages])
    used_info.update({"topic_score": res})
    used_info.update({"is_inline": np.array(is_inline)})
    used_info.update({"exp_score": np.array(used_info["exp_score"])})
    used_info.update({"time_score": np.array(used_info["time_score"])})
    imf = np.array(used_info["impact_factor"])
    imf = (np.max(imf) - imf) / (np.max(imf) - np.min(imf))
    cat = np.array(used_info["citations"])
    cat = (np.max(cat) - cat) / (np.max(cat) - np.min(cat))
    used_info.update({"importance_score": cat * citation_weight + imf * (1-citation_weight)})
    del used_info["impact_factor"]
    del used_info["citations"]
    return used_info

def main():

    query_expression = r"a^2+b^2=c^2"
    expression_describtion = "this is about math"
    query_text = "we are going to find a paper about triangle"

    exp_keywords = get_keywords(expression_describtion)
    kw_exp = "expression:" + query_expression + "\nkeywords:" + exp_keywords
    first_outcome = initial_selection(kw_exp, 20)
    all_info = fatch_infomation(first_outcome)
    metrix = compute_final(all_info, query_text)
    weights = AHP()
    if weights is None:
        return
    weights_map = {"exp_score": weights[0], "time_score": weights[3], "topic_score": weights[1], "is_inline": weights[4], "importance_score": weights[2]}
    final_score = np.zeros_like(metrix["exp_score"])
    for key, value in metrix.items():
        if key == "math_id":
            continue
        final_score += metrix[key] * weights_map[key]
    metrix.update({"final_score": final_score})

    show_table = pd.DataFrame(all_info)
    del show_table["exp_score"]
    del show_table["abstract"]
    del show_table["keywords"]
    del show_table["num_exp"]
    del show_table["impact_factor"]
    del show_table["citations"]
    del show_table["time_score"]

    show_table = show_table[["math_id", "title", "latex_expression",]]

    show_table["exp_score"] = metrix["exp_score"]
    show_table["topic_score"] = metrix["topic_score"]
    show_table["importance_score"] = metrix["importance_score"]
    show_table["time_score"] = metrix["time_score"]
    show_table["inline_score"] = metrix["is_inline"]
    show_table["final_score"] = metrix["final_score"]
    
    show_table = show_table.sort_values("final_score", ascending=False)
    show_table = show_table.applymap(lambda x: textwrap.fill(str(x), width=20))
    print("查询结果为：")
    print(tabulate(show_table, headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    main()