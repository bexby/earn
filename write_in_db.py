import pymssql
import os
import json
import ipdb
from tqdm import tqdm

conn = pymssql.connect(server="DESKTOP-JC7DEJ3", host='',port=1433, user='sa', password='123', database='retrieval_math_expression')
cursor = conn.cursor()
math_info_folder = r"C:\vscode_project\retrieval\math_info"
all_paper_file = r"C:\vscode_project\retrieval\all_paper_info.jsonl"

def write_all_paper_info(path):
    data = []
    # ipdb.set_trace()
    with open(path, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            d = json.loads(line)
            data.append(tuple(d.values()))
    
    cursor.executemany("insert into all_paper_info (id, title, abstract, keywords, num_exp) values (%s, %s, %s, %s, %d)", data)
    cursor.connection.commit()



def write_math_info(path):
    math_info_list = os.listdir(math_info_folder)
    for file in tqdm(math_info_list, desc="imported files"):
        data = []
        with open(os.path.join(path, file), "r") as fr:
            lines = fr.readlines()
            for line in lines:
                d = json.loads(line)
                data.append(tuple(d.values()))
        
        cursor.executemany("insert into math_info (math_id, latex_expression, previous_text, text_keywords, nor_expression, is_inline) values (%s, %s, %s, %s, %s, %s)", data)
        cursor.connection.commit()


if __name__ == "__main__":
    write_all_paper_info(all_paper_file)
    write_math_info(math_info_folder)

    conn.close()