import pymssql
import os
import json
import ipdb
from tqdm import tqdm

conn = pymssql.connect(server="DESKTOP-JC7DEJ3", host='',port=1433, user='sa', password='123', database='retrieval_math_expression')
cursor = conn.cursor()
embeddings_folder = r"C:\vscode_project\retrieval\embeddings"

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
            cursor.execute("UPDATE math_info SET embedding = %s WHERE row_id = %s", (item["embedding"], item["math_id"]))
        cursor.connection.commit()


if __name__ == "__main__":
    write_math_info(embeddings_folder)
    conn.close()