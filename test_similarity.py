import pymssql
import os
import json
import ipdb
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

conn = pymssql.connect(server="DESKTOP-JC7DEJ3", host='',port=1433, user='sa', password='123', database='retrieval_math_expression')
cursor = conn.cursor()
model_path = ""
model = SentenceTransformer(model_path, device="cuda")

def get_infomation(expression):

    cursor.execute(f"SELECT math_id, embedding FROM math_info")
    ipdb.set_trace()
    while True:
        values = cursor.fetchmany(10000)
        if not values:
            break


if __name__ == "__main__":
    expression = r"a^2+b^2=c^2"
    get_infomation(expression)
    conn.close()