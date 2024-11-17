import os
import re
import ipdb
from bs4 import BeautifulSoup
from distutils.filelist import findall
from bs4 import NavigableString
import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm
import json

latex_symbols = {
    # 基础运算符
    '+', '-', '*', '/', '=', '<', '>', '!', '%', '^', '(', '[', '{', '|', '&', ':', ';', '#',
    '\\neq', '\\pm', '\\mp', '\\div', '\\times', '\\ast', '\\circ',
    # 关系运算符
    '=', '\\neq', '\\equiv', '\\sim', '\\approx', '\\leq', '\\geq', '<', '>', '\\subset', '\\supset',
    '\\subseteq', '\\supseteq', '\\in', '\\notin', '\\parallel', '\\perp', '\\ll', '\\gg',
    # 集合运算符
    '\\cup', '\\cap', '\\setminus', '\\subset', '\\supset', '\\subseteq', '\\supseteq', '\\emptyset',
    '\\in', '\\notin', '\\forall', '\\exists', '\\neg', '\\models',
    # 逻辑运算符
    '\\land', '\\lor', '\\neg', '\\lnot', '\\implies', '\\iff', '\\forall', '\\exists', '\\bot', '\\top',
    '\\vdash', '\\dashv',
    # 算数运算符
    '+', '-', '\\pm', '\\mp', '\\times', '\\div', '\\ast', '\\cdot', '\\bullet',
    # 积和运算符
    '\\sum', '\\prod', '\\int', '\\oint', '\\coprod', '\\bigsqcup', '\\bigcup', '\\bigcap', '\\bigvee', '\\bigwedge',
    '\\int_{a}^{b}', '\\lim', '\\limsup', '\\liminf',
    # 其他常用运算符
    '\\mod', '\\gcd', '\\text{lcm}', '\\argmax', '\\argmin', '\\log', '\\ln', '\\exp', '\\det', '\\dim', '\\ker',
    '\\deg', '\\Im', '\\Re', '\\Pr', '\\inf', '\\sup', '\\max', '\\min', '\\arg',
    # 积分符号
    '\\int', '\\oint', '\\iint', '\\iiint', '\\oint_C',
    # 矩阵运算符
    '\\det', '\\text{tr}', 
    # 等号和不等号
    '=', '\\neq', '\\equiv', '\\approx', '\\sim', '\\simeq', '\\doteq', '\\stackrel{?}{=}',
    # 大于小于符号
    '<', '>', '\\leq', '\\geq', '\\ll', '\\gg', '\\nless', '\\ngtr', '\\lneq', '\\gneq',
    # 运算符
    '\\sum', '\\prod', '\\int', '\\oint', '\\lim', '\\limsup', '\\liminf', '\\sqrt', '\\frac', '\\binom', '\\mod', '\\gcd',
    '\\max', '\\min', '\\argmax', '\\argmin', '\\inf', '\\sup', '\\text{gcd}', '\\text{lcm}',
    # 数学函数
    '\\sin', '\\cos', '\\tan', '\\sec', '\\csc', '\\cot', '\\log', '\\ln', '\\exp', '\\limsup', '\\liminf', '\\max', '\\min',
    '\\arg', '\\det', '\\dim', '\\deg', '\\ker', '\\Im', '\\Re', '\\Pr', '\\varkappa', '\\varphi', '\\varpi', '\\varrho', '\\varsigma', '\\varOmega',
    # 上标下标
    '^', '_',
}

latex_format_symbols = {
    # 格式符
    '\\textbf', '\\textit', '\\texttt', '\\emph', '\\underline', '\\overline', '\\overbrace', '\\underbrace', '\\left', '\\right',
    '\\big', '\\Big', '\\bigg', '\\Bigg', '\\textit', '\\textsf', '\\textrm', '\\mathcal', '\\mathbb', '\\mathfrak', '\\mathscr', '\\displaystyle', '\\textrm',
}

class extract:
    def __init__(self, path) -> None:
        self.path = path
        f = open(path, encoding='utf-8')
        text = f.read()
        # text = text.replace("\f", "\\f")
        # text = text.replace("\r", "\\r")
        # text = text.replace("\n", "")
        self.soup = BeautifulSoup(text, "html5lib")
    
    def get_title(self):
        title = self.soup.find("meta", attrs={"name": "twitter:title"})
        if title:
            return title.get("content")
        else:
            return None
    
    def get_keywords(self):
        keywards = self.soup.find("meta", attrs={"name": "keywords"})
        if keywards:
            return keywards.get("content")
        else:
            return None

    def get_abstract(self):
        abstract = self.soup.find("div", attrs={"class": "ltx_abstract"})
        if abstract:
            for child in abstract.children:
                if child.name == "p":
                    return child.text
        else:
            return None

    def is_inline(self, tag):
        # if tag.parents == None:
        #     return None
        for p in tag.parents:
            if p.name == "p":
                return True
            if p.name == "tbody":
                return False
        return False

    def get_right_text(self, s):
        pattern = re.compile("[A-Z]\s?\.")
        if re.search(pattern, s[::-1]) == None:
            return s
        else:
            start = re.search(pattern, s[::-1]).span()[0]
            return s[-start-1:]
       
    def is_pass(self, s):
        count = 0
        for sym in latex_symbols:
            count += s.count(sym)
            if count > 5:
                return False
        return True

    def get_math(self):
        math_list = self.soup.find_all("math")
        
        result = {"latex_expression": [], "previous_text": []}
        
        for math_tag in math_list:
            try:
                math_s = math_tag["alttext"]
            except:
                continue
            math_s = math_s.replace(r"\bm", "")
            if self.is_pass(math_s):
                continue
            else:
                pre_text = ""
                if self.is_inline(math_tag) and math_tag.previous_siblings != None:
                    for brother in math_tag.previous_siblings:
                        if isinstance(brother, NavigableString) and brother.string != "\n":
                            pre_text = brother.string + pre_text
                        elif brother.name == "math":
                            try:
                                pre_text = brother["alttext"] + pre_text
                            except:
                                continue
                elif not self.is_inline(math_tag):
                    pre_grap = None
                    for pre in math_tag.previous_elements:
                        if pre.name == "p":
                            pre_grap = pre
                            break
                    if pre_grap != None:
                        for bro in reversed(list(pre_grap.children)):
                            if isinstance(bro, NavigableString) and bro.string != "\n":
                                pre_text = bro.string + pre_text
                            elif bro.name == "math":
                                try:
                                    pre_text = bro["alttext"] + pre_text
                                except:
                                    continue
                
                pre_text = self.get_right_text(pre_text) 
                # ipdb.set_trace()             
                result["latex_expression"].append(math_s)
                result["previous_text"].append(pre_text)
        return result


def load_file_name(root_path) -> dict[str, list[str]]:
    ls_dir = os.listdir(root_path)
    result = dict()
    for dir in ls_dir:
        ls_file = os.listdir(os.path.join(root_path, dir))
        tem = []
        for f in ls_file:
            tem.append(f)
        result.update({dir: tem})
    return result

def extract_keywords(kw_model,ls_text): 
    keywords = kw_model.extract_keywords(ls_text, keyphrase_ngram_range=(1, 2))
    result = []
    for kw in keywords:
        if kw == []:
            result.append("")
        else:
            result.append(kw[0][0])
    return result

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


def main():
    data_path = r"C:\Users\86159\Downloads\ar5iv_1710-2209\ar5iv"
    save_path = r"C:\vscode_project\retrieval"
    ff = load_file_name(data_path)
    all_paper_info = []    # {"id": name, "title": title, "abstract": abstract, "keywords": keywords, }
    paper_math_info = []    # {"math_id": math_id, "latex_expression": exp, "previous_text": pret, "text_keywords": tkw, "nor_expression": nor_exp}
    avaliable_file = []

    # node = extract(r"C:\Users\86159\Downloads\ar5iv_1710-2209\ar5iv\1712\1712.00130.html")
    # res = node.get_math()
    # df = pd.DataFrame(res)
    # df.to_excel("temp3.xlsx")

    kw_model = KeyBERT()

    for folder, file_ls in tqdm(ff.items(), desc="processed folders", position=0):
        if folder <= "1912":
            continue
        for re_file in tqdm(file_ls, desc="files in folder", position=1):
            file = os.path.join(data_path, folder, re_file)
            if os.path.getsize(file) < 50000:
                continue
            avaliable_file.append(file)
            node = extract(file)
            title = node.get_title()
            abstract = node.get_abstract()
            keywords = node.get_keywords()
            paper_id = re_file.split(".")[0] + "." + re_file.split(".")[1]
            
            math_info = node.get_math()
            num_exp = len(math_info["latex_expression"])
            all_paper_info.append({"id": paper_id, "title": title, "abstract": abstract, "keywords": keywords, "num_exp": num_exp})
            text_kw = extract_keywords(kw_model, math_info["previous_text"])
            # math_info.update({"text_keywords": text_kw})
            nor_math = []
            math_id = []
            for i, m in enumerate(math_info["latex_expression"]):
                nor_math.append(normalize_format(m))
                math_id.append(paper_id + "." + str(i).zfill(4))
            # math_info.update({"nor_math_exp": nor_math})
            # math_info.update({"math_id": math_id})

            with open(os.path.join(save_path, "all_paper_info.jsonl"), "a") as fw:
                fw.write(json.dumps(all_paper_info[-1]))
                fw.write("\n")
            
            with open(os.path.join(save_path, "folder" + folder + "_math_info.jsonl"), "a") as fw:
                for i in range(num_exp):
                    try:
                        data = {"math_id": math_id[i], "latex_expression": math_info['latex_expression'][i], "previous_text": math_info['previous_text'][i], "text_keywords": text_kw[i], "nor_expression": nor_math[i]}
                    except:
                        continue
                    fw.write(json.dumps(data))
                    fw.write("\n")


    # ipdb.set_trace()

if __name__ == '__main__':
    main()