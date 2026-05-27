import pandas as pd
from os import listdir
from os.path import isfile, join

dir_path = "C:\\Users\\tomer\\Desktop\\Pokemon\\Toutnament Results"

onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

dic = {}
required_cols = ["Name", "Value"]
for f in onlyfiles:
    if "~" in f:
        continue
    df = pd.read_excel(dir_path + "\\" + f)
    if "Name" in df.columns and "Value" in df.columns:
        df = df.dropna(subset=["Name", "Value"])
        labels = df["Name"].tolist()
        sizes = df["Value"].tolist()
        for i in range(len(labels)):
            if labels[i] not in dic.keys():
                dic[labels[i]] = sizes[i]
            else:
                dic[labels[i]] += sizes[i]
pre_keys = dic.keys()
counter = 0
for key in pre_keys:
    if dic[key] <= 2:
        counter += dic[key]
        dic[key] = 0
dic["Other"] = counter
pd.DataFrame([dic.keys(), dic.values()]).transpose().sort_values(by=1).rename(columns={0:"Name", 1:"Value"}).to_excel("Output.xlsx")
