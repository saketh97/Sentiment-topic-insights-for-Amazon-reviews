import os, re, gzip, json
import pandas as pd
from pathlib import Path
from typing import Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

Stopwords = set(stopwords.words('english'))

def _read_jsonl_gz(path: str) -> pd.DataFrame:
    """Read Amazon reviews json into  a dataframe"""
    records = []
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, "rt",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj=json.loads(line)
            records.append(obj)
        return pd.DataFrame.from_records(records)

def clean_text(s:str) -> str:
    if not isinstance(s,str):
        return ""
    s=s.lower()
    s=re.sub(r"http\S+|www\.\S+"," ",s) 
    s=re.sub(r"^[a-z\s]"," ",s)
    tokens=word_tokenize(s)
    tokens = [t for t in tokens if t not in Stopwords and len(t) > 2]
    return " ".join(tokens)

def load_and_prepare(input_path: str, output_csv: str, drop_neutral: bool= True) -> pd.DataFrame:
    """ 
    Load Amazon reviews , creates sentiment label  from rating: 
    y=1 if overall >=4, y=0  if overall<=2 drop '3' if drop_neutral = True
    returns cleaned DataFrame and writes to csv
    """
    
    df = _read_jsonl_gz(input_path)
    
    keep_cols = ["reviewText", "overall", "summary", "asin","reviewTime"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df.rename(columns={"reviewText":"text", "overall":"rating"}, inplace=True)
    
    if drop_neutral:
        df = df[df['rating'].isin([1,2,4,54])]
    df["label"]=(df['rating']>=4).astype(int)
    
    df["text_clean"]=df["text"].apply(clean_text)
    
    df=df[df["text_clean"].str.len() > 0].dropna(subset=["text_clean"])
    Path(os.path.dirname(output_csv)).mkdir(parents=True,exist_ok=True)
    df.to_csv(output_csv,index=False)
    return df

if __name__ =="__main__":
    in_path="data/raw/Musical_Instruments_5.json.gz"
    out_path="data/processed/amazon_mi_clean.csv"
    df = load_and_prepare(in_path,out_path)
    print(df.head())
    print(df["label"].value_counts())