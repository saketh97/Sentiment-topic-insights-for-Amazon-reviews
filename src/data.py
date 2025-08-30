import os, re, gzip, json
import pandas as pd
from pathlib import Path
from typing import Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup


''' Cleaning for the reviews column
1. remove urls
2. Replace \n \t with space
3. remove special characters replace with ""
4. remove punctuation
5. lowercase
6. strip leading/trailing spaces
7. remove stop words
'''
Stopwords = set(stopwords.words('english'))

def _read_jsonl_gz(path: str) -> pd.DataFrame:
    """Read json into  a dataframe"""
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
    #1.Remove everything between &nbsp and the next ; (inclusive), repeatedly
   
    soup = BeautifulSoup(s, "html.parser")
    for tag in soup.find_all("div"):
        tag.decompose()                # remove the div and its contents
    for tag in soup.find_all("input"):
        tag.decompose()                # remove input elements
    for tag in soup.find_all("a"):
        tag.decompose()                # remove the a and its contents
    s = str(soup)
    
    # remove URLs
    s = re.sub(r'http\S+|www\S+|https\S+', ' ', s, flags=re.IGNORECASE)
    # remove &nbsp; entirely
    s = s.replace('&nbsp;', '')
    #1. Replace \n \t with space
    s = re.sub(r"[\n\t\r]+", " ",s)
    #2. remove special characters replace with ""
    s = re.sub(r"[^a-zA-Z\s]"," ",s)
    #3. remove punctuation
    s=re.sub(r"[^\w\s]"," ",s)
    #4. lowercase
    s=s.lower()
    #5. strip leading/trailing spaces
    s=s.strip()
    tokens=word_tokenize(s)
    #6. remove stop words
    tokens = [t for t in tokens if t not in Stopwords and len(t) > 2]
    return " ".join(tokens)

def load_and_prepare(input_path: str, output_csv: str) -> pd.DataFrame:
    """ 
    Load Amazon reviews , creates sentiment label  from rating: 
    y=2 if overall >=4, y=0  if overall<=2, y=1 for overall=3
    returns cleaned DataFrame and writes to csv
    """
    
    df = _read_jsonl_gz(input_path)
    
    keep_cols = ["reviewText", "overall","reviewTime"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df.rename(columns={"reviewText":"review", "overall":"rating"}, inplace=True)
    df["label"]=df["rating"].apply(lambda x: 2 if x >=4 else (0 if x <=2 else 1))
    df["text_clean"]=df["review"].apply(clean_text)
    
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