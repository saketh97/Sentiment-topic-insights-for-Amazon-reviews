import os
import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump

import mlflow

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def topic_diversity(top_terms_per_topic):
    # topic diversity = unique terms across all topics / (k* top n)
    all_terms = [t for terms in top_terms_per_topic for t in terms]
    return len(set(all_terms)) / len(all_terms)

def top_terms(H, vocab, topn=12):
    """return list of top terms per topic"""
    K=H.shape[0]
    out=[]
    for k in range(K):
        idx = np.argsort(H[k])[::-1][:topn]
        out.append([vocab[i] for i in idx])
    return out

def save_topic_barplots(H,vocab,out_dir,topn=10):
    ensure_dir(out_dir)
    K=H.shape[0]
    for k in range(K):
        idx = np.argsort(H[k])[::-1][:topn]
        terms = [vocab[i] for i in idx]
        weights = H[k,idx]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(terms[::-1],weights[::-1])
        ax.set_title(f"Topic {k} top terms")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir,f"topic_{k}_top_terms.png"),dpi=150)
        plt.close(fig)

def main(args):
    data_path = args.data
    n_topics = args.n_topics
    max_df = args.max_df
    min_df = args.min_df
    topn=args.topn
    
    fig_dir = "reports/figures/topics"
    model_dir="models/topics"
    artifacts_dir="reports/topics"
    ensure_dir(fig_dir)
    ensure_dir(model_dir)
    ensure_dir(artifacts_dir)
    
    df = pd.read_csv(data_path)
    if "text_clean" not in df.columns:
        raise ValueError("Expected 'text_clean' in processed csv")
    
    vect = CountVectorizer(min_df=min_df,max_df=max_df,ngram_range=(1,1))
    X=vect.fit_transform(df['text_clean'])
    
    # LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=2,
        n_jobs=-1
    )
        
    
    mlflow.set_experiment("marketing-nlp")
    with mlflow.start_run(run_name=f"lda_sklearn_{n_topics} topics") as run:
        W = lda.fit_transform(X)
        H=lda.components_
        
        perplex  = lda.perplexity(X)
        mlflow.log_metric("lda_perplexity",perplex)
        mlflow.log_param("n_topics",n_topics)
        mlflow.log_param("min_df",min_df)
        mlflow.log_param("max_df",max_df)
        
        vocab = vect.get_feature_names_out()
        top_lists = top_terms(H,vocab,topn=topn)
        diversity = topic_diversity(top_lists)
        mlflow.log_metric("topic_diversity",diversity)
        
        #save top terms per topic to CSV
        
        rows=[]
        for k, terms in enumerate(top_lists):
            for rank, term in enumerate(terms,start=1):
                rows.append({'topic':k, "rank":rank, "term":term})
        top_terms_df = pd.DataFrame(rows)
        top_terms_csv=os.path.join(artifacts_dir,"lda_top_terms.csv")
        top_terms_df.to_csv(top_terms_csv,index=False)
        mlflow.log_artifact(top_terms_csv,artifact_path="topics")
        
        #save document wise dominant topic
        dom_topic = W.argmax(axis=1)
        dom_df = pd.DataFrame({"doc_id": np.arange(W.shape[0]),"dominant topic":dom_topic})
        dom_csv = os.path.join(artifacts_dir,"lda_doc_topics.csv")
        dom_df.to_csv(dom_csv,index=False)
        mlflow.log_artifact(dom_csv,artifact_path="topics")
        
        #save bar plots
        save_topic_barplots(H,vocab,fig_dir,topn=min(topn,10))
        mlflow.log_artifact(fig_dir,artifact_path="figures/topics")
        
        # save models for later inference in streamlit
        dump(vect,os.path.join(model_dir,"count_vectorizer.joblib"))
        dump(lda,os.path.join(model_dir,"lda_model.joblib"))
        
        labels_rows = []
        for k in range(len(top_lists)):
            # join first 3 top terms as a label suggestion
            label = " / ".join(top_lists[k][:3]) if len(top_lists[k]) >= 3 else "Topic " + str(k)
            labels_rows.append({"topic": k, "label": label})

        labels_df = pd.DataFrame(labels_rows)
        labels_csv = os.path.join(artifacts_dir, "lda_topic_labels.csv")
        labels_df.to_csv(labels_csv, index=False)
        mlflow.log_artifact(labels_csv, artifact_path="topics")

        print(f"perplexity : {perplex:.2f}")
        print(f"topic diversity : {diversity:.2f}")
        print(f"saved : {top_terms_csv},{dom_csv}")
        print(f"models: {model_dir}/count_vwctorizer.joblib, lda_model.joblib")
        print(f"MLflow run:{run.info.run_id}")
        
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=str,default="data/processed/amazon_mi_clean.csv")
    p.add_argument("--n_topics",type=int,default=10)
    p.add_argument("--min_df",type=int,default=5)
    p.add_argument("--max_df",type=float,default=0.9)
    p.add_argument("--topn",type=int,default=12)
    args=p.parse_args()
    main(args)
    
        
        