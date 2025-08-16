import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import matplotlib.pyplot as plt

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_and_save_cm(y_true, y_pred, out_path: str, labels=("Neg","Pos")):
    cm=confusion_matrix(y_true, y_pred, labels=[0,1])
    fig,ax = plt.subplots()
    im=ax.imshow(cm)
    ax.set_title("confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1], labels)
    ax.set_yticks([0,1], labels)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center")
    fig.tight_layout()
    fig.savefig(out_path,dpi=150)
    plt.close(fig)
    
def main(args):
    data_path=args.data
    model_dir="models"
    fig_dir="reports/figures"
    ensure_dir(model_dir)
    ensure_dir(fig_dir)
    
    # load data
    df = pd.read_csv(data_path)
    if"text_clean" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns text_clean and label  in the processed csv")
    
    #basic hygine checks
    df= df.dropna(subset=["text_clean","label"])
    df =df[df['text_clean'].str.len()> 0]
    df["label"]=df["label"].astype(int)
    
    X = df["text_clean"].values
    y= df["label"].values
    
    # train test split
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=2,stratify=y)
    
    #pipeline
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1,2),
                min_df =3,max_df=0.95,max_features=50000
            )),
            ("clf", LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=2
            )),
        ]
    )

#mlflow setup
    mlflow.set_experiment("marketing_nlp")
    with mlflow.start_run(run_name="tf_idf_logreg_baseline") as run:
        #train
        pipe.fit(X_train, y_train)
        
        #eval
        y_pred = pipe.predict(X_val)
        if hasattr(pipe.named_steps["clf"],"predict_proba"):
            y_proba=pipe.predict_proba(X_val)[:,1]
        else:
            y_proba=None
        acc=accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred,average="binary", zero_division=0)
        mlflow.log_metric("val_accuracy",acc)
        mlflow.log_metric("val_precision",prec)
        mlflow.log_metric("val_recall",rec)
        mlflow.log_metric("val_f1",f1)
        
        if y_proba is not None and len(np.unique(y_val))==2:
            try:
                auc = roc_auc_score(y_val,y_proba)
                mlflow.log_metric(y_val,y_proba)
            except Exception:
                pass
        # confusion matrix image
        cm_path = os.path.join(fig_dir,"confusion matrix_baseline.png")
        plot_and_save_cm(y_val,y_pred,cm_path)
        mlflow.log_artifact(cm_path,artifact_path="figures")
        
        #classification report as text artifact
        report = classification_report(y_val,y_pred,digits=4)
        report_path=os.path.join(fig_dir,"classification_report_baseline.txt")
        with open(report_path, "w",encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(report_path,artifact_path="reports")
        
        #save model locally
        local_model_path = os.path.join(model_dir,"baseline_tfidf_logreg.joblib")
        dump(pipe,local_model_path)
        
        #log model to mlflow
        # Use numpy array (or a plain list) for signature + input_example
        signature = infer_signature(np.array(["great product", "poor quality"]), np.array([1, 0]))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",  # deprecation warning is OK for now
            signature=signature,
            input_example=["amazing fit and quality", "terrible stitching and size"]  # list, not Series
        )
        
        
        print("-----------------------------METRICS----------------------------")
        print(f"Accuracy:{acc:.4f}")
        print(f"Precision:{prec:.4f}")
        print(f"Recall:{rec:.4f}")
        print(f"f1:{f1:.4f}")
        if y_proba is not None and len(np.unique(y_val))==2:
            try:
                print(f"AUC :{auc:.4f}")
            except Exception:
                pass
        print("\n-----------classification Report --------------")
        print(report)
        print(f"\n Saved model ->{local_model_path}")
        print(f"MLflow run -> {run.info.run_id}")
        
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", type=str,default="data/processed/amazon_mi_clean.csv")
    args=parser.parse_args()
    main(args)


                
        
        