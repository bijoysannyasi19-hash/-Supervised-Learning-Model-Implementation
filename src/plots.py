import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import base64
import urllib.request

def p_dist(y, p):
    plt.figure()
    sns.histplot(y, kde=True)
    plt.title("Target Distribution (SalePrice)")
    plt.xlabel("SalePrice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "target_dist.png"), dpi=200)
    plt.close()

def p_miss(df, p):
    plt.figure(figsize=(10, 6))
    mc = df.isnull().sum()
    mc = mc[mc > 0].sort_values(ascending=False)
    mc.plot(kind="bar")
    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Count of Missing Values")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "missing_values.png"), dpi=200)
    plt.close()

def p_feat(df, p):
    plt.figure()
    sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"])
    plt.title("SalePrice vs GrLivArea")
    plt.xlabel("Above Ground Living Area (sq ft)")
    plt.ylabel("SalePrice")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "feat_grlivarea.png"), dpi=200)
    plt.close()

def p_feat2(df, p):
    plt.figure()
    sns.histplot(df['YearBuilt'], kde=True)
    plt.title('Distribution of YearBuilt')
    plt.xlabel('Year Built')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(p, 'feat_yearbuilt.png'), dpi=200)
    plt.close()

def p_corr(df, p):
    plt.figure(figsize=(10, 8))
    nc = df.select_dtypes(include=[np.number])
    c = nc.corr()
    sns.heatmap(c.iloc[:15, :15], annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Top 15 Numeric)")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "corr_heatmap.png"), dpi=200)
    plt.close()

def p_comp(res, p):
    df = pd.DataFrame(res).T
    plt.figure()
    plt.errorbar(df.index, df["test_rmse"], yerr=df["test_rmse_std"], fmt='o')
    plt.title("Model Comparison (RMSE)")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "model_comparison.png"), dpi=200)
    plt.close()

def p_lc(m, X, y, p):
    from sklearn.model_selection import learning_curve
    ts, tr, te = learning_curve(m, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    trm = -np.mean(tr, axis=1)
    tem = -np.mean(te, axis=1)
    plt.figure()
    plt.plot(ts, trm, label="Train RMSE")
    plt.plot(ts, tem, label="Val RMSE")
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(p, "learning_curve.png"), dpi=200)
    plt.close()

def p_eval(y, yp, p):
    plt.figure()
    plt.scatter(y, yp, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "pred_vs_actual.png"), dpi=200)
    plt.close()

def p_res(y, yp, p):
    rs = y - yp
    plt.figure()
    plt.scatter(yp, rs, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "residuals.png"), dpi=200)
    plt.close()

def p_resh(y, yp, p):
    rs = y - yp
    plt.figure()
    sns.histplot(rs, kde=True)
    plt.title("Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "residual_hist.png"), dpi=200)
    plt.close()

def p_imp(fi, p):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=fi.head(15))
    plt.title("Permutation Feature Importance (Top 15)")
    plt.xlabel("Importance Decrease in RMSE")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "feature_importance.png"), dpi=200)
    plt.close()

def p_err(y, yp, X, p):
    rs = np.abs(y - yp)
    plt.figure()
    sns.scatterplot(x=X["GrLivArea"], y=rs)
    plt.title("Absolute Error vs GrLivArea")
    plt.xlabel("Above Ground Living Area (sq ft)")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig(os.path.join(p, "error_analysis.png"), dpi=200)
    plt.close()

def p_arch(p):
    c = "graph TD\nRaw[Raw Data] --> Split[Train/Test Split]\nSplit --> Pipe[Preprocessing Pipeline]\nPipe --> FE[Feature Engineering]\nFE --> Comp[Model Comparison]\nComp --> Tune[Hyperparameter Tuning]\nTune --> Final[Final Model]\nFinal --> Eval[Test Evaluation]\nEval --> Interp[Interpretation]\nInterp --> Rep[Report]"
    try:
        j = json.dumps({"code": c, "mermaid": {"theme": "default"}})
        s = base64.urlsafe_b64encode(j.encode()).decode()
        u = "https://mermaid.ink/img/" + s
        urllib.request.urlretrieve(u, os.path.join(p, "architecture.png"))
    except Exception:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Architecture Diagram\n(Mermaid Export Failed)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(os.path.join(p, "architecture.png"), dpi=200)
        plt.close()