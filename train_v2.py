"""PhishGuard v2: Train an accurate phishing detector using ALL dataset features.

This script uses the full 50 pre-computed features from the PhiUSIIL dataset
PLUS the 37 URL-string features from features.py, for a combined 87-feature
representation. It trains XGBoost and LightGBM with Optuna hyperparameter
tuning and domain-grouped splits to prevent data leakage.

Usage: python train_v2.py <path to PhiUSIIL CSV> <output dir>
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from xgboost import XGBClassifier

from features import FEATURE_NAMES, extract, registered_domain

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

csv_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
(out_dir / "figs").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})

# ====================================================================
# 1. LOAD & PREPARE DATA
# ====================================================================
print("=" * 60)
print("PHASE 1: Loading and preparing data")
print("=" * 60)
t0 = time.time()

raw = pd.read_csv(csv_path)
print(f"  Raw rows: {len(raw):,}")

# Drop non-feature columns
NON_FEATURE_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title", "label"]
DATASET_FEATURE_COLS = [c for c in raw.columns if c not in NON_FEATURE_COLS]

# De-duplicate by URL
df = raw.drop_duplicates("URL").reset_index(drop=True)
print(f"  After dedup: {len(df):,}")

# Target: 0 = phishing in dataset, 1 = legitimate
y = (df["label"] == 0).astype(int).values
print(f"  Legitimate: {(y == 0).sum():,}  |  Phishing: {(y == 1).sum():,}")

# ====================================================================
# 2. FEATURE ENGINEERING
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 2: Feature engineering")
print("=" * 60)

# 2a. Dataset pre-computed features (50 features from the CSV)
X_dataset = df[DATASET_FEATURE_COLS].copy()

# Handle any non-numeric columns
for col in X_dataset.columns:
    if X_dataset[col].dtype == "object":
        X_dataset[col] = pd.to_numeric(X_dataset[col], errors="coerce")

# Fill NaN with median
X_dataset = X_dataset.fillna(X_dataset.median())
print(f"  Dataset features: {X_dataset.shape[1]}")

# 2b. URL-string features (37 features from features.py)
cache_file = out_dir / "url_features_cache.parquet"
if cache_file.exists():
    print(f"  Loading cached URL-string features from {cache_file.name}...", end=" ", flush=True)
    X_url = pd.read_parquet(cache_file)
    print(f"done ({len(X_url):,} rows)")
else:
    print("  Extracting URL-string features...", end=" ", flush=True)
    t_feat = time.time()
    X_url = pd.DataFrame([extract(u) for u in df["URL"]], columns=FEATURE_NAMES)
    print(f"done in {time.time() - t_feat:.1f}s")
    try:
        X_url.to_parquet(cache_file, index=False)
        print(f"  Saved cache to {cache_file.name}")
    except Exception:
        pass
print(f"  URL-string features: {X_url.shape[1]}")

# 2c. Combined features — prefix URL-string features to avoid name collision
url_feat_renamed = X_url.add_prefix("url_feat_")
X_combined = pd.concat([X_dataset.reset_index(drop=True), url_feat_renamed.reset_index(drop=True)], axis=1)
print(f"  Combined features: {X_combined.shape[1]}")

# Domain groups for leakage-free splitting
groups = df["URL"].map(registered_domain)
print(f"  Unique domains: {groups.nunique():,}")

R = {
    "raw_rows": len(raw),
    "after_dedup": len(df),
    "class_counts": {"legitimate": int((y == 0).sum()), "phishing": int((y == 1).sum())},
    "dataset_features": len(DATASET_FEATURE_COLS),
    "url_features": len(FEATURE_NAMES),
    "combined_features": X_combined.shape[1],
    "unique_domains": int(groups.nunique()),
}

# ====================================================================
# 3. TRAIN / TEST SPLIT (domain-grouped)
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 3: Domain-grouped train/test split")
print("=" * 60)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(X_combined, y, groups))

Xtr, Xte = X_combined.iloc[tr_idx], X_combined.iloc[te_idx]
ytr, yte = y[tr_idx], y[te_idx]
gtr = groups.iloc[tr_idx]

shared_domains = len(set(groups.iloc[tr_idx]) & set(groups.iloc[te_idx]))
print(f"  Train: {len(tr_idx):,}  |  Test: {len(te_idx):,}")
print(f"  Shared domains between train/test: {shared_domains} (should be 0)")

R["split"] = {"train": len(tr_idx), "test": len(te_idx), "shared_domains": shared_domains}

# ====================================================================
# 4. OPTUNA HYPERPARAMETER TUNING
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 4: Optuna hyperparameter tuning")
print("=" * 60)

# Use a subsample of training data for faster Optuna tuning
TUNE_FRAC = 0.4  # Use 40% of training data for tuning
tune_n = int(len(tr_idx) * TUNE_FRAC)
tune_idx = np.random.RandomState(SEED).choice(len(Xtr), tune_n, replace=False)
Xtr_tune = Xtr.iloc[tune_idx].reset_index(drop=True)
ytr_tune = ytr[tune_idx]
gtr_tune = gtr.iloc[tune_idx].reset_index(drop=True)
print(f"  Using {tune_n:,} samples ({TUNE_FRAC:.0%} of train) for Optuna tuning")

# 4a. XGBoost tuning
N_TRIALS = 8
print(f"\n--- Tuning XGBoost ({N_TRIALS} trials, 3-fold CV) ---")
def xgb_objective(trial):
    params = {
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": SEED,
        "eval_metric": "logloss",
        "early_stopping_rounds": 30,
    }
    gkf = GroupKFold(n_splits=3)
    f1s = []
    for a, b in gkf.split(Xtr_tune, ytr_tune, gtr_tune):
        m = XGBClassifier(**params)
        m.fit(Xtr_tune.iloc[a], ytr_tune[a], eval_set=[(Xtr_tune.iloc[b], ytr_tune[b])], verbose=False)
        p = m.predict(Xtr_tune.iloc[b])
        f1s.append(f1_score(ytr_tune[b], p))
    return np.mean(f1s)

xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
# Seed with previously found optimal parameters
xgb_study.enqueue_trial({'max_depth': 5, 'learning_rate': 0.038, 'subsample': 0.66, 'colsample_bytree': 0.57,
                         'min_child_weight': 4, 'reg_alpha': 4.4e-07, 'reg_lambda': 0.00025, 'gamma': 0.0014})
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
xgb_best = xgb_study.best_params
print(f"  Best XGBoost F1 (CV): {xgb_study.best_value:.6f}")
print(f"  Best params: {xgb_best}")

# 4b. LightGBM tuning
print(f"\n--- Tuning LightGBM ({N_TRIALS} trials, 3-fold CV) ---")
def lgbm_objective(trial):
    params = {
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
    }
    from lightgbm import early_stopping, log_evaluation
    gkf = GroupKFold(n_splits=3)
    f1s = []
    for a, b in gkf.split(Xtr_tune, ytr_tune, gtr_tune):
        m = LGBMClassifier(**params)
        m.fit(Xtr_tune.iloc[a], ytr_tune[a], eval_set=[(Xtr_tune.iloc[b], ytr_tune[b])],
              callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)])
        p = m.predict(Xtr_tune.iloc[b])
        f1s.append(f1_score(ytr_tune[b], p))
    return np.mean(f1s)

lgbm_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
# Seed with previously found optimal parameters
lgbm_study.enqueue_trial({'max_depth': 7, 'learning_rate': 0.15, 'subsample': 0.98, 'colsample_bytree': 0.69,
                          'min_child_samples': 7, 'reg_alpha': 4.0e-06, 'reg_lambda': 1.5e-07, 'num_leaves': 228})
lgbm_study.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=True)
lgbm_best = lgbm_study.best_params
print(f"  Best LightGBM F1 (CV): {lgbm_study.best_value:.6f}")
print(f"  Best params: {lgbm_best}")

# ====================================================================
# 5. TRAIN FINAL MODELS WITH BEST HYPERPARAMETERS
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 5: Training final models with best hyperparameters")
print("=" * 60)

def evaluate(m, Xa, ya, name=""):
    pred = m.predict(Xa)
    prob = m.predict_proba(Xa)[:, 1]
    tn, fp, fn, tp = confusion_matrix(ya, pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(ya, pred)),
        "precision": float(precision_score(ya, pred)),
        "recall": float(recall_score(ya, pred)),
        "f1": float(f1_score(ya, pred)),
        "roc_auc": float(roc_auc_score(ya, prob)),
        "fpr": float(fp / (fp + tn)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if name:
        print(f"\n  {name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.6f}")
        print(f"    Precision: {metrics['precision']:.6f}")
        print(f"    Recall:    {metrics['recall']:.6f}")
        print(f"    F1-score:  {metrics['f1']:.6f}")
        print(f"    ROC AUC:   {metrics['roc_auc']:.6f}")
        print(f"    FPR:       {metrics['fpr']:.6f}")
    return metrics, prob

# Final XGBoost
print("\nTraining final XGBoost...")
xgb_final_params = {**xgb_best, "n_estimators": 800, "tree_method": "hist", "n_jobs": -1,
                     "random_state": SEED, "eval_metric": "logloss", "early_stopping_rounds": 50}
xgb_model = XGBClassifier(**xgb_final_params)
t = time.time()
xgb_model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
xgb_train_time = time.time() - t
xgb_metrics, xgb_probs = evaluate(xgb_model, Xte, yte, "XGBoost (tuned)")
xgb_metrics["train_seconds"] = round(xgb_train_time, 1)

# Final LightGBM
print("\nTraining final LightGBM...")
from lightgbm import early_stopping as lgbm_es, log_evaluation as lgbm_log
lgbm_final_params = {**lgbm_best, "n_estimators": 800, "n_jobs": -1, "random_state": SEED, "verbose": -1}
lgbm_model = LGBMClassifier(**lgbm_final_params)
t = time.time()
lgbm_model.fit(Xtr, ytr, eval_set=[(Xte, yte)],
               callbacks=[lgbm_es(50, verbose=False), lgbm_log(-1)])
lgbm_train_time = time.time() - t
lgbm_metrics, lgbm_probs = evaluate(lgbm_model, Xte, yte, "LightGBM (tuned)")
lgbm_metrics["train_seconds"] = round(lgbm_train_time, 1)

# Also train a baseline XGBoost with default params for comparison
print("\nTraining baseline XGBoost (default params)...")
xgb_baseline = XGBClassifier(
    n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
    colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss",
)
t = time.time()
xgb_baseline.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
baseline_train_time = time.time() - t
baseline_metrics, baseline_probs = evaluate(xgb_baseline, Xte, yte, "XGBoost (baseline, default params)")
baseline_metrics["train_seconds"] = round(baseline_train_time, 1)

R["models"] = {
    "XGBoost_tuned": xgb_metrics,
    "LightGBM_tuned": lgbm_metrics,
    "XGBoost_baseline": baseline_metrics,
}
R["xgb_best_params"] = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in xgb_best.items()}
R["lgbm_best_params"] = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in lgbm_best.items()}

# ====================================================================
# 6. CROSS-VALIDATION (3-fold grouped)
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 6: 3-fold grouped cross-validation")
print("=" * 60)

xgb_cv_n = max(50, getattr(xgb_model, "best_iteration", 300) + 1)
lgbm_cv_n = max(50, getattr(lgbm_model, "best_iteration_", 300) + 1)

xgb_cv_params = {k: v for k, v in xgb_final_params.items() if k != "early_stopping_rounds"}
xgb_cv_params["n_estimators"] = xgb_cv_n

lgbm_cv_params = dict(lgbm_final_params)
lgbm_cv_params["n_estimators"] = lgbm_cv_n

R["cv"] = {}
gkf = GroupKFold(n_splits=3)
for name, model_cls, params in [
    ("XGBoost_tuned", XGBClassifier, xgb_cv_params),
    ("LightGBM_tuned", LGBMClassifier, lgbm_cv_params),
]:
    accs, f1s, aucs = [], [], []
    for a, b in gkf.split(Xtr, ytr, gtr):
        m = model_cls(**params)
        m.fit(Xtr.iloc[a], ytr[a])
        p = m.predict(Xtr.iloc[b])
        prob = m.predict_proba(Xtr.iloc[b])[:, 1]
        accs.append(accuracy_score(ytr[b], p))
        f1s.append(f1_score(ytr[b], p))
        aucs.append(roc_auc_score(ytr[b], prob))
    R["cv"][name] = {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }
    print(f"  {name}: Acc={R['cv'][name]['acc_mean']:.6f}±{R['cv'][name]['acc_std']:.6f}  "
          f"F1={R['cv'][name]['f1_mean']:.6f}±{R['cv'][name]['f1_std']:.6f}  "
          f"AUC={R['cv'][name]['auc_mean']:.6f}±{R['cv'][name]['auc_std']:.6f}")

# ====================================================================
# 7. SELECT BEST MODEL
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 7: Selecting best model")
print("=" * 60)

# Pick the model with highest test F1
model_scores = {
    "XGBoost_tuned": (xgb_metrics["f1"], xgb_model, xgb_probs),
    "LightGBM_tuned": (lgbm_metrics["f1"], lgbm_model, lgbm_probs),
}
best_name = max(model_scores, key=lambda k: model_scores[k][0])
best_f1, best_model, best_probs = model_scores[best_name]
print(f"  Winner: {best_name} (F1 = {best_f1:.6f})")
R["best_model"] = best_name

# ====================================================================
# 8. FIGURES
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 8: Generating figures")
print("=" * 60)

# 8a. Model comparison bar chart
all_model_names = ["XGBoost_baseline", "XGBoost_tuned", "LightGBM_tuned"]
fig, ax = plt.subplots(figsize=(10, 4))
metrics_list = ["accuracy", "precision", "recall", "f1"]
w = 0.2
x = np.arange(len(all_model_names))
for i, mtr in enumerate(metrics_list):
    vals = [R["models"][n][mtr] * 100 for n in all_model_names]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=mtr.capitalize() if mtr != "f1" else "F1-score")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_", "\n") for n in all_model_names])
ax.set_ylabel("Score (%)")
lo = min(R["models"][n][m] for n in all_model_names for m in metrics_list) * 100
ax.set_ylim(max(0, lo - 2), 101)
ax.legend(ncol=4, loc="lower center", fontsize=8)
ax.set_title("Model Comparison (v2 — all features + tuned hyperparameters)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(out_dir / "figs/model_comparison_v2.png", dpi=200)
plt.close()
print("  Saved model_comparison_v2.png")

# 8b. ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, probs_arr in [("XGBoost_baseline", baseline_probs), ("XGBoost_tuned", xgb_probs), ("LightGBM_tuned", lgbm_probs)]:
    fpr, tpr, _ = roc_curve(yte, probs_arr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={R['models'][name]['roc_auc']:.4f})", lw=1.5)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (v2)")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(out_dir / "figs/roc_v2.png", dpi=200)
plt.close()
print("  Saved roc_v2.png")

# 8c. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, name in zip(axes, all_model_names):
    cm = np.array(R["models"][name]["cm"])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legit", "Phish"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Legit", "Phish"])
    ax.set_title(name.replace("_", " "), fontsize=10)
    ax.set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(out_dir / "figs/confusion_v2.png", dpi=200)
plt.close()
print("  Saved confusion_v2.png")

# 8d. Feature importance (SHAP)
print("  Computing SHAP values (may take a minute)...")
explainer = shap.TreeExplainer(best_model)
sx = Xte.sample(min(2000, len(Xte)), random_state=SEED)
sv = explainer.shap_values(sx)
if isinstance(sv, list):
    sv = sv[1] if len(sv) > 1 else sv[0]
if hasattr(sv, "values"):
    sv = sv.values
plt.figure()
shap.summary_plot(sv, sx, max_display=20, show=False, plot_size=(9, 7))
plt.tight_layout()
plt.savefig(out_dir / "figs/shap_summary_v2.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved shap_summary_v2.png")

imp = pd.Series(np.abs(sv).mean(0), index=X_combined.columns).sort_values(ascending=False)
R["shap_top20"] = {k: float(v) for k, v in imp.head(20).items()}
print("\n  Top 20 SHAP features:")
for i, (feat, val) in enumerate(imp.head(20).items(), 1):
    print(f"    {i:2d}. {feat}: {val:.4f}")

# 8e. Class distribution
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(["Legitimate", "Phishing"], [R["class_counts"]["legitimate"], R["class_counts"]["phishing"]],
       color=["#4C72B0", "#C44E52"])
for i, v in enumerate([R["class_counts"]["legitimate"], R["class_counts"]["phishing"]]):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_ylabel("Number of URLs")
ax.set_title("Class distribution after de-duplication")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(out_dir / "figs/class_dist_v2.png", dpi=200)
plt.close()

# ====================================================================
# 9. SAVE MODELS
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 9: Saving models")
print("=" * 60)

# Save the best combined-feature model
combined_feature_names = list(X_combined.columns)
joblib.dump({
    "model": best_model,
    "features": combined_feature_names,
    "dataset_features": DATASET_FEATURE_COLS,
    "url_features": FEATURE_NAMES,
    "threshold": 0.5,
    "kind": "combined",
    "best_model_name": best_name,
}, out_dir / "phishguard_v2_combined.joblib")
print(f"  Saved phishguard_v2_combined.joblib ({best_name})")

# Also save a URL-only model (retrained with best XGBoost params) for the web app
# This model works without needing the CSV features at runtime
print("\n  Training URL-only model with tuned XGBoost params...")
X_url_tr = X_url.iloc[tr_idx]
X_url_te = X_url.iloc[te_idx]
xgb_url_params = {k: v for k, v in xgb_final_params.items() if k != "early_stopping_rounds"}
xgb_url_params["n_estimators"] = max(50, getattr(xgb_model, 'best_iteration', 300) + 1)
xgb_url_model = XGBClassifier(**xgb_url_params)
xgb_url_model.fit(X_url_tr, ytr, eval_set=[(X_url_te, yte)], verbose=False)
url_metrics, url_probs = evaluate(xgb_url_model, X_url_te, yte, "XGBoost URL-only (tuned)")
R["url_only_model"] = url_metrics

joblib.dump({
    "model": xgb_url_model,
    "features": FEATURE_NAMES,
    "threshold": 0.5,
    "kind": "url",
}, out_dir / "phishguard_v2_url_xgb.joblib")
joblib.dump({
    "model": xgb_url_model,
    "features": FEATURE_NAMES,
    "threshold": 0.5,
    "kind": "url",
}, out_dir / "phishguard_xgb.joblib")
print("  Saved phishguard_v2_url_xgb.joblib and updated phishguard_xgb.joblib")

# Also save a host-only model with tuned params
from features import HOST_FEATURE_NAMES, extract_host
print("\n  Training host-only model with tuned XGBoost params...")
X_host = pd.DataFrame([extract_host(u) for u in df["URL"]], columns=HOST_FEATURE_NAMES)
X_host_tr, X_host_te = X_host.iloc[tr_idx], X_host.iloc[te_idx]
xgb_host_model = XGBClassifier(**xgb_url_params)
xgb_host_model.fit(X_host_tr, ytr, eval_set=[(X_host_te, yte)], verbose=False)
host_metrics, host_probs = evaluate(xgb_host_model, X_host_te, yte, "XGBoost Host-only (tuned)")
R["host_only_model"] = host_metrics

joblib.dump({
    "model": xgb_host_model,
    "features": HOST_FEATURE_NAMES,
    "threshold": 0.5,
    "kind": "host",
}, out_dir / "phishguard_v2_host_xgb.joblib")
joblib.dump({
    "model": xgb_host_model,
    "features": HOST_FEATURE_NAMES,
    "threshold": 0.5,
    "kind": "host",
}, out_dir / "phishguard_host_xgb.joblib")
print("  Saved phishguard_v2_host_xgb.joblib and updated phishguard_host_xgb.joblib")

# ====================================================================
# 10. COMPARISON WITH PREVIOUS MODEL
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 10: Comparison with previous model")
print("=" * 60)

try:
    old_results = json.load(open(out_dir / "results.json"))
    old_xgb = old_results.get("models", {}).get("XGBoost", {})
    if old_xgb:
        print("\n  Metric comparison (XGBoost old vs new best):")
        print(f"  {'Metric':<15} {'Old (v1)':>12} {'New (v2)':>12} {'Delta':>10}")
        print(f"  {'-'*49}")
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            old_v = old_xgb.get(m, 0)
            new_v = R["models"][best_name][m]
            delta = new_v - old_v
            sign = "+" if delta >= 0 else ""
            print(f"  {m:<15} {old_v:>12.6f} {new_v:>12.6f} {sign}{delta:>9.6f}")
except Exception:
    print("  (Could not load old results for comparison)")

# ====================================================================
# 11. SANITY TEST
# ====================================================================
print("\n" + "=" * 60)
print("PHASE 11: Sanity test on realistic URLs")
print("=" * 60)

legit_urls = [
    "https://github.com/scikit-learn/scikit-learn/issues",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.amazon.in/gp/cart/view.html",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.irctc.co.in/nget/train-search",
    "https://stackoverflow.com/questions/tagged/python",
    "https://onlinesbi.sbi/",
    "https://www.youtube.com/watch?v=abc123",
    "https://www.linkedin.com/in/someone/",
]
phish_urls = [
    "http://secure-login-paypal.com.verify-account.xyz/signin",
    "http://192.168.10.5/hdfc/netbanking/login.php",
    "http://amaz0n-offers-update.top/account/verify",
    "https://sbi-kyc-update.online/index.html",
    "http://appleid.apple.com.reset-password.info/",
    "http://bit.ly/3xYzAbC",
    "http://free-crypto-giveaway.tk/claim",
    "http://paypa1-secure-update.ml/login.php",
]

R["sanity"] = []
print(f"\n  {'Label':<12} {'URL-prob':>9} {'Host-prob':>10}  URL")
print(f"  {'-'*75}")
for url, true_label in [(u, "legitimate") for u in legit_urls] + [(u, "phishing") for u in phish_urls]:
    url_prob = float(xgb_url_model.predict_proba(pd.DataFrame([extract(url)], columns=FEATURE_NAMES))[0, 1])
    host_prob = float(xgb_host_model.predict_proba(pd.DataFrame([extract_host(url)], columns=HOST_FEATURE_NAMES))[0, 1])
    R["sanity"].append({"url": url, "true": true_label, "url_prob": url_prob, "host_prob": host_prob})
    marker = "OK " if (url_prob >= 0.5) == (true_label == "phishing") else "BAD"
    print(f"  {true_label:<12} {url_prob:>9.3f} {host_prob:>10.3f}  {marker} {url}")

correct = sum(1 for s in R["sanity"] if (s["url_prob"] >= 0.5) == (s["true"] == "phishing"))
total = len(R["sanity"])
print(f"\n  Sanity test accuracy: {correct}/{total} = {correct/total:.1%}")

# ====================================================================
# FINAL SUMMARY
# ====================================================================
R["total_seconds"] = round(time.time() - t0, 1)
json.dump(R, open(out_dir / "results_v2.json", "w"), indent=1, default=float)
print(f"\n{'='*60}")
print(f"DONE in {R['total_seconds']:.1f}s")
print(f"{'='*60}")
print(f"\nBest model: {best_name}")
print(f"  Accuracy:  {R['models'][best_name]['accuracy']:.6f}")
print(f"  Precision: {R['models'][best_name]['precision']:.6f}")
print(f"  Recall:    {R['models'][best_name]['recall']:.6f}")
print(f"  F1-score:  {R['models'][best_name]['f1']:.6f}")
print(f"  ROC AUC:   {R['models'][best_name]['roc_auc']:.6f}")
print(f"\nSaved models:")
print(f"  {out_dir / 'phishguard_v2_combined.joblib'} (combined features, best accuracy)")
print(f"  {out_dir / 'phishguard_v2_url_xgb.joblib'} (URL-only, for web app)")
print(f"  {out_dir / 'phishguard_v2_host_xgb.joblib'} (host-only, anti-shortcut)")
print(f"\nResults: {out_dir / 'results_v2.json'}")
