"""
XGBoost Classifier — Hyperparameter Tuning
Estimation of Obesity Levels Based on Eating Habits and Physical Condition 

Pipeline:
  1. Load data & Preprocess (imported from Preprocessing.py)
  2. Baseline XGBoost model
  3. Hyperparameter tuning (RandomizedSearchCV)
  4. Evaluation (accuracy, classification report, ROC-AUC, confusion matrix, feature importance)
  5. Sample Decision Tree visualization
  6. Hyperparameter sensitivity plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import matplotlib
from pathlib import Path

from sklearn.model_selection import learning_curve, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree._tree import Tree as SklearnTree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Import centralized preprocessing function
from Preprocessing import get_preprocessed_data

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "XGBoost"
TUNING_DIR = PROJECT_ROOT / "results" / "tuning" / "XGBoost"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_DIR.mkdir(parents=True, exist_ok=True)
(TUNING_DIR / "csv").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test, preprocessor, target_encoder = get_preprocessed_data()

print("\nClasses (in ordinal order):", list(target_encoder.classes_))
n_classes = len(target_encoder.classes_)

# --------------------------------------------------------------------------
# 2. BASELINE MODEL
# --------------------------------------------------------------------------
baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

baseline_pipeline.fit(X_train, y_train)
baseline_preds = baseline_pipeline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline XGBoost accuracy: {baseline_acc:.4f}")

# --------------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (RandomizedSearchCV)
# --------------------------------------------------------------------------
param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 400, 500, 600],
    "classifier__max_depth": [3, 4, 5, 6, 7, 8, 9],
    "classifier__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 2, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.5],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1, 5],
    "classifier__reg_lambda": [0.5, 1, 1.5, 2, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

random_search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    scoring="accuracy",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nStarting hyperparameter search (this can take a few minutes)...")
random_search.fit(X_train, y_train)
results = pd.DataFrame(random_search.cv_results_)

# Save Mean CV Accuracy for Each Hyperparameter
parameters = [
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__learning_rate",
    "param_classifier__subsample",
    "param_classifier__colsample_bytree",
    "param_classifier__min_child_weight",
    "param_classifier__gamma",
    "param_classifier__reg_alpha",
    "param_classifier__reg_lambda"
]

for param in parameters:
    summary = (
        results.groupby(param)["mean_test_score"]
        .mean()
        .reset_index()
        .rename(columns={"mean_test_score": "Mean_CV_Accuracy"})
        .sort_values("Mean_CV_Accuracy", ascending=False)
    )
    filename = param.replace("param_classifier__", "")
    summary.to_csv(TUNING_DIR / "csv" / f"{filename}_mean_cv_accuracy.csv", index=False)

print("\nBest parameters found:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_

# --------------------------------------------------------------------------
# 4. FINAL EVALUATION ON TEST SET
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nTuned model test accuracy: {final_acc:.4f}")
print(f"(Baseline test accuracy was: {baseline_acc:.4f})")

print("\nClassification report:\n")
print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

# ROC-AUC (multiclass, one-vs-rest)
final_probs = best_model.predict_proba(X_test)
macro_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="macro")
weighted_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="weighted")
per_class_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average=None)

print(f"\nMacro-average ROC-AUC (OvR): {macro_roc_auc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")

roc_auc_df = pd.DataFrame({
    "Class": target_encoder.classes_,
    "ROC-AUC": per_class_roc_auc,
}).sort_values("ROC-AUC", ascending=False)
print("\nPer-class ROC-AUC (OvR):")
print(roc_auc_df.to_string(index=False))

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Tuned XGBoost")
plt.tight_layout()
cm_path = RESULTS_DIR / "xgboost_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nSaved {cm_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
raw_feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
clean_feature_names = [name.split("__")[-1] for name in raw_feature_names]

importances = best_model.named_steps["classifier"].feature_importances_
feat_imp = (
    pd.Series(importances, index=clean_feature_names)
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
plt.title("Top 20 Feature Importances — Tuned XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
feat_imp_path = RESULTS_DIR / "xgboost_feature_importance.png"
plt.savefig(feat_imp_path, dpi=150)
plt.close()
print(f"Saved {feat_imp_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 6. SAMPLE DECISION TREE (XGBOOST) — rendered via sklearn's plot_tree
# --------------------------------------------------------------------------
def _entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

def _parse_xgb_tree(tree_json):
    nodes = {}
    def walk(node):
        nid = node["nodeid"]
        if "children" in node:
            nodes[nid] = {
                "feature": node["split"],
                "threshold": node["split_condition"],
                "yes": node["yes"],  # branch taken when condition is TRUE (feature < threshold)
                "no": node["no"],
                "leaf": False,
            }
            for child in node["children"]:
                walk(child)
        else:
            nodes[nid] = {"leaf": True}
    walk(tree_json)
    return nodes

def _compute_stats(nodes, root_id, X_df, y_enc, class_names):
    stats = {}
    def walk(nid, mask):
        counts = np.array([int(np.sum((y_enc == i) & mask)) for i in range(len(class_names))])
        samples = int(mask.sum())
        stats[nid] = {"samples": samples, "value": counts, "entropy": _entropy(counts)}
        node = nodes[nid]
        if not node["leaf"]:
            feat, thr = node["feature"], node["threshold"]
            left_mask = mask & (X_df[feat].values < thr)
            right_mask = mask & ~(X_df[feat].values < thr)
            walk(node["yes"], left_mask)
            walk(node["no"], right_mask)
    walk(root_id, np.ones(len(y_enc), dtype=bool))
    return stats

def _node_depth(nodes, nid, depth=0):
    node = nodes[nid]
    if node["leaf"]:
        return depth
    return max(_node_depth(nodes, node["yes"], depth + 1),
                _node_depth(nodes, node["no"], depth + 1))

def _get_live_node_dtype():
    # Fit a throwaway tree just to read this sklearn version's internal node dtype,
    # so we don't hardcode fields that vary across sklearn versions.
    tiny = DecisionTreeClassifier(max_depth=1).fit([[0], [1]], [0, 1])
    return tiny.tree_.__getstate__()["nodes"].dtype

def build_fake_sklearn_tree(nodes, stats, feature_names, n_classes):
    node_ids = sorted(nodes.keys())
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    node_dtype = _get_live_node_dtype()
    node_arr = np.zeros(n_nodes, dtype=node_dtype)
    values = np.zeros((n_nodes, 1, n_classes), dtype=np.float64)

    for nid in node_ids:
        i, node, s = id_map[nid], nodes[nid], stats[nid]
        node_arr[i]["n_node_samples"] = s["samples"]
        node_arr[i]["weighted_n_node_samples"] = float(s["samples"])
        node_arr[i]["impurity"] = s["entropy"]
        values[i, 0, :] = s["value"]
        if node["leaf"]:
            node_arr[i]["left_child"] = -1
            node_arr[i]["right_child"] = -1
            node_arr[i]["feature"] = -2
            node_arr[i]["threshold"] = -2.0
        else:
            node_arr[i]["left_child"] = id_map[node["yes"]]
            node_arr[i]["right_child"] = id_map[node["no"]]
            node_arr[i]["feature"] = feature_names.index(node["feature"])
            node_arr[i]["threshold"] = node["threshold"]
        if "missing_go_to_left" in node_dtype.names:
            node_arr[i]["missing_go_to_left"] = 1

    tree = SklearnTree(len(feature_names), np.array([n_classes], dtype=np.intp), 1)
    tree.__setstate__({
        "max_depth": int(max(_node_depth(nodes, nid) for nid in node_ids)),
        "node_count": n_nodes,
        "nodes": node_arr,
        "values": values,
    })
    return tree

def build_fake_tree_classifier(nodes, stats, feature_names, class_names):
    n_classes, n_features = len(class_names), len(feature_names)
    tree = build_fake_sklearn_tree(nodes, stats, feature_names, n_classes)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    clf.fit(np.zeros((n_classes, n_features)), np.arange(n_classes))  # dummy fit to populate attrs
    clf.tree_ = tree
    clf.classes_ = np.array(class_names)
    clf.n_classes_ = n_classes
    clf.n_outputs_ = 1
    clf.n_features_in_ = n_features
    return clf

# Generate the Tree Plot
xgb_estimator = best_model.named_steps["classifier"]
class_names = list(target_encoder.classes_)
X_train_transformed = best_model.named_steps["preprocessor"].transform(X_train)

booster = xgb_estimator.get_booster()
booster.feature_names = list(clean_feature_names)

tree_json = json.loads(booster.get_dump(dump_format="json")[0])
nodes = _parse_xgb_tree(tree_json)
X_df = pd.DataFrame(X_train_transformed, columns=clean_feature_names)
stats = _compute_stats(nodes, 0, X_df, y_train, class_names)

fake_clf = build_fake_tree_classifier(nodes, stats, clean_feature_names, class_names)

plt.figure(figsize=(25, 12))
plot_tree(
    fake_clf,
    feature_names=clean_feature_names,
    class_names=class_names,
    filled=False,  # sklearn's impurity-shading assumes child entropy <= parent entropy,
                    # which isn't guaranteed for XGBoost splits (unlike CART) and can
                    # produce invalid color values; keep boxes unfilled to avoid that.
    rounded=True,
    max_depth=3,  # Capped at 3 so the image is readable in a document
    fontsize=9,
)
plt.title("Sample Decision Tree from Tuned XGBoost (Tree 0)")
plt.tight_layout()
tree_path = RESULTS_DIR / "xgboost_sample_tree.png"
plt.savefig(tree_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {tree_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 7. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
model_path = MODELS_DIR / "xgboost_obesity_model.pkl"
encoder_path = MODELS_DIR / "xgboost_target_encoder.pkl"

joblib.dump(best_model, model_path)
joblib.dump(target_encoder, encoder_path)

print(f"\nSaved trained pipeline to {model_path.relative_to(PROJECT_ROOT)}")
print(f"Saved target label encoder to {encoder_path.relative_to(PROJECT_ROOT)}")

# --------------------------------------------------------------------------
# 8. LEARNING CURVE & TUNING PLOTS
# --------------------------------------------------------------------------
print("\nGenerating tuning and learning curve plots...")
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, marker='o', label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker='s', label="Validation Accuracy")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Tuned XGBoost")
plt.grid(True)
plt.legend()
plt.tight_layout()
lc_path = TUNING_DIR / "xgboost_learning_curve.png"
plt.savefig(lc_path, dpi=150)
plt.close()
print(f"Saved {lc_path.relative_to(PROJECT_ROOT)}")

# Utility for generating parameter sensitivity plots
def plot_tuning_result(param_col, title, xlabel, filename):
    plt.figure(figsize=(7,5))
    results.groupby(param_col)["mean_test_score"].mean().plot(marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Mean CV Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(TUNING_DIR / filename, dpi=150)
    plt.close()

plot_tuning_result("param_classifier__n_estimators", "Accuracy vs Number of Trees", "n_estimators", "xgb_n_estimators.png")
plot_tuning_result("param_classifier__max_depth", "Accuracy vs Max Depth", "Max Depth", "xgb_max_depth.png")
plot_tuning_result("param_classifier__learning_rate", "Accuracy vs Learning Rate", "Learning Rate", "xgb_learning_rate.png")
plot_tuning_result("param_classifier__subsample", "Accuracy vs Subsample", "Subsample", "xgb_subsample.png")
plot_tuning_result("param_classifier__colsample_bytree", "Accuracy vs Colsample By Tree", "colsample_bytree", "xgb_colsample.png")
plot_tuning_result("param_classifier__gamma", "Accuracy vs Gamma", "Gamma", "xgb_gamma.png")
plot_tuning_result("param_classifier__min_child_weight", "Accuracy vs Min Child Weight", "Min Child Weight", "xgb_child_weight.png")
plot_tuning_result("param_classifier__reg_alpha", "Accuracy vs Alpha (L1)", "reg_alpha", "xgb_alpha.png")
plot_tuning_result("param_classifier__reg_lambda", "Accuracy vs Lambda (L2)", "reg_lambda", "xgb_lambda.png")

print(f"Saved all hyperparameter sensitivity plots to {TUNING_DIR.relative_to(PROJECT_ROOT)}")