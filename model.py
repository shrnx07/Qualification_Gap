"""
model.py — Model training and evaluation
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix

FEATURES = ["loan_amnt", "annual_inc", "dti", "emp_length_yrs"]
TARGET   = "outcome"

def load_features(con, test_size=0.2, random_state=42):
    df = con.execute(f"""
        SELECT {', '.join(FEATURES)}, {TARGET}
        FROM applicants WHERE emp_length_yrs >= 0
    """).df()
    X = df[FEATURES]; y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    print("Models trained.")
    return lr, dt, scaler

def evaluate_models(lr, dt, scaler, X_test, y_test):
    results = []
    for name, model, Xt in [("Logistic Regression", lr, scaler.transform(X_test)),
                              ("Decision Tree",       dt, X_test)]:
        preds = model.predict(Xt)
        proba = model.predict_proba(Xt)[:, 1]
        rep   = classification_report(y_test, preds, output_dict=True)
        auc   = roc_auc_score(y_test, proba)
        results.append({"Model": name, "Accuracy": round(rep["accuracy"],4),
                        "Precision": round(rep["weighted avg"]["precision"],4),
                        "Recall": round(rep["weighted avg"]["recall"],4),
                        "AUC": round(auc,4)})
        print(f"{name}: AUC={auc:.4f}")
    return pd.DataFrame(results)
