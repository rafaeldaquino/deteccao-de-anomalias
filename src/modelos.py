import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import shap

def treinar_regressao_logistica(X_train, y_train, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model

def treinar_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

def treinar_xgboost(X_train, y_train, scale_pos_weight=10):
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb

def otimizar_xgboost_grid(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [50, 100]
    }
    grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid,
        scoring="recall",
        cv=3
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def avaliar_modelo(model, X_test, y_test, custom_threshold=None):
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.predict(X_test)
        
    if custom_threshold:
        y_pred = (y_probs >= custom_threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        
    print("--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred))
    
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, y_probs)
        print(f"ROC AUC Score: {auc:.4f}")

def plotar_curvas(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(fpr, tpr, color='blue')
    ax[0].set_title("ROC Curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    
    ax[1].plot(recall, precision, color='green')
    ax[1].set_title("Precision-Recall Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    
    plt.show()

def plotar_importancia_variaveis(model, X_train):
   
    importancias = model.feature_importances_
    importancias_df = pd.Series(importancias, index=X_train.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    importancias_df.plot(kind='bar', color='blue')
    
    # 4. Customizar títulos e labels
    plt.title("Importância de Variáveis - XGBoost")
    plt.tight_layout()
    
    plt.show()

def plotar_shap(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])
    shap.plots.bar(shap_values)