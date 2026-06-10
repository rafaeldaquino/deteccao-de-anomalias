import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def carregar_dados(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def aplicar_log(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    df[f"{coluna}_log"] = np.log1p(df[coluna])
    df = df.drop(columns=[coluna])
    return df

def std_scaler(df: pd.DataFrame, coluna: str, scaler=None):
    df_copy = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df_copy[f"{coluna}_scaler"] = scaler.fit_transform(df_copy[[coluna]])
    else:
        df_copy[f"{coluna}_scaler"] = scaler.transform(df_copy[[coluna]])
        
    df_copy = df_copy.drop(columns=[coluna])
    return df_copy, scaler

def executar_undersampling(df: pd.DataFrame, alvo: str = "Class") -> pd.DataFrame:
    fraudes = df[df[alvo] == 1]
    normais = df[df[alvo] == 0].sample(len(fraudes), random_state=42)
    return pd.concat([fraudes, normais])

def executar_smote(X: pd.DataFrame, y: pd.Series):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def preparar_dados_treino_teste(df: pd.DataFrame, alvo: str = "Class", test_size: float = 0.2):
    X = df.drop(columns=[alvo])
    y = df[alvo]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test