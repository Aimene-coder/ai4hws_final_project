import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def plot_correlation_matrix(df):
    cols = [c for c in df.columns if c not in ['x', 'y']]
    corr = df[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")  
    plt.close()                             
    print("✅ Matrix saved : correlation_matrix.png")

    return corr




def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    print("=== Results===")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    
    return y_pred


# The "plot_feature_importance" function plots the importance of the predictive features for the XGBoost model

def plot_feature_importance(model, features):
    
    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importances.plot(kind='barh')
    plt.title("Importance of the variables")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png")  
    plt.close()                             
    
    print("✅ Graph saved : feature_importance.png")
    print("\nImportance of variables :")
    print(importances.sort_values(ascending=False))
