import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from preprocessing import (
    inspect_tif,
    remove_nodata,
    raster_to_dataframe,
    get_reference_file,
    align_rasters,
    merge_dataframes
)

from model_preparation import (plot_correlation_matrix, evaluate_model, plot_feature_importance)


# ============================================================
# 1. LIST THE FILES
# ============================================================
filenames = glob.glob("wildfire_probailitiy/raster_input/*.tif")
print(f"{len(filenames)} files found :\n")
for f in filenames:
    print(f"  - {f}")

# ============================================================
# 2. INSPECT THE FILES
# ============================================================
print("\n=== INSPECTION ===")
for f in filenames:
    print(f"\n--- {f} ---")
    inspect_tif(f)

# ============================================================
# 3. FIND THE REFERENCE FILE
# ============================================================
print("\n=== RESOLUTION ===")
reference_file, lowest_res = get_reference_file(filenames)

# ============================================================
# 4. ALIGN ALL FILES
# ============================================================
print("\n=== ALIGNEMENT ===")
aligned_data = align_rasters(filenames, reference_file=reference_file)

# ============================================================
# 5. RETRIEVE THE TRANSFORM AND NO_DATA REFERENCE
# ============================================================
with rasterio.open(reference_file) as ref:
    ref_transform = ref.transform
    ref_nodata = ref.nodata
    ref_height    = ref.height      
    ref_width     = ref.width       
    ref_crs       = ref.crs         

# ============================================================
# 6. TRANSFORM INTO DATAFRAMES
# ============================================================
print("\n=== CONVERSION INTO DATAFRAMES ===")
dataframes = {}
for name, data in aligned_data.items():
    mask= remove_nodata(data[0], nodata_value=ref_nodata)
    df = raster_to_dataframe(data[0], ref_transform, mask=mask)
    dataframes[name] = df
    print(f"✅ {name} — shape: {df.shape}")

# ============================================================
# 7. MERGE THE DATAFRAMES
# ============================================================
print("\n=== FUSION ===")
df_final = merge_dataframes(dataframes)

print("\n=== FINAL RESULT ===")
print(df_final.shape)
print(df_final.head())
print(df_final.describe())

print(f"Nombre de lignes : {df_final.shape[0]:,}")
print(f"Nombre de colonnes : {df_final.shape[1]}")
print(f"Mémoire utilisée : {df_final.memory_usage().sum() / 1e9:.2f} GB")

# ============================================================
# 8. CORRELATION MATRIX
# ============================================================
print("\n=== CORRELATION ===")
corr = plot_correlation_matrix(df_final)

print("\nCorrelation with the target variable :")
print(corr['wildfires_25yrs'].sort_values(ascending=False))

# ============================================================
# 9. FINAL CLEANING
# ============================================================
print("\nShape before cleaning :", df_final.shape)
df_final = df_final[~(df_final == -9999).any(axis=1)]
df_final = df_final.dropna()
print("Shape after cleaning :", df_final.shape)

print("\nNew distribution of wildfires_25yrs :")
print(df_final['wildfires_25yrs'].describe())

# ============================================================
# 10. FEATURE SELECTION
# ============================================================

features = [c for c in df_final.columns if c not in ['x', 'y', 'wildfires_25yrs']]


X = df_final[features]
y = df_final['wildfires_25yrs']

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"\nDistribution of y :")
print(y.describe())

# Checking
print("NaN dans X :", X.isna().sum().sum())
print("NaN dans y :", y.isna().sum())
print("Inf dans X :", np.isinf(X).sum().sum())
print("Inf dans y :", np.isinf(y).sum())
print(X.dtypes)
print(y.dtype)


# ============================================================
# 10. LOG TRANSFORMATION OF THE TARGET VARIABLE
# ============================================================
import numpy as np

# Check the distribution before transformation
print("Distribution of y before log transformation :")
print(y.describe())

# Apply log transformation (adding epsilon to avoid log(0))
epsilon = 1e-6
y_log = np.log(y + epsilon)

print("\nDistribution of y after log transformation :")
print(y_log.describe())

# Visualize the distribution before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(y, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of y — Original')
axes[0].set_xlabel('Wildfire Probability')
axes[0].set_ylabel('Count')

axes[1].hist(y_log, bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('Distribution of y — Log Transformed')
axes[1].set_xlabel('Log(Wildfire Probability)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('y_distribution.png', dpi=150)
plt.close()
print("✅ Distribution plot saved : y_distribution.png")

# ============================================================
# 11. SPLIT TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log,         
    test_size=0.2,
    random_state=42
)

print(f"\nTrain : {X_train.shape}")
print(f"Test  : {X_test.shape}")

# ============================================================
# 12. TRAINING
# ============================================================
param_grid = {
    'n_estimators': [400, 800, 1000],
    'max_depth': [4,6,8],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='r2',
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters :", grid_search.best_params_)
print("Best R²         :", grid_search.best_score_)

model = grid_search.best_estimator_

# ============================================================
# 13. EVALUATION
# ============================================================

# Predict in log space
y_pred_log = model.predict(X_test)

# Convert back to original scale
y_pred_original = np.exp(y_pred_log) - epsilon
y_test_original = np.exp(y_test) - epsilon

# Evaluate on original scale
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae  = mean_absolute_error(y_test_original, y_pred_original)
r2   = r2_score(y_test_original, y_pred_original)

print("\n=== Results (original scale) ===")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================
plot_feature_importance(model, features)

# ============================================================
# 15. PREDICTION MAP
# ============================================================

# Predict on full dataset in log space then convert back
y_pred_full_log      = model.predict(X)
y_pred_full_original = np.exp(y_pred_full_log) - epsilon

# Rebuild empty grid at reference dimensions
grid = np.full((ref_height, ref_width), np.nan)

# Convert x/y coordinates back to row/col indices
rows_idx, cols_idx = rasterio.transform.rowcol(
    ref_transform,
    df_final['x'].values,
    df_final['y'].values
)

grid[rows_idx, cols_idx] = y_pred_full_original

# Display the map
plt.figure(figsize=(12, 8))
plt.imshow(grid, cmap='YlOrRd', interpolation='none')
plt.colorbar(label='Wildfire Probability')
plt.title('Prediction Map — Wildfire Probability')
plt.axis('off')
plt.tight_layout()
plt.savefig('wildfire_prediction_map.png', dpi=150)
plt.close()
print("✅ Map saved : wildfire_prediction_map.png")

# Save as .tif for QGIS
with rasterio.open(
    'wildfire_prediction.tif',
    'w',
    driver='GTiff',
    height=ref_height,
    width=ref_width,
    count=1,
    dtype='float32',
    crs=ref_crs,
    transform=ref_transform
) as dst:
    dst.write(grid.astype('float32'), 1)
print("✅ Raster saved : wildfire_prediction.tif")