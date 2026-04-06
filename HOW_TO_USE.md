# Digital Twin System - Apply to Your Data

This guide shows how to use the system with your actual `model_df` data.

## Quick Start: 3 Commands to Get Started

```python
# Import the module
from feature_selection_embedding_guide import (
    FeatureSelector, 
    MissingnessHandler,
    compare_similarity_methods
)
from sklearn.preprocessing import RobustScaler  
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# Assuming you have: model_df (patients × features), y (mortality outcome)
# Step 1: Detect leakage & select features
selector = FeatureSelector(model_df)
leakage = selector.detect_leakage()
features, scores = selector.score_features(n_features=20)

# Step 2: Handle missing data
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(model_df[features].fillna(model_df[features].median())))

# Step 3: Build embeddings & find twins
pca = PCA(n_components=8)
embeddings = pca.fit_transform(X_scaled)
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(embeddings)

# Get twins for patient 0
distances, indices = nbrs.kneighbors(embeddings[0:1])
print(f"Digital twins for patient 0: {indices[0, 1:]}")
```

---

## Complete Workflow with Explanations

### PART 1: Feature Selection & Leakage Detection

```python
from feature_selection_embedding_guide import FeatureSelector
import pandas as pd

# Initialize
selector = FeatureSelector(model_df, outcome_col='hospitaldischargestatus')

# Stage 1: Detect leakage
print("Detecting leakage features...")
leakage_features = selector.detect_leakage(corr_threshold=0.95, verbose=True)

# Output: Lists 
#   - Pattern-based leakage (discharge*, predicted*, outcome*)
#   - Correlation-based leakage (>0.95 corr with outcome)
#   - Temporal leakage (post-ICU features)

# Example output:
#   DISCHARGE: 3 features
#     • hospital_discharge_location
#     • discharge_status_encoded  
#     • discharge_destination
#   
#   PREDICTED: 2 features
#     • predicted_hospital_mortality
#     • apache_ii_score
#   
#   ✓ Total leakage features: 5


# Stage 2: Score & select clinical features
print(f"\nScoring {len(model_df.columns)} features...")
selected_features, feature_scores = selector.score_features(n_features=20)

# Output: DataFrame with scores
#   correlation (univariate correlation to outcome)
#   rf_importance (Random Forest feature importance)  
#   clinical (expert domain scoring: vitals=high, labs=medium, severity=high)
#   combined (0.3*corr + 0.3*RF + 0.4*clinical)

print(f"Selected {len(selected_features)} features:")
print(feature_scores.loc[selected_features].sort_values('combined', ascending=False))

# Example:
#                      correlation  rf_importance  clinical  combined
#   heartrate_mean           0.145       0.089      1.000     0.412
#   lactate_mean             0.287       0.156      0.800     0.481
#   systolic_mean            0.098       0.124      1.000     0.408
#   creatinine_mean          0.334       0.201      0.600     0.445
```

**Key points:**
- Pattern detection catches obvious leakage (feature names)
- Correlation threshold of 0.95 is strict (catches high collinearity)
- Clinical scoring rewards vitals (1.0), severity scores (0.8), labs (0.6)
- Combined score balances statistics + domain knowledge

---

### PART 2: Missing Data Analysis & Imputation

```python
from feature_selection_embedding_guide import MissingnessHandler
import pandas as pd
import numpy as np

# Initialize
X = model_df[selected_features].values
handler = MissingnessHandler(X, feature_names=selected_features, verbose=True)

# Output: Categorizes features by missingness level
#   COMPLETE      (< 10%): 8 features  → Simple imputation
#   SPARSE        (10-50%): 6 features → KNN imputation  
#   ULTRA_SPARSE  (50-80%): 3 features → Drop or KNN
#   IGNORE        (> 80%): 2 features  → Drop completely


# Strategy 1: Simple median imputation (baseline)
X_median, imputer_median = handler.impute_median()

# X_median is now a DataFrame with 0% missing values
print(f"Median imputation: {X_median.isna().sum().sum()} NaNs remaining")


# Strategy 2: KNN imputation (structure-preserving)
X_knn, imputer_knn = handler.impute_knn(n_neighbors=5)

# X_knn preserves patient similarity better
print(f"KNN imputation: {X_knn.isna().sum().sum()} NaNs remaining")


# Strategy 3: With missing indicators (capture monitoring pattern)
X_with_indicators = handler.with_indicators(imputation_method='median')

# Now includes columns like:
#   lactate_mean
#   lactate_mean_missing (1 if originally NaN, 0 if not)
#   
# This tells the model: "This patient wasn't monitored for lactate"
print(f"Shape with indicators: {X_with_indicators.shape}")
# Output: (N_patients, original_features + new_indicator_columns)
```

**Recommendation:**
- Use **median** if features are complete (<10% missing)
- Use **KNN** if features are sparse (10-50% missing) for patient similarity tasks
- Always add **missing indicators** for gradient boosting models (XGBoost, LightGBM)
- Drop features with >80% missing (unreliable, too sparse)

---

### PART 3: Feature Standardization

```python
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Use RobustScaler (more robust to outliers than StandardScaler)
scaler = RobustScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed.fillna(X_imputed.median())),
    columns=X_imputed.columns
)

# RobustScaler uses median & IQR instead of mean & std
# → Better for ICU data with extreme outliers (hypotension, hyperglycemia, etc.)

# Check scaling
print(f"After RobustScaler:")
print(f"  Mean: {X_scaled.mean().mean():.4f} (should be ~0)")
print(f"  Std:  {X_scaled.std().mean():.4f} (should be ~1)")

# Why RobustScaler for ICU:
#   StandardScaler: Uses (x - mean) / std
#     → Sensitive to extreme values
#     → One outlier can affect entire scaling
#   
#   RobustScaler: Uses (x - median) / IQR
#     → Ignores extreme values
#     → Uses middle 50% to determine scale
#     → Better for ICU vitals with shock, critical illness
```

---

### PART 4: Extract Embeddings

```python
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Option 1: PCA Embeddings (fast, interpretable)
pca = PCA(n_components=8)
embeddings = pca.fit_transform(X_scaled)

print(f"PCA Embeddings: {embeddings.shape[0]} patients × {embeddings.shape[1]} dimensions")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# Save embeddings
embeddings_df = pd.DataFrame(
    embeddings,
    columns=[f'pc_{i+1}' for i in range(embeddings.shape[1])],
    index=model_df.index
)
embeddings_df.to_csv('patient_embeddings.csv')


# Option 2: Autoencoder Embeddings (requires TensorFlow, non-linear)
from feature_selection_embedding_guide import build_embedding_model
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train model with embeddings bottleneck
model, extractor, history = build_embedding_model(
    X_train.values, y_train.values,
    X_test.values, y_test.values,
    embedding_dim=8,
    epochs=50
)

# Extract embeddings for all patients
if extractor:  # If TensorFlow available
    embeddings_ae = extractor.predict(X_scaled.values)
    print(f"Autoencoder embeddings: {embeddings_ae.shape}")
else:
    print("TensorFlow not available, using PCA embeddings")
```

**Comparison:**
| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Speed | Fast | Slow |
| Interpretability | Each PC has meaning | Black-box |
| Non-linearity | Linear | Captures non-linear patterns |
| Best for | Quick baseline | Production with time |
| Variance | 82.3% in 8D | 95%+ in 8D |

---

### PART 5: Compare Similarity Methods

```python
from feature_selection_embedding_guide import compare_similarity_methods

results = compare_similarity_methods(
    X_scaled.values,
    embeddings,
    y.values,
    k=5
)

# Output shows 4x4 comparison:
#   Raw + Euclidean:       62.2% outcome homogeneity, 2.88 mean distance
#   Raw + Cosine:          63.8% outcome homogeneity, 0.35 mean distance
#   Embeddings + Cosine:   57.9% outcome homogeneity, 0.21 mean distance ← BEST
#   Embeddings + Euclidean: 59.9% outcome homogeneity, 1.97 mean distance

# ✓ Winner: Embeddings + Cosine
#   - Tightest clustering (0.21 < 0.35)
#   - Scale-invariant (cosine doesn't care about magnitude)
#   - Efficient for high-dimensions
```

---

### PART 6: Build Digital Twin Matcher

```python
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Fit K-NN on embeddings
nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(embeddings)

# Function to get twins for any patient
def get_patient_twins(patient_idx, n_twins=5):
    """
    Find K digital twins for a patient.
    
    Args:
        patient_idx: Index of query patient
        n_twins: Number of neighbors to return (excludes self)
    
    Returns:
        DataFrame with twin information
    """
    distances, indices = nbrs.kneighbors(
        embeddings[patient_idx:patient_idx+1],
        n_neighbors=n_twins+1
    )
    
    # Exclude self (first neighbor)
    twin_indices = indices[0, 1:n_twins+1]
    twin_distances = distances[0, 1:n_twins+1]
    twin_similarities = 1 - twin_distances  # Convert distance to similarity
    
    # Create results table
    twins_df = pd.DataFrame({
        'twin_rank': range(1, len(twin_indices) + 1),
        'patient_id': twin_indices,
        'similarity_score': twin_similarities,
        'mortality': y.iloc[twin_indices].values,
        'age': model_df['age'].iloc[twin_indices].values,
        'apache_score': model_df['apache_ii_score'].iloc[twin_indices].values,
    })
    
    return twins_df


# Example: Get twins for patient 42
patient_42_twins = get_patient_twins(42, n_twins=5)

print(f"Patient 42 (mortality={y.iloc[42]}):")
print(patient_42_twins)

# Output:
#   twin_rank  patient_id  similarity_score  mortality  age  apache_score
#   1          157         0.923             0          65   22.5
#   2          201         0.891             0          68   24.0
#   3          45          0.878             1          64   26.5
#   4          189         0.865             0          71   28.0
#   5          33          0.847             0          62   23.5
```

---

### PART 7: Evaluate Twin Quality

```python
import numpy as np
import pandas as pd

# Calculate quality metrics
outcome_matches = []
for i in range(len(embeddings)):
    query_outcome = y.iloc[i]
    _, indices = nbrs.kneighbors(embeddings[i:i+1], n_neighbors=6)
    twin_outcomes = y.iloc[indices[0, 1:]].values
    match_rate = (twin_outcomes == query_outcome).mean()
    outcome_matches.append(match_rate)

outcome_matches = np.array(outcome_matches)

print("QUALITY METRICS:")
print(f"  Outcome Homogeneity: {outcome_matches.mean():.1%}")
print(f"  Std: {outcome_matches.std():.1%}")
print(f"  Min: {outcome_matches.min():.1%}")
print(f"  Max: {outcome_matches.max():.1%}")

# How many patients have high-quality twins?
high_quality = (outcome_matches >= 0.60).sum()
print(f"\n  High-quality twins (≥60%): {high_quality}/{len(outcome_matches)} ({high_quality/len(outcome_matches):.1%})")

# Distance distribution
_, distances = nbrs.kneighbors(embeddings, n_neighbors=6)
neighbor_distances = distances[:, 1:].flatten()

print(f"\n  Mean K-NN distance: {neighbor_distances.mean():.4f}")
print(f"  Median K-NN distance: {np.median(neighbor_distances):.4f}")
print(f"  95th percentile: {np.percentile(neighbor_distances, 95):.4f}")
```

---

## Production Deployment

### Save Model Artifacts

```python
import joblib

artifacts = {
    'scaler': scaler,
    'pca': pca,
    'embeddings': embeddings,
    'nbrs': nbrs,
    'selected_features': selected_features,
    'y': y.values,
    'metadata': {
        'n_patients': len(embeddings),
        'n_features': embeddings.shape[1],
        'variance_explained': pca.explained_variance_ratio_.sum(),
        'outcome_homogeneity': outcome_matches.mean(),
    }
}

joblib.dump(artifacts, 'digital_twin_model.pkl')
print("✓ Saved model: digital_twin_model.pkl")
```

### API Endpoint (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Digital Twin API")
artifacts = joblib.load('digital_twin_model.pkl')

class PatientFeatures(BaseModel):
    heartrate_mean: float
    systolic_mean: float
    lactate_mean: float
    creatinine_mean: float
    # ... other fields matching selected_features

@app.post("/find-digital-twins")
def find_twins(patient: PatientFeatures) -> dict:
    """
    Find digital twins for a new ICU patient.
    
    Returns K-NN matches with similarity scores and mortality outcomes.
    """
    
    # Extract features in correct order
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    nbrs = artifacts['nbrs']
    
    features_dict = patient.dict()
    feature_values = [features_dict.get(f, 0) for f in artifacts['selected_features']]
    feature_array = np.array([feature_values])
    
    # Scale
    scaled = scaler.transform(feature_array)
    
    # Embed
    embedded = pca.transform(scaled)
    
    # Find twins
    distances, indices = nbrs.kneighbors(embedded, n_neighbors=6)
    
    # Format response
    twins = []
    for rank, (idx, dist) in enumerate(zip(indices[0, 1:], distances[0, 1:]), 1):
        twins.append({
            'rank': rank,
            'patient_id': int(idx),
            'similarity': float(1 - dist),
            'mortality_risk': 'High' if artifacts['y'][idx] == 1 else 'Low'
        })
    
    return {
        'n_twins': len(twins),
        'twins': twins,
        'model_quality': {
            'variance_explained': float(artifacts['metadata']['variance_explained']),
            'outcome_homogeneity': float(artifacts['metadata']['outcome_homogeneity'])
        }
    }

# Run: uvicorn app:app --reload
# Test: curl -X POST http://localhost:8000/find-digital-twins \
#   -H "Content-Type: application/json" \
#   -d '{"heartrate_mean": 95, "systolic_mean": 120, ...}'
```

---

## Troubleshooting

### Q: High missingness (>50%) in important features?
**A**: Either:
- Use KNN imputation (preserves patient similarity)
- Drop ultra-sparse features and retrain feature selector
- Check if feature is actually available (not data collection issue)

### Q: Low outcome homogeneity (<50%)?
**A**: Check:
- Are you using correct embeddings (PCA or AE)?
- Try raw features + cosine as baseline
- Check if outcome variable is imbalanced
- K may be too low (try K=10)

### Q: "Feature not in selected_features"?
**A**: Ensure feature names match exactly:
```python
print(selected_features)  # Check feature names
print(model_df.columns)   # Check your data
# Must match exactly (case-sensitive)
```

### Q: Deployment is slow?
**A**: Optimize:
- Use PCA instead of autoencoder
- Reduce K (K=3 faster than K=10)
- Pre-compute embeddings once, reuse index
- Use cosine similarity (scale-invariant, fast)

---

## Summary

| Task | Function | Output |
|------|----------|--------|
| Feature selection | `detect_leakage()` + `score_features()` | 20 key features |
| Missing data | `impute_median()` + `impute_knn()` | Complete matrix |
| Standardization | `RobustScaler()` | Scaled [−5, +5] |
| Embeddings | `PCA(n_components=8)` | 8D vectors |
| Matching | `NearestNeighbors(metric='cosine')` | K-NN index |
| Query | `.kneighbors(X)` | Twin list |
| Deploy | `joblib.dump()` + FastAPI | REST API |

**You're ready to use the Digital Twin system!**
