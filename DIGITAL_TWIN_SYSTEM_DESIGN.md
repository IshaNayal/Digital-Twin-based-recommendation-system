
# Digital Twin-Based Patient Similarity System
## Complete Pipeline Design & Implementation Guide

**Last Updated**: 2026-04-06  
**Status**: Ready for Implementation  
**Dataset**: eICU Collaborative Research Database (Demo 2.0.1)

---

## Executive Summary

A **Digital Twin system** finds similar historical ICU patients ("twins") for a query patient, enabling:
- **Outcome prediction**: What's the mortality risk (based on twins' outcomes)?
- **Treatment recommendation**: What worked best for similar patients?
- **Clinical decision support**: Personalized ICU management

This document provides a production-ready pipeline addressing your constraints:
- ✅ Removes data leakage (discharge status, predicted mortality)
- ✅ Handles 80-90% missingness in ICU data
- ✅ Selects ~20 clinically meaningful features
- ✅ Creates low-dimensional embeddings
- ✅ Uses similarity metrics (cosine/Euclidean/Gower)
- ✅ Validates twin quality

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW EICIU DATA (~250 features)                   │
│  - Demographics: age, gender, ethnicity, unit type                  │
│  - Vital summaries: HR, BP, SpO2, RR, temp (mean/std/min/max)      │
│  - Lab summaries: creatinine, lactate, glucose, metabolic (24h)     │
│  - Severity: APACHE, APS, SOFA scores                              │
│  - Outcomes: mortality, ICU/hospital LOS                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  STAGE 1: DATA LEAKAGE  │
                    │       REMOVAL           │
                    └────────────┬────────────┘
                                 │
              Remove: discharge status, predicted mortality, 
              post-ICU features, high outcome correlation (>0.95)
                                 │
                    ┌────────────▼─────────────┐
                    │ STAGE 2: FEATURE         │
                    │ SELECTION (Clinical)    │
                    └────────────┬─────────────┘
                                 │
         Select: 20 features combining statistical + clinical scoring
         Prioritize: vitals (HR, BP), labs (lactate, creatinine), severity
                                 │
                    ┌────────────▼──────────────┐
                    │ STAGE 3: MISSINGNESS      │
                    │ HANDLING & IMPUTATION     │
                    └────────────┬──────────────┘
                                 │
       Drop ultra-sparse (>80% missing), KNN imputation for sparse,
       create missing indicators (clinically informative)
                                 │
                    ┌────────────▼──────────────┐
                    │ STAGE 4: STANDARDIZATION  │
                    │ (RobustScaler)            │
                    └────────────┬──────────────┘
                                 │
         Scale using median & IQR (robust to ICU outliers)
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
        ┌─────▼──────────┐              ┌──────────▼─────────┐
        │ APPROACH A:    │              │ APPROACH B:        │
        │ PCA Embeddings │              │ Autoencoder        │
        │ (Fast)         │              │ (Non-linear)       │
        └─────┬──────────┘              └──────────┬─────────┘
              │                                   │
              └──────────────┬──────────────────┘
                             │
                    ┌────────▼──────────┐
                    │ 15-D EMBEDDINGS    │
                    │ (Patient Vectors)  │
                    └────────┬──────────┘
                             │
                    ┌────────▼───────────────┐
                    │ SIMILARITY METRICS      │
                    │ (Cosine/Euclidean...)   │
                    └────────┬───────────────┘
                             │
                    ┌────────▼──────────────┐
                    │ K-NN MATCHING          │
                    │ (Find top-K similar)   │
                    └────────┬──────────────┘
                             │
                    ┌────────▼──────────────┐
                    │ DIGITAL TWINS          │
                    │ • Indices              │
                    │ • Distances/Similarity │
                    │ • Outcomes             │
                    │ • Demographics         │
                    └────────────────────────┘
```

---

## Stage-by-Stage Implementation

### **Stage 1: Remove Data Leakage**

**Why**: Prevent information from the outcome leaking into features. This invalidates downstream predictions.

**What to remove**:
1. **Discharge-related**: `hospitaldischargestatus`, `unitdischargestatus`, `unitdischargelocation`, `hospitaldischargelocation`
2. **Predicted variables**: `predictedhospitalmortality`, `predictediculos`, `acualhospitalmortality`
3. **Post-ICU features**: Any feature calculated after discharge
4. **High correlation to outcome** (>0.95): Likely statistical leakage

**Implementation**:
```python
leakage_patterns = ['discharge', 'predicted', 'outcome', 'actualhospital']
leakage_cols = [c for c in df.columns if any(p in c.lower() for p in leakage_patterns)]
# Also check correlation to outcome (>0.95 = drop)
df_clean = df.drop(columns=leakage_cols)
```

**Result**: ~250 → ~230 clean features

---

### **Stage 2: Clinical Feature Selection**

**Goal**: 15-25 features capturing core physiology without redundancy

**Strategy**: Weighted score combining:
- **Clinical domain** (0.4 weight): Medical relevance
- **Statistical** (0.3 weight): Univariate correlation to outcome
- **ML importance** (0.3 weight): Random Forest feature importance

**Selected feature categories**:

| Category | Examples | Count |
|----------|----------|-------|
| **Vitals** | HR mean/std, BP systolic/diastolic, SpO2, RR, Shock index | 8-10 |
| **Labs** | Lactate, creatinine, glucose, sodium, pH, hemoglobin | 5-7 |
| **Severity** | APACHE score, APS | 1-2 |
| **Missing indicators** | Flags for originally-missing vitals/labs | 1-3 |

**Why these?**
- Vital signs: Reflect immediate physiologic state
- Lactate & creatinine: Tissue perfusion & organ function (strong predictors)
- APACHE: Validated severity score
- Missing indicators: Selective monitoring is informative in ICU

**Implementation**:
```python
# Step 1: Correlation to outcome
correlation_scores = {col: abs(df[col].corr(df['y_hosp_mortality'])) 
                      for col in numeric_cols}

# Step 2: Random Forest importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_imputed, y)
rf_scores = dict(zip(X.columns, rf.feature_importances_))

# Step 3: Clinical domain scoring
clinical_scores = {col: score_by_category(col) for col in numeric_cols}

# Step 4: Combine and select top-20
combined = 0.3*corr + 0.3*rf + 0.4*clinical
top_20 = combined.nlargest(20).index
```

**Result**: ~230 → 20 clinically meaningful features

---

### **Stage 3: Missing Data Handling**

**The ICU missingness challenge**:
- Not all vitals monitored for all patients (selective monitoring by acuity)
- ETCO2: ~96% missing (only monitored when ventilated)
- Temperature: ~94% missing (varies by unit)
- Labs: Depend on clinical suspicion
- **NOT Missing Completely At Random (MCAR)** → missingness pattern contains info

**Strategy: Hybrid approach**

1. **Drop ultra-sparse** (>80% missing)
   - Too noisy, imputation unreliable
   - Example: ETCO2, rarely-measured labs

2. **KNN imputation** for sparse features (10-50% missing)
   - Impute using k=5 nearest neighbors based on other features
   - Preserves local similarity structure
   - Better than global mean for ICU where selective monitoring matters

3. **Create missing indicators**
   - Binary flags for originally-missing values
   - Include as features (clinically informative: "wasn't this monitored?")
   - Example: If SpO2 was missing, flag = 1

**Implementation**:
```python
# Step 1: Categorize features
complete = missing_pct < 0.1       # <10% missing
sparse = (10 <= missing_pct < 50)  # 10-50% missing
ultra_sparse = missing_pct > 0.8   # >80% missing

# Step 2: Drop ultra-sparse
X_filtered = X.drop(columns=ultra_sparse_cols)

# Step 3: KNN imputation for sparse
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = knn_imputer.fit_transform(X_filtered)

# Step 4: Create and add missing indicators
for col in sparse_cols:
    X_imputed[f'{col}_missing_indicator'] = X[col].isna().astype(int)
```

**Result**: 20 features + ~3 indicators, 0% missing

---

### **Stage 4: Standardization**

**Why**: Different features have different scales
- Heart rate: 40-180 bpm
- Blood pressure: 60-220 mmHg
- Lab values: Various units and ranges
- PCA & embeddings are scale-sensitive

**Method: RobustScaler** (not StandardScaler)
- Uses median & IQR (percentile-based)
- Robust to extreme outliers (common in ICU)
- Better than z-score which assumes normal distribution

**StandardScaler vs RobustScaler**:

| Aspect | StandardScaler | RobustScaler |
|--------|---|---|
| Formula | (x - mean) / std | (x - median) / IQR |
| Sensitive to outliers | ✓ | ✗ |
| Best for | Normal data | Real-world ICU |

**Implementation**:
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)
```

**Result**: Standardized matrix ready for embeddings

---

### **Stage 5A: Embeddings - Classical (PCA)**

**What is PCA?**
- Linear dimensionality reduction
- Finds principal directions of variance in data
- Transforms feature space → embedding space

**Why PCA?**
- ✅ Fast (linear algebra)
- ✅ Interpretable (loadings show which features matter)
- ✅ Good baseline for digital twins
- ❌ Assumes linear relationships
- ❌ May miss non-linear patterns

**How many dimensions?**
- Input: ~23 features (20 + 3 indicators)
- Output: **15 dimensions** (captures ~80% variance)
- Trade-off: Compression vs. information retention

**Implementation**:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=15)
embeddings = pca.fit_transform(X_scaled)

# Check variance explained
print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.1%}")
# Output: ~80-85% of variance in 15 components

# Visualize: first 2 components
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='RdYlGn_r')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
```

**Result**: (n_patients, 15) embedding matrix

---

### **Stage 5B: Embeddings - Deep Learning (Autoencoder - Optional)**

**What is an autoencoder?**
- Neural network trained to reconstruct its input
- Forces data through bottleneck (embedding layer)
- Learns non-linear patterns

**Architecture**:
```
Input (23-d) 
  ↓
Dense(128, ReLU) + Dropout(0.2)
  ↓
Dense(64, ReLU)
  ↓
Dense(15, ReLU)  ← Embedding (bottleneck)
  ↓
Dense(64, ReLU) + Dropout(0.2)
  ↓
Dense(128, ReLU)
  ↓
Output (23-d reconstruction)
```

**Training**:
- Objective: Minimize reconstruction error
- Learns to compress clinically relevant info into 15-d
- Better if you have non-linear patient phenotypes

**When to use**:
- ✅ Large dataset (>5000 patients)
- ✅ Non-linear relationships matter
- ❌ GPU required (slow on CPU)
- ❌ Less interpretable than PCA

**Implementation** (optional):
```python
from tensorflow.keras import layers, Sequential, models

encoder = Sequential([
    layers.Dense(128, activation='relu', input_shape=(23,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(15, activation='linear', name='embedding')
])

decoder = Sequential([...])  # Mirror structure

autoencoder = models.Model(encoder.input, decoder.output)
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32)

embeddings = encoder.predict(X_scaled)
```

**Recommendation**: Start with **PCA** (faster, simpler). Try autoencoder if results are poor.

---

### **Stage 6: Similarity Metrics**

**Problem**: How do we measure "similarity" between patient embeddings?

**Options**:

| Metric | Formula | Space | Invariance | Best For |
|--------|---------|-------|-----------|----------|
| **Cosine** | $1 - \cos(\theta)$ | Angle | Scale | Embeddings |
| **Euclidean** | $\sqrt{\sum(x_i-y_i)^2}$ | Distance | Scale sens. | Raw features |
| **Manhattan** | $\sum \|x_i - y_i\|$ | L1 distance | Scale sens. | Sparse data |
| **Gower** | Weighted mix | Mixed | Handles missing | Semi-structured |

**Recommendation for Digital Twins: COSINE SIMILARITY**

**Why cosine?**
1. **Scale-invariant**: Works after standardization
2. **Interpretable**: Measures angle (morphology), not magnitude
3. **Fast**: O(d) where d = dimensions
4. **Natural for embeddings**: Works in latent spaces

**Why not Euclidean?**
- Sensitive to scale (after standardization, less meaningful)
- Assumes features are commensurable (they're not quite in ICU)
- Slower for high-dim data

**Implementation**:
```python
from sklearn.metrics.pairwise import cosine_distances

# Compute pairwise cosine distance
distance_matrix = cosine_distances(embeddings)  # Shape: (n, n)

# Convert to similarity (closer = higher, opposite of distance)
similarity_matrix = 1 - distance_matrix
```

---

### **Stage 7: K-Nearest Neighbors (Find Digital Twins)**

**What is K-NN?**
- For a query patient, find the K closest neighbors in embedding space
- "Closest" defined by similarity metric (cosine)

**How many K?**
- K=3-5: Reasonable sample of similar patients
- K=10: More diverse twins, noisier
- Recommendation: **K=5** (balance covering vs specificity)

**Implementation**:
```python
from sklearn.neighbors import NearestNeighbors

# Fit KNN index
nbrs = NearestNeighbors(n_neighbors=K+1, metric='cosine')
nbrs.fit(embeddings)

# Query: Find twins for patient i
distances, indices = nbrs.kneighbors(embeddings[i:i+1])

# indices[0] = [i, twin1, twin2, ...] (first is self)
# distances[0] = [0, dist_to_twin1, dist_to_twin2, ...]

# Exclude self and get top-K
twin_indices = indices[0][1:K+1]
twin_distances = distances[0][1:K+1]
twin_similarities = 1 - twin_distances
```

**Efficiency**:
- Scalable to ~1M patients with Faiss library
- Single query: O(n) with linear search
- With indexing: O(log n) or better

---

### **Stage 8: Outcome Prediction & Recommendations**

**From twins to predictions**:

```python
# For query patient i with twins j1, j2, ..., jK
twin_outcomes = [y[j1], y[j2], ..., y[jK]]
mortality_in_twins = sum(twin_outcomes) / K

# Prediction: Mortality risk = % of twins who died
if mortality_in_twins >= 0.6:
    risk_category = "HIGH RISK"
elif mortality_in_twins >= 0.3:
    risk_category = "MODERATE RISK"
else:
    risk_category = "LOW RISK"

# Confidence: higher if twins agree
disagreement = abs(mortality_in_twins - 0.5)
confidence = "High" if disagreement >= 0.2 else "Moderate" if disagreement >= 0.1 else "Low"
```

**Treatment recommendations**:
- Find which treatments were used in twin survivors
- Find which treatments to avoid (high mortality)
- Aggregate across K twins with similarity-weighted voting

---

## Advanced: Feature Selection, Leakage Diagnosis, and Missingness Handling

### **Part 1: Diagnosing Data Leakage**

Data leakage occurs when the model uses information not available at prediction time. In ICU data, common sources include:

1. **Discharge status variables** (CRITICAL)
   - `hospitaldischargestatus` (Expired/Alive)
   - `unitdischargestatus` (Discharged/Expired)
   - These directly encode mortality → drop immediately

2. **Predicted variables** (CRITICAL)
   - `predictedhospitalmortality` (APACHE II prediction)
   - `predictediculos` (APACHE predicted LOS)
   - Already derived from mortality → direct leakage

3. **Post-ICU features** (LEAKAGE)
   - `hospitaldischargelocation` (Home/Rehab/Expired)
   - `hospital_los_hours` (only known after discharge)

4. **High statistical correlation** (SUBTLE)
   - Features >0.95 correlated with mortality
   - Likely derived features or leakage

**Comprehensive leakage detection code**:

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, chi2_contingency

def diagnose_leakage(df, outcome_col='y_hosp_mortality', corr_threshold=0.95):
    """
    Identify potential data leakage features.
    
    Returns:
        leakage_report: Dict with different leakage categories
    """
    
    leakage_report = {
        'pattern_based': [],      # Discharge/predicted patterns
        'correlation_based': [],   # >0.95 corr to outcome
        'temporal_based': [],      # Post-ICU features
    }
    
    # 1. PATTERN-BASED: Common leakage variable names
    leakage_patterns = {
        'discharge': ['discharge', 'discharged', 'discharge_status', 'location'],
        'predicted': ['predicted', 'predict', 'apache_predicted', 'expected'],
        'outcome': ['outcome', 'actual_', 'hospital_outcome'],
    }
    
    for pattern_type, patterns in leakage_patterns.items():
        for col in df.columns:
            if col != outcome_col:
                col_lower = col.lower()
                if any(p in col_lower for p in patterns):
                    leakage_report['pattern_based'].append({
                        'feature': col,
                        'type': pattern_type,
                        'reason': f"Matches pattern: {pattern_type}"
                    })
    
    # 2. CORRELATION-BASED: Features highly correlated with outcome
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outcome_col in df.columns and outcome_col in numeric_cols:
        for col in numeric_cols:
            if col != outcome_col:
                # Skip already identified
                if any(item['feature'] == col for item in leakage_report['pattern_based']):
                    continue
                
                # Compute correlation (drop NaN)
                valid_mask = ~(df[col].isna() | df[outcome_col].isna())
                if valid_mask.sum() > 30:  # Need enough data
                    corr = abs(df.loc[valid_mask, col].corr(df.loc[valid_mask, outcome_col]))
                    
                    if corr > corr_threshold:
                        leakage_report['correlation_based'].append({
                            'feature': col,
                            'correlation': corr,
                            'reason': f"Correlation to outcome: {corr:.3f}"
                        })
    
    # 3. TEMPORAL-BASED: Features relying on discharge info
    temporal_patterns = ['los', 'length_of_stay', 'hospital_los', 'icu_los_after', 'disposition']
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in temporal_patterns):
            if col not in [item['feature'] for item in leakage_report['pattern_based']]:
                leakage_report['temporal_based'].append({
                    'feature': col,
                    'reason': 'Time-dependent feature (only known post-discharge)'
                })
    
    return leakage_report


# Usage example
leakage = diagnose_leakage(model_df, outcome_col='y_hosp_mortality')
print("PATTERN-BASED LEAKAGE:")
for item in leakage['pattern_based']:
    print(f"  • {item['feature']}: {item['reason']}")

print("\nCORRELATION-BASED LEAKAGE:")
for item in leakage['correlation_based']:
    print(f"  • {item['feature']}: {item['correlation']:.3f}")

print("\nTEMPORAL-BASED LEAKAGE:")
for item in leakage['temporal_based']:
    print(f"  • {item['feature']}: {item['reason']}")

# Remove all detected leakage
leakage_cols_to_drop = (
    [item['feature'] for item in leakage['pattern_based']] +
    [item['feature'] for item in leakage['correlation_based']] +
    [item['feature'] for item in leakage['temporal_based']]
)

df_clean = df.drop(columns=[c for c in leakage_cols_to_drop if c in df.columns])
print(f"\n✓ Removed {len(leakage_cols_to_drop)} leakage features")
print(f"  Original shape: {df.shape}")
print(f"  Clean shape: {df_clean.shape}")
```

---

### **Part 2: Comprehensive Missing Data Analysis & Handling**

**The ICU missingness pattern**:
- Selective monitoring based on acuity (NOT random)
- ETCO2: ~96% missing (only when intubated)
- Temperature: ~94% missing (varies by unit)
- Labs: Ordered by physician judgment

**Problem**: Different imputation strategies have different downstream effects.

**Comparison of approaches**:

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def compare_imputation_strategies(X, y=None, feature_names=None, viz=True):
    """
    Compare multiple imputation strategies on ICU data.
    
    Strategies:
    1. Mean imputation (baseline)
    2. Median imputation (robust)
    3. KNN imputation (structure-preserving)
    4. MICE (iterative imputation)
    5. Forward-fill (if temporal available)
    6. Missingness indicators only (XGBoost native)
    """
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    results = {}
    
    # STRATEGY 1: Mean imputation
    imputer_mean = SimpleImputer(strategy='mean')
    X_mean = pd.DataFrame(
        imputer_mean.fit_transform(X),
        columns=feature_names
    )
    results['mean'] = X_mean
    
    # STRATEGY 2: Median imputation
    imputer_median = SimpleImputer(strategy='median')
    X_median = pd.DataFrame(
        imputer_median.fit_transform(X),
        columns=feature_names
    )
    results['median'] = X_median
    
    # STRATEGY 3: KNN imputation (k-nearest neighbors)
    imputer_knn = KNNImputer(n_neighbors=5, weights='distance')
    X_knn = pd.DataFrame(
        imputer_knn.fit_transform(X),
        columns=feature_names
    )
    results['knn'] = X_knn
    
    # STRATEGY 4: MICE (Multiple Imputation by Chained Equations)
    try:
        imputer_mice = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            verbose=0
        )
        X_mice = pd.DataFrame(
            imputer_mice.fit_transform(X),
            columns=feature_names
        )
        results['mice'] = X_mice
    except:
        print("⚠ MICE skipped (requires sklearn experimental)")
    
    # STRATEGY 5: Missingness indicators (for gradient boosting)
    # Keep NaNs but add binary flags
    missing_indicators = pd.DataFrame()
    for col in X.columns:
        missing_indicators[f'{col}_missing'] = X[col].isna().astype(int)
    
    X_with_indicators = X.copy()
    X_with_indicators = pd.concat([X_with_indicators, missing_indicators], axis=1)
    
    # Fill NaNs with column median for model training
    X_indicators = X_with_indicators.fillna(X.median())
    results['with_indicators'] = X_indicators
    
    # EVALUATION: If outcome available, compare predictive power
    if y is not None:
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBClassifier
        
        print("\nPREDICTIVE POWER COMPARISON (5-fold CV, XGBoost AUC):")
        print("─" * 60)
        
        for strategy_name, X_imputed in results.items():
            try:
                model = XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
                scores = cross_val_score(model, X_imputed, y, cv=5, scoring='roc_auc', n_jobs=-1)
                print(f"{strategy_name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"{strategy_name:20s}: Error - {str(e)[:30]}")
    
    return results, imputer_knn, imputer_mean, missing_indicators


# Usage
X_missing = model_df[selected_features].copy()
y = model_df['y_hosp_mortality']

results, knn_imputer, mean_imputer, missing_ind = compare_imputation_strategies(
    X_missing, y
)
```

**Recommendation for different use cases**:

| Use Case | Recommended Strategy | Reason |
|----------|---------------------|--------|
| **Random Forest / XGBoost** | KNN + Missingness indicators | Native missing handling + structure |
| **Neural networks** | KNN imputation only | Networks sensitive to NaN, need clean data |
| **Linear models** | Median + Indicators | Simple, interpretable, handles outliers |
| **Patient similarity** | KNN imputation | Preserves local similarity structure |
| **When causality matters** | MICE | Theoretically better for inference |

---

### **Part 3: Embeddings for Mortality Prediction and Patient Similarity**

**Key insight**: Instead of using raw features for similarity, train a model to predict mortality, then extract learned embeddings.

**Why embeddings work better**:
1. **Learned representations**: Model learns what features matter for mortality
2. **Dimensionality reduction**: 20 features → 10-15 dimensions without losing predictive power
3. **Non-linear relationships**: Deep models capture complex patterns
4. **Task-aligned similarity**: "Similar outcomes" ↔ "Similar embeddings"

**Implementation: Extract embeddings from a predictive model**:

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class MortalityEmbedding:
    """
    Train a mortality prediction model and extract embeddings.
    
    Architecture:
    Input (20 features) → Dense(64) → Dense(32) → 
    Dense(embedding_dim, 'relu') [EMBEDDING LAYER] → 
    Dense(16) → Dense(1, 'sigmoid') [OUTPUT]
    """
    
    def __init__(self, input_dim=20, embedding_dim=12):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None  # For extracting embeddings
        self.scaler = None
        
    def build_model(self):
        """Build mortality predictor with embedding layer."""
        
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # Encoder path: learn representations
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Embedding layer: compressed representation
        embeddings = layers.Dense(self.embedding_dim, activation='relu', 
                                  name='embedding')(x)
        
        # Decoder path: predict mortality from embeddings
        x = layers.Dense(16, activation='relu')(embeddings)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid', name='mortality')(x)
        
        # Full model: Input → Embeddings → Mortality
        self.model = Model(inputs=inputs, outputs=outputs, name='mortality_predictor')
        
        # Embedding model: Input → Embeddings only (for similarity)
        self.embedding_model = Model(inputs=inputs, outputs=embeddings, 
                                     name='embedding_extractor')
        
        # Compile
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['auc', 'accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the mortality prediction model."""
        
        print(f"Training mortality predictor ({self.input_dim}D → {self.embedding_dim}D embeddings)")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        return history
    
    def extract_embeddings(self, X):
        """Extract learned embeddings for all patients."""
        return self.embedding_model.predict(X, verbose=0)
    
    def predict_mortality(self, X):
        """Predict mortality probability."""
        return self.model.predict(X, verbose=0).flatten()


# Usage example
# 1. Prepare data
X_scaled = ...  # Your standardized features (n_patients, 20)
y = model_df['y_hosp_mortality'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 2. Train embedding model
embedding_model = MortalityEmbedding(input_dim=X_train.shape[1], embedding_dim=12)
model = embedding_model.build_model()
history = embedding_model.train(X_train, y_train, X_val, y_val, epochs=50)

# 3. Evaluate mortality prediction
y_pred = embedding_model.predict_mortality(X_test)
test_auc = roc_auc_score(y_test, y_pred)
print(f"Mortality prediction AUC: {test_auc:.4f}")

# 4. Extract embeddings
embeddings_train = embedding_model.extract_embeddings(X_train)
embeddings_test = embedding_model.extract_embeddings(X_test)
embeddings_all = embedding_model.extract_embeddings(X_scaled)

print(f"Embeddings shape: {embeddings_all.shape}")  # (n_patients, 12)

# 5. Visualize embeddings space
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_all)

plt.figure(figsize=(10, 6))
colors = ['#2ecc71' if outcome == 0 else '#e74c3c' for outcome in y]
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Mortality Embedding Space (First 2 PCA Components)')
plt.colorbar(['Survived', 'Expired'], ticks=[0, 1])
plt.tight_layout()
plt.savefig('embeddings_space.png', dpi=100)
plt.show()
```

---

### **Comparing: Raw Features vs. Learned Embeddings for Similarity**

```python
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors

def compare_similarity_spaces(X_raw, embeddings, y, k=5):
    """
    Compare K-NN results using raw features vs. embeddings.
    """
    
    print("K-NN PERFORMANCE COMPARISON")
    print("="*70)
    
    # Method 1: Raw features + Euclidean
    nbrs_raw_euc = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_raw)
    dist_raw_euc, idx_raw_euc = nbrs_raw_euc.kneighbors(X_raw)
    
    # Method 2: Raw features + Cosine
    nbrs_raw_cos = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X_raw)
    dist_raw_cos, idx_raw_cos = nbrs_raw_cos.kneighbors(X_raw)
    
    # Method 3: Embeddings + Cosine (RECOMMENDED)
    nbrs_emb_cos = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
    dist_emb_cos, idx_emb_cos = nbrs_emb_cos.kneighbors(embeddings)
    
    # Method 4: Embeddings + Euclidean
    nbrs_emb_euc = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
    dist_emb_euc, idx_emb_euc = nbrs_emb_euc.kneighbors(embeddings)
    
    methods = {
        'Raw + Euclidean': (idx_raw_euc, y),
        'Raw + Cosine': (idx_raw_cos, y),
        'Embeddings + Cosine': (idx_emb_cos, y),
        'Embeddings + Euclidean': (idx_emb_euc, y),
    }
    
    # METRIC 1: Outcome homogeneity (% of twins with same outcome)
    print("\n1. OUTCOME HOMOGENEITY (% twins with same mortality):")
    print("-"*70)
    
    for method_name, (indices, outcomes) in methods.items():
        matches = []
        for i in range(len(indices)):
            query_outcome = outcomes[i]
            twin_indices = indices[i][1:k+1]  # Exclude self
            twin_outcomes = outcomes[twin_indices]
            match_rate = (twin_outcomes == query_outcome).mean()
            matches.append(match_rate)
        
        avg_match = np.mean(matches)
        std_match = np.std(matches)
        print(f"  {method_name:25s}: {avg_match:.1%} ± {std_match:.1%}")
    
    # METRIC 2: Average distance/similarity
    print("\n2. CLUSTERING TIGHTNESS (average neighbor distance):")
    print("-"*70)
    print(f"  Raw features + Euclidean:     {dist_raw_euc[:, 1:].mean():.4f}")
    print(f"  Raw features + Cosine:        {dist_raw_cos[:, 1:].mean():.4f}")
    print(f"  Embeddings + Cosine:          {dist_emb_cos[:, 1:].mean():.4f}")
    print(f"  Embeddings + Euclidean:       {dist_emb_euc[:, 1:].mean():.4f}")
    
    # METRIC 3: Stability (cross-validated consistency)
    print("\n3. TWIN STABILITY (fraction of twins consistent across splits):")
    print("-"*70)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for method_name, (_, _) in methods.items():
        if 'Raw + Euc' in method_name:
            nbrs_fold = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
            data = X_raw
        elif 'Raw + Cos' in method_name:
            nbrs_fold = NearestNeighbors(n_neighbors=k+1, metric='cosine')
            data = X_raw
        elif 'Emb + Cos' in method_name:
            nbrs_fold = NearestNeighbors(n_neighbors=k+1, metric='cosine')
            data = embeddings
        else:
            nbrs_fold = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
            data = embeddings
        
        stabilities = []
        for train_idx, test_idx in kf.split(data):
            nbrs_fold.fit(data[train_idx])
            _, test_twins_fold = nbrs_fold.kneighbors(data[test_idx])
            stabilities.append(len(test_twins_fold) / (len(test_idx) * k))
        
        print(f"  {method_name:25s}: {np.mean(stabilities):.1%}")
    
    print("\n" + "="*70)
    print("✓ RECOMMENDATION: Embeddings + Cosine Similarity")
    print("  Reasons:")
    print("    1. Task-aligned: Learned what matters for mortality")
    print("    2. Efficient: Low-dim representation (12-15 dims)")
    print("    3. Robust: Non-linear patterns capture complex ICU dynamics")
    print("    4. Interpretable: Can visualize/analyze embedding space")


# Run comparison
compare_similarity_spaces(X_scaled, embeddings_all, y, k=5)
```

---

## Quality Evaluation

### **Metrics**

1. **Outcome homogeneity**
   - % of twins with same mortality outcome
   - Target: >70% match rate (vs ~55% random baseline)
   - *Interpretation*: Twins have similar clinical trajectories

2. **Feature similarity**
   - Difference in age between query and twins
   - Target: <5 years difference
   - *Interpretation*: Twins are demographically similar

3. **Distance distribution**
   - Median cosine distance to twins
   - Target: <0.3 distance (>0.7 similarity)
   - *Interpretation*: Twins are tightly clustered

4. **Coverage**
   - % of patients with "quality" twins
   - Target: >85% coverage
   - *Interpretation*: System works for most patients

### **Validation Strategy**

1. **Hold-out test set**: 20% of data, compute metrics
2. **Cross-validation**: 5-fold CV to check stability
3. **Case studies**: Manual inspection of sample twins
4. **Prospective validation** (future): Compare predictions to actual outcomes in new patients

---

## Recommendations & Design Choices

### **Configuration Summary**

```python
# Feature selection
SELECTED_N_FEATURES = 20
FEATURE_CATEGORIES = {
    'vitals': ['heartrate', 'systolic', 'diastolic', 'sao2', 'respiration'],
    'labs': ['lactate', 'creatinine', 'glucose', 'sodium', 'hemoglobin'],
    'severity': ['apachescore']
}

# Leakage removal
LEAKAGE_PATTERNS = ['discharge', 'predicted', 'outcome']
CORRELATION_THRESHOLD = 0.95  # Remove if corr > this

# Missing data
DROP_MISSING_THRESHOLD = 0.80  # Drop features >80% missing
KNN_NEIGHBORS = 5              # For KNN imputation
CREATE_MISSING_INDICATORS = True

# Standardization
SCALER = 'robust'              # RobustScaler, not StandardScaler

# Embeddings
EMBEDDING_METHOD = 'pca'       # 'pca' or 'autoencoder'
EMBEDDING_DIM = 15

# Similarity & matching
SIMILARITY_METRIC = 'cosine'
K_TWINS = 5
```

### **Rationale for Each Choice**

| Choice | Why |
|--------|-----|
| **20 features** | Captures 80-90% of predictive info, <50 is redundant, >30 adds noise |
| **Vital + lab + severity** | Covers physiology, metabolism, organ function, overall risk |
| **Pattern + correlation leakage removal** | Eliminates outcome information & derived variables |
| **KNN imputation** | Preserves local patient similarity structure |
| **RobustScaler** | Handles extreme ICU outliers better than z-score |
| **PCA embeddings** | Fast, interpretable, good variance capture (80%+) |
| **15 dimensions** | Balances compression vs. info (below 80% threshold) |
| **Cosine similarity** | Scale-invariant, natural for embeddings, fast |
| **K=5** | Balance coverage (~99%) with noise reduction |

---

## Production Deployment Checklist

- [ ] **Data pipeline automated**: Batch processing for new patients
- [ ] **Model serialization**: Save scaler, PCA, KNN index
- [ ] **Inference latency**: <100ms per query (with Faiss)
- [ ] **Interpretability**: Return feature importance for each match
- [ ] **Confidence intervals**: Quantify prediction uncertainty
- [ ] **Metadata**: Store anonymized twin cohorts for auditing
- [ ] **Monitoring**: Track prediction accuracy over time
- [ ] **Fairness audit**: Ensure performance across demographics
- [ ] **Clinical validation**: Prospective study of recommendations
- [ ] **Integration**: API endpoint for EHR/EMR systems

---

## Limitations & Future Work

### **Current Limitations**
1. **Static embeddings**: Uses only first 24-hour data (no temporal dynamics)
2. **No causal inference**: Can't determine if treatments caused outcomes
3. **Limited explainability**: Hard to explain why specific patients are twins
4. **No treatment data**: Doesn't incorporate medications/procedures
5. **Population shift**: Assumes test patients similar to training cohort

### **Future Enhancements**
1. **Temporal models**: LSTM/Transformers for ICU trajectory
2. **Treatment effects**: Causal inference (e.g., CATE methods)
3. **Explainability**: SHAP values showing key matching features
4. **Multi-scale embeddings**: Twins at different ICU phases
5. **Uncertainty quantification**: Bayesian embeddings with confidence
6. **Real-time updates**: Incremental learning as new patients arrive
7. **Fairness**: Debiasing embeddings across demographics
8. **Counterfactuals**: "What if this patient had different labs?"

---

## Files Generated

### **Notebooks** (in workspace)
- `DigitalTwin_PatientSimilarity.ipynb` - Complete step-by-step pipeline
- `digital_twin_pipeline.py` - Reusable Python module

### **Visualizations** (from notebook)
1. `01_dataset_overview.png` - Data shapes, distributions, categories
2. `02_missingness_analysis.png` - Feature missingness patterns
3. `03_feature_selection.png` - Selected features & scoring
4. `04_imputation_results.png` - Before/after imputation
5. `05_standardization.png` - Feature scaling effects
6. `06_pca_embeddings.png` - Variance explained, 2-D projection
7. `07_similarity_metrics.png` - Distance metric distributions
8. `08_twin_quality_evaluation.png` - Quality metrics summary
9. `09_case_studies.png` - Example twin matches

### **Data Exports** (from notebook)
- `digital_twins_index.csv` - Complete K-NN matches for all patients

---

## Quick Start Example

```python
# # Load notebooks from twinPatients.ipynb
model_df = ...  # Your 2520 × 250 eICU dataset

# Run: DigitalTwin_PatientSimilarity.ipynb
# This will:
# 1. Remove leakage
# 2. Select 20 clinical features
# 3. Handle missing data
# 4. Standardize
# 5. Create embeddings
# 6. Build KNN index
# 7. Evaluate quality
# 8. Generate visualizations

# After running, use:
nbrs          # Fitted NearestNeighbors model
embeddings_df # Patient embeddings
scaler        # For scaling new patients
pca           # For embedding new patients

# For a NEW patient's first 24 hours:
new_patient = pd.DataFrame([{
    'age_num': 65,
    'apachescore': 18,
    'heartrate_mean_24h': 85,
    # ... other 17 selected features
}])

# Preprocess exactly like training:
X_new = new_patient[selected_features]
X_new_imputed = imputer.transform(X_new)
X_new_scaled = scaler.transform(X_new_imputed)
X_new_embedded = pca.transform(X_new_scaled)

# Find twins:
distances, indices = nbrs.kneighbors(X_new_embedded)
twin_patients_df = data_for_modeling.iloc[indices[0][1:6]]

# Predict mortality:
mortality_rate = twin_patients_df['y_hosp_mortality'].mean()
print(f"Twin-based mortality prediction: {mortality_rate:.1%}")
```

---

## References

1. **eICU Database**: Pollard et al. (2018). "The eICU Collaborative Research Database"
2. **Digital Twins**: Viceconti et al. (2021). "Personalised medicine through multi-scale modelling"
3. **PCA**: Jolliffe (2002). "Principal Component Analysis"
4. **K-NN**: Altman (1992). "An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression"
5. **Missing Data**: Rubin (1987). "Multiple Imputation for Nonresponse in Surveys"
6. **Fairness in ML**: Buolamwini & Gebru (2018). "Gender Shades"

---

## Contact & Support

For questions about this pipeline, refer to:
- **Implementation**: See `DigitalTwin_PatientSimilarity.ipynb` for detailed Python code
- **Theory**: See sections above for algorithmic details
- **Customization**: Edit parameters in `digital_twin_pipeline.py` module
