"""
FEATURE SELECTION, MISSING DATA HANDLING, AND EMBEDDINGS
=========================================================
Practical implementation guide for building a Digital Twin system.

Covers:
1. Feature selection (leakage removal, clinical scoring, signal detection)
2. Missing data handling (multiple strategies + comparison)
3. Embeddings extraction (mortality prediction → patient vectors)
4. Similarity computation (raw vs. embeddings comparison)

Author: Digital Twin System
Date: 2026-04-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: FEATURE SELECTION (Leakage + Signal Detection)
# ============================================================================

class FeatureSelector:
    """
    Comprehensive feature selection including:
    - Leakage detection (patterns, correlation, temporal)
    - Signal detection (correlation, feature importance, clinical relevance)
    - Clinical scoring (domain knowledge)
    """
    
    def __init__(self, df, outcome_col='y_hosp_mortality'):
        self.df = df
        self.outcome_col = outcome_col
        self.leakage_features = []
        self.selected_features = []
        
    def detect_leakage(self, corr_threshold=0.95, verbose=True):
        """Identify data leakage features."""
        
        leakage_patterns = {
            'discharge': ['discharge', 'discharged', 'location', 'disposition'],
            'predicted': ['predicted', 'predict', 'apache_ii'],
            'outcome': ['outcome', 'actual_', 'hospital_status'],
            'temporal': ['los', 'length_of_stay', 'hospital_los']
        }
        
        leakage_report = {}
        leakage_cols = []
        
        # Pattern-based leakage
        for pattern_type, patterns in leakage_patterns.items():
            matches = []
            for col in self.df.columns:
                if col != self.outcome_col:
                    col_lower = col.lower()
                    if any(p in col_lower for p in patterns):
                        matches.append(col)
            if matches:
                leakage_report[pattern_type] = matches
                leakage_cols.extend(matches)
        
        # Correlation-based leakage (>0.95 to outcome)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr_leakage = []
        
        if self.outcome_col in self.df.columns and self.outcome_col in numeric_cols:
            for col in numeric_cols:
                if col != self.outcome_col and col not in leakage_cols:
                    valid_mask = ~(self.df[col].isna() | self.df[self.outcome_col].isna())
                    if valid_mask.sum() > 30:
                        corr = abs(self.df.loc[valid_mask, col].corr(self.df.loc[valid_mask, self.outcome_col]))
                        if corr > corr_threshold:
                            corr_leakage.append((col, corr))
        
        if corr_leakage:
            leakage_report['high_correlation'] = corr_leakage
            leakage_cols.extend([item[0] for item in corr_leakage])
        
        self.leakage_features = list(set(leakage_cols))
        
        if verbose:
            print("="*70)
            print("DATA LEAKAGE DETECTION")
            print("="*70)
            for leak_type, features in leakage_report.items():
                print(f"\n{leak_type.upper()}: {len(features) if isinstance(features, list) else len(features)}")
                if isinstance(features, list):
                    for f in features[:5]:
                        print(f"  • {f}")
                else:
                    for f, score in features[:5]:
                        print(f"  • {f}: {score:.3f}")
                if len(features) > 5:
                    remain = len(features) - 5
                    print(f"  ... and {remain} more")
            
            print(f"\n✓ Total leakage features: {len(self.leakage_features)}")
        
        return self.leakage_features
    
    def score_features(self, n_features=20, method='combined'):
        """
        Score features by:
        1. Univariate correlation to outcome
        2. Random Forest importance
        3. Clinical domain relevance
        """
        
        # Prepare data
        df_clean = self.df.drop(columns=[c for c in self.leakage_features if c in self.df.columns])
        
        # Remove identifier columns
        exclude = {self.outcome_col, 'patientunitstayid', 'uniquepid', 'hospitalid'}
        numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns 
                       if c not in exclude]
        
        y = df_clean[self.outcome_col].dropna()
        df_numeric = df_clean[numeric_cols + [self.outcome_col]].dropna(subset=[self.outcome_col])
        X = df_numeric[numeric_cols]
        
        scores = pd.DataFrame(index=numeric_cols)
        
        # Score 1: Correlation
        print("Computing univariate correlations...")
        corr_scores = {}
        for col in numeric_cols:
            valid_mask = ~(X[col].isna() | y.isna())
            if valid_mask.sum() > 30:
                corr_scores[col] = abs(X.loc[valid_mask, col].corr(y.loc[valid_mask]))
            else:
                corr_scores[col] = 0.0
        
        scores['correlation'] = pd.Series(corr_scores)
        
        # Score 2: RF Importance
        print("Training Random Forest...")
        X_imputed = SimpleImputer(strategy='median').fit_transform(X)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        rf.fit(X_imputed, y)
        
        scores['rf_importance'] = pd.Series(
            dict(zip(numeric_cols, rf.feature_importances_))
        )
        
        # Score 3: Clinical relevance
        print("Scoring clinical relevance...")
        clinical_terms = {
            'vitals': ['heartrate', 'systolic', 'diastolic', 'sao2', 'respiration', 'shock_index'],
            'labs': ['lactate', 'creatinine', 'glucose', 'sodium', 'potassium', 'hemoglobin', 
                    'chloride', 'ph', 'pco2', 'po2', 'bicarbonate'],
            'severity': ['apache', 'aps', 'sofa', 'saps']
        }
        
        clinical_scores = {}
        for col in numeric_cols:
            col_lower = col.lower()
            score = 0.0
            
            # Missingness: reward complete, penalize sparse
            miss_pct = X[col].isna().mean()
            if miss_pct < 0.1:
                score += 0.1
            elif miss_pct > 0.8:
                score -= 0.3
            
            # Domain matching
            for domain, terms in clinical_terms.items():
                if any(term in col_lower for term in terms):
                    if domain == 'vitals':
                        score += 0.5
                    elif domain == 'labs':
                        score += 0.3
                    else:  # severity
                        score += 0.4
                    break
            
            # Aggregation preference (mean/median > min/max)
            if '_mean_' in col or '_median_' in col:
                score += 0.05
            elif '_std_' in col or '_min_' in col or '_max_' in col:
                score -= 0.02
            
            clinical_scores[col] = max(0, score)
        
        scores['clinical'] = pd.Series(clinical_scores)
        
        # Combine scores (normalize first)
        for col_name in ['correlation', 'rf_importance', 'clinical']:
            if scores[col_name].max() > 0:
                scores[col_name] = scores[col_name] / scores[col_name].max()
        
        scores['combined'] = 0.3*scores['correlation'] + 0.3*scores['rf_importance'] + 0.4*scores['clinical']
        
        self.selected_features = scores.nlargest(n_features, 'combined').index.tolist()
        
        print(f"\n✓ Selected {len(self.selected_features)} features:")
        print(scores.loc[self.selected_features].sort_values('combined', ascending=False))
        
        return self.selected_features, scores


# ============================================================================
# PART 2: MISSING DATA HANDLING (Compare Strategies)
# ============================================================================

class MissingnessHandler:
    """Handle missing data in ICU datasets with multiple strategies."""
    
    def __init__(self, X, feature_names=None, verbose=True):
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            self.feature_names = feature_names or [f"F_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=self.feature_names).copy()
        else:
            self.X = X.copy()
            self.feature_names = feature_names or list(self.X.columns)
        
        self.verbose = verbose
        self.missing_analysis = self._analyze_missingness()
    
    def _analyze_missingness(self):
        """Characterize missingness patterns."""
        
        missing_pct = (self.X.isna().sum() / len(self.X) * 100).sort_values(ascending=False)
        
        categorical = {
            'complete': missing_pct[missing_pct < 10],
            'sparse': missing_pct[(missing_pct >= 10) & (missing_pct < 50)],
            'ultra_sparse': missing_pct[(missing_pct >= 50) & (missing_pct < 80)],
            'ignore': missing_pct[missing_pct >= 80]
        }
        
        if self.verbose:
            print("MISSINGNESS ANALYSIS")
            print("="*70)
            for cat, features in categorical.items():
                print(f"{cat.upper():15s}: {len(features):3d} features")
        
        return categorical
    
    def impute_mean(self):
        """Simple mean imputation (baseline)."""
        imputer = SimpleImputer(strategy='mean')
        return pd.DataFrame(imputer.fit_transform(self.X), columns=self.feature_names), imputer
    
    def impute_median(self):
        """Median imputation (robust to outliers)."""
        imputer = SimpleImputer(strategy='median')
        return pd.DataFrame(imputer.fit_transform(self.X), columns=self.feature_names), imputer
    
    def impute_knn(self, n_neighbors=5):
        """KNN imputation (structure-preserving)."""
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        return pd.DataFrame(imputer.fit_transform(self.X), columns=self.feature_names), imputer
    
    def with_indicators(self, imputation_method='median'):
        """Add missing indicators without actual imputation."""
        
        # Create indicator columns
        indicators = pd.DataFrame()
        for col in self.X.columns:
            if self.X[col].isna().sum() > 0:
                indicators[f'{col}_missing'] = self.X[col].isna().astype(int)
        
        # Impute NaNs
        if imputation_method == 'median':
            X_imputed = self.X.fillna(self.X.median())
        elif imputation_method == 'mean':
            X_imputed = self.X.fillna(self.X.mean())
        else:
            raise ValueError("Unknown imputation method")
        
        # Combine
        X_combined = pd.concat([X_imputed.reset_index(drop=True), 
                               indicators.reset_index(drop=True)], axis=1)
        
        return X_combined


# ============================================================================
# PART 3: EMBEDDINGS FOR MORTALITY PREDICTION
# ============================================================================

def build_embedding_model(X_train, y_train, X_val, y_val, 
                          embedding_dim=12, epochs=50):
    """
    Build a neural network that predicts mortality and extracts embeddings.
    
    Architecture:
    Input(20) → Dense(64) → Dense(32) → Dense(embedding_dim) → Dense(16) → Output(1)
    """
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, optimizers
    except ImportError:
        print("⚠ TensorFlow not installed. Install with: pip install tensorflow")
        return None, None, None
    
    input_dim = X_train.shape[1]
    
    # Build model
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Embedding layer
    embeddings = layers.Dense(embedding_dim, activation='relu', name='embedding')(x)
    
    # Output layer
    x = layers.Dense(16, activation='relu')(embeddings)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create models
    model = Model(inputs=inputs, outputs=outputs, name='mortality_predictor')
    embedding_extractor = Model(inputs=inputs, outputs=embeddings, name='embedding_extractor')
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['auc']
    )
    
    # Train
    print(f"Training mortality prediction model ({input_dim}D → {embedding_dim}D embeddings)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate
    val_auc = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"✓ Validation AUC: {val_auc:.4f}")
    
    return model, embedding_extractor, history


# ============================================================================
# PART 4: SIMILARITY COMPARISON (Raw vs. Embeddings)
# ============================================================================

def compare_similarity_methods(X_raw, embeddings, y, k=5):
    """
    Compare patient similarity using different methods.
    
    Methods:
    1. Raw features + Euclidean
    2. Raw features + Cosine
    3. Embeddings + Cosine (RECOMMENDED)
    4. Embeddings + Euclidean
    """
    
    print("\n" + "="*70)
    print("SIMILARITY METHOD COMPARISON")
    print("="*70)
    
    methods = {
        'Raw + Euclidean': (X_raw, 'euclidean'),
        'Raw + Cosine': (X_raw, 'cosine'),
        'Embeddings + Cosine': (embeddings, 'cosine'),
        'Embeddings + Euclidean': (embeddings, 'euclidean'),
    }
    
    results = {}
    
    for method_name, (data, metric) in methods.items():
        # Fit KNN
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        # Metric 1: Outcome homogeneity
        outcome_matches = []
        for i in range(len(indices)):
            query_outcome = y[i]
            twin_outcomes = y[indices[i][1:k+1]]  # Exclude self
            match_rate = (twin_outcomes == query_outcome).mean()
            outcome_matches.append(match_rate)
        
        # Metric 2: Distance tightness
        neighbor_distances = distances[:, 1:k+1].flatten()
        
        results[method_name] = {
            'outcome_match': np.mean(outcome_matches),
            'outcome_match_std': np.std(outcome_matches),
            'mean_distance': neighbor_distances.mean(),
            'median_distance': np.median(neighbor_distances),
            'indices': indices,
            'distances': distances
        }
    
    # Print results
    print("\n1. OUTCOME HOMOGENEITY (% twins with same mortality):")
    print("-"*70)
    for method, result in results.items():
        print(f"  {method:25s}: {result['outcome_match']:.1%} ± {result['outcome_match_std']:.1%}")
    
    print("\n2. CLUSTERING TIGHTNESS (neighbor distance):")
    print("-"*70)
    for method, result in results.items():
        print(f"  {method:25s}: {result['mean_distance']:.4f} (median: {result['median_distance']:.4f})")
    
    print("\n" + "="*70)
    print("✓ RECOMMENDATION: Embeddings + Cosine Similarity")
    print("  • Task-aligned representation (learned for mortality)")
    print("  • Efficient dimensionality reduction")
    print("  • Non-linear pattern capture")
    print("  • Scale-invariant similarity metric")
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FEATURE SELECTION, MISSING DATA & EMBEDDINGS PIPELINE")
    print("="*70)
    
    print("""
    This module provides:
    
    1. FeatureSelector: Detect leakage + clinical feature selection
       Usage:
         selector = FeatureSelector(model_df)
         leakage = selector.detect_leakage()
         features, scores = selector.score_features(n_features=20)
    
    2. MissingnessHandler: Compare imputation strategies
       Usage:
         handler = MissingnessHandler(X)
         X_mean, _ = handler.impute_mean()
         X_knn, _ = handler.impute_knn()
         X_indicators = handler.with_indicators()
    
    3. build_embedding_model: Extract embeddings from mortality predictor
       Usage:
         model, extractor, hist = build_embedding_model(X_train, y_train, X_val, y_val)
         embeddings = extractor.predict(X)
    
    4. compare_similarity_methods: Compare raw vs. embedding similarity
       Usage:
         results = compare_similarity_methods(X_raw, embeddings, y)
    
    See DIGITAL_TWIN_SYSTEM_DESIGN.md for detailed examples.
    """)
