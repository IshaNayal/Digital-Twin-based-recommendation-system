"""
Digital Twin-based Patient Similarity System
=============================================
Comprehensive pipeline for finding similar ICU patients using clinical embeddings.

Stages:
1. Remove data leakage features
2. Feature selection (clinical + statistical)
3. Handle missing data (missingness-aware)
4. Standardization
5. Build embeddings (classical + optional deep learning)
6. Similarity metrics & KNN matching
7. Evaluation of twin quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STAGE 1: DATA LEAKAGE REMOVAL
# ============================================================================

def remove_leakage_features(model_df, outcome_col='y_hosp_mortality'):
    """
    Remove columns that would leak information about the outcome.
    
    Leakage patterns:
    - Discharge status / location (directly related to outcome)
    - Predicted mortality (directly derived from outcome)
    - Any column containing 'discharge' or 'outcome'
    - Any column with >95% correlation to outcome
    
    Args:
        model_df: DataFrame with all features
        outcome_col: Name of target column
    
    Returns:
        df_clean: DataFrame with leakage features removed
        removed_cols: List of removed column names
    """
    leakage_patterns = [
        'discharge', 'discharged', 'outcome',
        'hospital_discharge', 'unit_discharge',
        'unit_discharge_location', 'unit_discharge_status',
        'hospitaldischarge', 'unitdischarge', 'predicted'
    ]
    
    df = model_df.copy()
    removed_cols = []
    
    # Remove by naming pattern
    for col in df.columns:
        if any(pattern in col.lower() for pattern in leakage_patterns):
            if col != outcome_col:
                removed_cols.append(col)
    
    # Numeric columns: remove if correlation to outcome >0.95 (high leakage risk)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if outcome_col in df.columns and outcome_col in numeric_cols:
        outcome_values = df[outcome_col].dropna()
        for col in numeric_cols:
            if col != outcome_col and col not in removed_cols:
                # Safely compute correlation
                valid_mask = ~(df[col].isna() | df[outcome_col].isna())
                if valid_mask.sum() > 10:  # Only if enough data
                    corr = abs(df.loc[valid_mask, col].corr(df.loc[valid_mask, outcome_col]))
                    if corr > 0.95:
                        removed_cols.append(col)
    
    # Remove duplicates and status columns
    for col in ['age', 'uniquepid', 'patientunitstayid', 'hospitalid']:
        if col in df.columns and col in removed_cols:
            removed_cols.remove(col)  # Keep identifiers
    
    removed_cols = list(set(removed_cols))
    df_clean = df.drop(columns=[c for c in removed_cols if c in df.columns])
    
    return df_clean, removed_cols


# ============================================================================
# STAGE 2: CLINICAL FEATURE SELECTION
# ============================================================================

def select_clinical_features(model_df, outcome_col='y_hosp_mortality', 
                            n_features=20, method='combined'):
    """
    Select clinically meaningful features using multiple strategies:
    
    Methods:
    - 'statistical': Top features by univariate correlation
    - 'random_forest': Feature importance from RF classifier
    - 'combined': Hybrid approach (weighted combination)
    
    Prioritizes:
    1. Vital signs (core physiology)
    2. Lab results (metabolic state)
    3. Severity scores (APACHE, etc.)
    4. Derived features (e.g., shock index)
    
    Args:
        model_df: DataFrame with features
        outcome_col: Target variable name
        n_features: Number of features to select
        method: 'statistical', 'random_forest', or 'combined'
    
    Returns:
        selected_features: List of selected feature names
        feature_scores: DataFrame with scores for each feature
    """
    
    df = model_df.dropna(subset=[outcome_col]).copy()
    y = df[outcome_col]
    
    # Numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [outcome_col, 'patientunitstayid', 'uniquepid', 'hospitalid']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Impute for feature importance calculation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    scores_dict = {}
    
    # --- Statistical: correlation to outcome ---
    for col in X.columns:
        valid_mask = ~(X[col].isna() | y.isna())
        if valid_mask.sum() > 10:
            corr = abs(X.loc[valid_mask, col].corr(y.loc[valid_mask]))
            scores_dict[col] = {'correlation': corr}
    
    # --- Random Forest importance ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    for col in X.columns:
        if col not in scores_dict:
            scores_dict[col] = {}
        scores_dict[col]['rf_importance'] = rf_importance[col]
    
    # --- Clinical domain scoring: prioritize vitals, labs, severity ---
    clinical_priority = {
        'vitals': ['heartrate', 'systolic', 'diastolic', 'sao2', 'respiration', 'temperature', 'shock_index'],
        'labs': ['lactate', 'creatinine', 'glucose', 'sodium', 'potassium', 'chloride', 'hemoglobin', 'hematocrit', 'ph', 'pco2', 'po2'],
        'severity': ['apache', 'aps', 'sofa', 'news']
    }
    
    for col in X.columns:
        clinical_score = 0.0
        col_lower = col.lower()
        
        # Penalty for ultra-sparse features
        missingness = X[col].isna().mean()
        if missingness > 0.8:
            clinical_score -= 0.2
        elif missingness < 0.5:
            clinical_score += 0.1
        
        # Boost for vital signs & key labs
        if any(v in col_lower for v in clinical_priority['vitals']):
            clinical_score += 0.5
        if any(v in col_lower for v in clinical_priority['labs']):
            clinical_score += 0.3
        if any(v in col_lower for v in clinical_priority['severity']):
            clinical_score += 0.4
        
        # Boost for mean/median over min/max (better stability)
        if '_mean_' in col or '_median_' in col:
            clinical_score += 0.05
        
        if col not in scores_dict:
            scores_dict[col] = {}
        scores_dict[col]['clinical_score'] = clinical_score
    
    # --- Combine scores ---
    feature_scores = pd.DataFrame(scores_dict).T
    feature_scores = feature_scores.fillna(0)
    
    if method == 'statistical':
        feature_scores['combined_score'] = feature_scores.get('correlation', 0)
    elif method == 'random_forest':
        feature_scores['combined_score'] = feature_scores.get('rf_importance', 0)
    else:  # combined
        # Normalize each score to [0,1]
        for col in ['correlation', 'rf_importance', 'clinical_score']:
            if col in feature_scores.columns and feature_scores[col].max() > 0:
                feature_scores[col] = feature_scores[col] / feature_scores[col].max()
        
        # Weighted combination
        feature_scores['combined_score'] = (
            0.3 * feature_scores.get('correlation', 0) +
            0.3 * feature_scores.get('rf_importance', 0) +
            0.4 * feature_scores.get('clinical_score', 0)
        )
    
    # Select top-N features
    selected_features = feature_scores.nlargest(n_features, 'combined_score').index.tolist()
    
    return selected_features, feature_scores


# ============================================================================
# STAGE 3: MISSING DATA HANDLING
# ============================================================================

def analyze_missingness(X, threshold=0.8):
    """
    Analyze missingness patterns and recommend strategy.
    
    Returns features by missingness level:
    - Complete: <10% missing
    - Sparse: 10-50% missing (use KNN imputation)
    - Ultra-sparse: 50-80% missing (consider dropping or creating indicator vars)
    - Ignore: >80% missing (drop)
    """
    miss_pct = X.isna().sum() / len(X) * 100
    miss_df = pd.DataFrame({
        'feature': miss_pct.index,
        'missing_pct': miss_pct.values
    }).sort_values('missing_pct', ascending=False)
    
    complete = miss_df[miss_df['missing_pct'] < 10]['feature'].tolist()
    sparse = miss_df[(miss_df['missing_pct'] >= 10) & (miss_df['missing_pct'] < 50)]['feature'].tolist()
    ultra_sparse = miss_df[(miss_df['missing_pct'] >= 50) & (miss_df['missing_pct'] < threshold)]['feature'].tolist()
    drop = miss_df[miss_df['missing_pct'] >= threshold]['feature'].tolist()
    
    return {
        'complete': complete,
        'sparse': sparse,
        'ultra_sparse': ultra_sparse,
        'drop': drop,
        'summary': miss_df
    }


def handle_missing_data(X, strategy='hybrid', indicator_vars=True):
    """
    Handle missing data with strategy:
    
    Strategies:
    - 'median': Simple median imputation
    - 'knn': K-nearest neighbors imputation (better for related features)
    - 'hybrid': KNN for sparse features, median for ultra-sparse
    
    Args:
        X: Feature DataFrame
        strategy: Imputation strategy
        indicator_vars: If True, create binary indicators for originally-missing values
    
    Returns:
        X_imputed: Imputed DataFrame
        imputer: Fitted imputer object (for test set)
        missing_indicators: Dict of indicator columns created
    """
    
    X_work = X.copy()
    missing_indicators = {}
    
    # Create missing indicators BEFORE imputation
    if indicator_vars:
        for col in X_work.columns:
            if X_work[col].isna().sum() > 0:
                missing_indicators[f'{col}_missing'] = X_work[col].isna().astype(int)
    
    # Analyze missingness
    miss_analysis = analyze_missingness(X_work)
    
    # Drop ultra-sparse columns (>80% missing)
    cols_to_drop = miss_analysis['drop']
    X_work = X_work.drop(columns=[c for c in cols_to_drop if c in X_work.columns])
    
    # Impute remaining
    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_work),
            columns=X_work.columns,
            index=X_work.index
        )
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_work),
            columns=X_work.columns,
            index=X_work.index
        )
    else:  # hybrid
        imputer_knn = KNNImputer(n_neighbors=5, weights='distance')
        imputer_median = SimpleImputer(strategy='median')
        
        # Apply KNN to sparse features
        sparse_cols = miss_analysis['sparse']
        complete_cols = miss_analysis['complete']
        all_cols_to_impute = sparse_cols + complete_cols
        
        if len(all_cols_to_impute) > 0:
            X_imputed = pd.DataFrame(
                imputer_knn.fit_transform(X_work[all_cols_to_impute]),
                columns=all_cols_to_impute,
                index=X_work.index
            )
        else:
            X_imputed = X_work.copy()
    
    # Add back missing indicators
    for ind_name, ind_series in missing_indicators.items():
        X_imputed[ind_name] = ind_series.values
    
    return X_imputed, imputer, missing_indicators


# ============================================================================
# STAGE 4: FEATURE STANDARDIZATION
# ============================================================================

def standardize_features(X, method='robust', return_scaler=True):
    """
    Standardize features for embedding.
    
    Methods:
    - 'standard': Z-score normalization (assumes normal distribution)
    - 'robust': Median and IQR (robust to outliers, better for ICU data)
    - 'minmax': [0,1] scaling
    
    Recommendation for ICU: Use 'robust' as ICU data often has extreme outliers
    
    Args:
        X: Feature DataFrame
        method: Scaling method
        return_scaler: If True, return fitted scaler for test data
    
    Returns:
        X_scaled: Standardized DataFrame
        scaler: Fitted scaler object
    """
    
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled, scaler


# ============================================================================
# STAGE 5: BUILD EMBEDDINGS
# ============================================================================

def build_pca_embeddings(X_scaled, n_components=15, explained_variance_threshold=0.85):
    """
    Classical approach: PCA embeddings for dimensionality reduction.
    
    Advantages:
    - Interpretable (linear combinations of features)
    - Fast & scalable
    - Good baseline
    
    Disadvantages:
    - Assumes linear relationships
    - May not capture complex clinical patterns
    
    Args:
        X_scaled: Standardized feature matrix
        n_components: Number of PCA components (or fraction of variance)
        explained_variance_threshold: Stop when this much variance is explained
    
    Returns:
        embeddings: PCA embeddings (n_samples × n_components)
        pca: Fitted PCA object
        explained_var: Cumulative explained variance ratio
    """
    
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"PCA Embeddings:")
    print(f"  Components: {pca.n_components_}")
    print(f"  Explained variance: {explained_var[-1]:.3f}")
    print(f"  Shape: {embeddings.shape}")
    
    return embeddings, pca, explained_var


def build_autoencoder_embeddings(X_scaled, embedding_dim=15, epochs=50, batch_size=32, 
                                val_split=0.2, verbose=0):
    """
    Deep learning approach: Autoencoder embeddings.
    
    Architecture: Input → Dense(128) → Dense(embedding_dim) → Dense(128) → Output
    
    Advantages:
    - Captures non-linear patterns
    - Can learn clinically meaningful representations
    - Good for finding semantic "twins"
    
    Disadvantages:
    - Requires more data
    - Less interpretable
    - Longer training time
    
    Note: Requires tensorflow/keras. Install with:
        pip install tensorflow
    
    Args:
        X_scaled: Standardized feature matrix
        embedding_dim: Dimensionality of bottleneck layer
        epochs: Training epochs
        batch_size: Batch size
        val_split: Validation split fraction
        verbose: Verbosity level
    
    Returns:
        embeddings: Autoencoder embeddings
        encoder_model: Trained encoder (input → bottleneck)
        autoencoder_model: Full autoencoder
    """
    
    try:
        from tensorflow.keras import layers, Model, Sequential
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        print("TensorFlow not installed. Returning None.")
        return None, None, None
    
    input_dim = X_scaled.shape[1]
    
    # Encoder
    encoder = Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(embedding_dim, activation='relu', name='embedding')
    ])
    
    # Decoder
    decoder = Sequential([
        layers.Dense(64, activation='relu', input_shape=(embedding_dim,)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='gaussian')
    ])
    
    # Full autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=verbose
    )
    
    # Extract embeddings
    embeddings = encoder.predict(X_scaled, verbose=0)
    
    print(f"Autoencoder Embeddings:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Final reconstruction loss: {history.history['loss'][-1]:.4f}")
    
    return embeddings, encoder, autoencoder


# ============================================================================
# STAGE 6: SIMILARITY METRICS & KNN MATCHING
# ============================================================================

def evaluate_similarity_metrics(embeddings, reference_idx=0, top_k=5):
    """
    Compare different similarity metrics on a reference patient.
    
    Metrics:
    - Cosine: Good for high-dim data, robust angle-based similarity
    - Euclidean: Distance in feature space, sensitive to scale
    - Manhattan: L1 distance, robust to outliers
    
    For clinical digital twins: Cosine usually best (morphology-based without scale bias)
    """
    
    reference_emb = embeddings[reference_idx:reference_idx+1]
    
    metrics = {
        'cosine': 1 - cosine_distances(reference_emb, embeddings)[0],
        'euclidean': -euclidean_distances(reference_emb, embeddings)[0],  # Negative for consistency
        'manhattan': -cdist(reference_emb, embeddings, metric='cityblock')[0],
    }
    
    results = pd.DataFrame(metrics)
    results['patient_id'] = range(len(embeddings))
    
    print(f"\nSimilarity scores for reference patient {reference_idx}:")
    for metric_name in metrics.keys():
        top_indices = np.argsort(-results[metric_name].values)[:top_k]
        print(f"\n{metric_name.upper()}:")
        for rank, idx in enumerate(top_indices[:top_k], 1):
            print(f"  {rank}. Patient {idx}: {results.loc[idx, metric_name]:.4f}")
    
    return results


def find_digital_twins(embeddings, indices=None, k=5, metric='cosine'):
    """
    Find K nearest neighbors (digital twins) for all patients.
    
    Args:
        embeddings: Patient embeddings (n_patients × embedding_dim)
        indices: List of patient indices to find twins for (None = all)
        k: Number of nearest neighbors to return
        metric: 'cosine', 'euclidean', or 'manhattan'
    
    Returns:
        neighbors_result: Dict with neighbors and distances for each patient
    """
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric).fit(embeddings)
    distances, indices_nn = nbrs.kneighbors(embeddings)
    
    neighbors_result = {
        'distances': distances,
        'indices': indices_nn,
        'metric': metric
    }
    
    return neighbors_result


def get_twins_for_patient(patient_idx, neighbors_result, model_df, k=5, exclude_self=True):
    """
    Retrieve digital twin information for a specific patient.
    
    Args:
        patient_idx: Index of query patient
        neighbors_result: Output from find_digital_twins()
        model_df: Original dataframe with patient features
        k: Number of twins to return
        exclude_self: If True, don't count the patient itself
    
    Returns:
        twins_df: DataFrame with twin indices, distances, and key features
    """
    
    indices_nn = neighbors_result['indices'][patient_idx]
    distances = neighbors_result['distances'][patient_idx]
    
    # Exclude self if requested
    if exclude_self and indices_nn[0] == patient_idx:
        indices_nn = indices_nn[1:k+1]
        distances = distances[1:k+1]
    else:
        indices_nn = indices_nn[:k]
        distances = distances[:k]
    
    twins_df = pd.DataFrame({
        'twin_patient_id': indices_nn,
        f'distance ({neighbors_result["metric"]})': distances,
    })
    
    # Add key features for comparison
    key_features = ['age_num', 'y_hosp_mortality', 'apachescore']
    key_features = [f for f in key_features if f in model_df.columns]
    
    for feat in key_features:
        twins_df[f'twin_{feat}'] = model_df.iloc[indices_nn][feat].values
        twins_df[f'query_{feat}'] = model_df.iloc[patient_idx][feat]
    
    return twins_df


# ============================================================================
# STAGE 7: EVALUATION OF TWIN QUALITY
# ============================================================================

def evaluate_twin_quality(model_df, neighbors_result, embedding_method='pca', k=5):
    """
    Evaluate quality of found twins using multiple metrics.
    
    Metrics:
    1. Outcome homogeneity: Do twins have similar outcomes?
    2. Feature similarity: Are key clinical features similar?
    3. Cohort coverage: What fraction of patients have good twins?
    
    Args:
        model_df: Original dataframe with labels and features
        neighbors_result: Output from find_digital_twins()
        embedding_method: 'pca', 'autoencoder', etc. for reporting
        k: Number of neighbors used
    
    Returns:
        quality_report: Dict with evaluation metrics
    """
    
    indices_nn = neighbors_result['indices']
    distances = neighbors_result['distances']
    
    query_outcomes = model_df['y_hosp_mortality'].values
    quality_report = {
        'method': embedding_method,
        'k': k,
        'metric': neighbors_result['metric']
    }
    
    # --- 1. Outcome homogeneity ---
    outcome_matches = 0
    for i, neighbor_indices in enumerate(indices_nn):
        # Exclude self (first index)
        neighbor_outcomes = query_outcomes[neighbor_indices[1:k+1]]
        query_outcome = query_outcomes[i]
        match_rate = (neighbor_outcomes == query_outcome).mean()
        outcome_matches += match_rate
    
    quality_report['avg_outcome_match_rate'] = outcome_matches / len(indices_nn)
    
    # --- 2. Distance distribution ---
    all_distances = distances[:, 1:k+1].flatten()  # Exclude self
    quality_report['mean_neighbor_distance'] = all_distances.mean()
    quality_report['median_neighbor_distance'] = np.median(all_distances)
    quality_report['std_neighbor_distance'] = all_distances.std()
    
    # --- 3. Feature similarity for key variables ---
    feature_correlations = {}
    for feat in ['age_num', 'apachescore']:
        if feat in model_df.columns:
            corr_list = []
            for i, neighbor_indices in enumerate(indices_nn):
                neighbor_values = model_df.iloc[neighbor_indices[1:k+1]][feat].values
                query_value = model_df.iloc[i][feat]
                if not pd.isna(query_value) and not pd.isna(neighbor_values).all():
                    valid_neighbors = neighbor_values[~pd.isna(neighbor_values)]
                    if len(valid_neighbors) > 0:
                        corr_list.append(abs(valid_neighbors.mean() - query_value))
            if corr_list:
                feature_correlations[f'{feat}_mean_abs_diff'] = np.mean(corr_list)
    
    quality_report.update(feature_correlations)
    
    # --- 4. Coverage (percentage with usable twins) ---
    coverage = (distances[:, 1] < np.quantile(distances[:, 1], 0.95)).mean()
    quality_report['coverage_at_95pct_distance'] = coverage
    
    return quality_report


# ============================================================================
# COMPLETE PIPELINE ORCHESTRATION
# ============================================================================

def run_complete_pipeline(model_df, outcome_col='y_hosp_mortality', 
                         n_features=20, embedding_method='pca', k=5,
                         use_autoencoder=False, debug=True):
    """
    Complete Digital Twin pipeline from raw data to nearest neighbors.
    
    Args:
        model_df: Raw feature dataframe
        outcome_col: Target column name
        n_features: Number of features to select
        embedding_method: 'pca' or 'autoencoder'
        k: Number of neighbors to find
        use_autoencoder: If True, train autoencoder (requires TensorFlow)
        debug: If True, print progress and visualizations
    
    Returns:
        results_dict: Dict containing:
            - model_df_clean: Cleaned dataframe
            - X_selected: Selected features
            - X_scaled: Standardized features
            - embeddings: Patient embeddings
            - neighbors_result: K-NN results
            - quality_report: Evaluation metrics
            - feature_info: Selected feature information
    """
    
    results_dict = {}
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 1: REMOVE DATA LEAKAGE")
        print("="*70)
    
    # Stage 1: Remove leakage
    model_df_clean, removed_cols = remove_leakage_features(model_df, outcome_col)
    if debug:
        print(f"Removed {len(removed_cols)} leakage features:")
        print(f"  {removed_cols[:10]}")
    results_dict['model_df_clean'] = model_df_clean
    results_dict['removed_leakage_cols'] = removed_cols
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 2: FEATURE SELECTION")
        print("="*70)
    
    # Stage 2: Select features
    selected_features, feature_scores = select_clinical_features(
        model_df_clean, outcome_col, n_features=n_features, method='combined'
    )
    if debug:
        print(f"Selected {len(selected_features)} features:")
        print(feature_scores.loc[selected_features].sort_values('combined_score', ascending=False))
    
    results_dict['selected_features'] = selected_features
    results_dict['feature_scores'] = feature_scores
    
    # Extract selected features
    feature_cols = selected_features + [outcome_col]
    feature_cols = [c for c in feature_cols if c in model_df_clean.columns]
    X_selected = model_df_clean[feature_cols].dropna(subset=[outcome_col]).copy()
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 3: MISSING DATA HANDLING")
        print("="*70)
    
    # Stage 3: Handle missing data
    X_features = X_selected[[c for c in selected_features if c in X_selected.columns]].copy()
    miss_analysis = analyze_missingness(X_features)
    if debug:
        print(miss_analysis['summary'].head(10))
    
    X_imputed, imputer, missing_indicators = handle_missing_data(X_features, strategy='hybrid', indicator_vars=True)
    if debug:
        print(f"Imputed shape: {X_imputed.shape}")
        print(f"Created {len(missing_indicators)} missing indicators")
    
    results_dict['X_imputed'] = X_imputed
    results_dict['imputer'] = imputer
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 4: STANDARDIZATION")
        print("="*70)
    
    # Stage 4: Standardize
    X_scaled, scaler = standardize_features(X_imputed, method='robust')
    if debug:
        print(f"Standardized shape: {X_scaled.shape}")
        print(f"Mean: {X_scaled.mean().mean():.6f}, Std: {X_scaled.std().mean():.6f}")
    
    results_dict['X_scaled'] = X_scaled
    results_dict['scaler'] = scaler
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 5: BUILD EMBEDDINGS")
        print("="*70)
    
    # Stage 5: Build embeddings
    if use_autoencoder:
        embeddings, encoder, autoencoder = build_autoencoder_embeddings(
            X_scaled, embedding_dim=15, epochs=50, verbose=0
        )
        results_dict['embeddings'] = embeddings
        results_dict['encoder'] = encoder
        results_dict['autoencoder'] = autoencoder
        results_dict['embedding_method'] = 'autoencoder'
    else:
        embeddings, pca, explained_var = build_pca_embeddings(X_scaled, n_components=15)
        results_dict['embeddings'] = embeddings
        results_dict['pca'] = pca
        results_dict['explained_variance'] = explained_var
        results_dict['embedding_method'] = 'pca'
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 6: SIMILARITY METRICS & KNN")
        print("="*70)
    
    # Stage 6: Find twins
    neighbors_result = find_digital_twins(embeddings, k=k, metric='cosine')
    if debug:
        print(f"Found {k} nearest neighbors for {len(embeddings)} patients using cosine similarity")
        print("\nExample: First patient's twins:")
        print(neighbors_result)
    
    results_dict['neighbors_result'] = neighbors_result
    
    if debug:
        print("\n" + "="*70)
        print("STAGE 7: EVALUATION")
        print("="*70)
    
    # Stage 7: Evaluate quality
    quality_report = evaluate_twin_quality(
        X_selected,
        neighbors_result,
        embedding_method=results_dict['embedding_method'],
        k=k
    )
    
    if debug:
        print("\nQuality Report:")
        for key, value in quality_report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    results_dict['quality_report'] = quality_report
    
    return results_dict


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_twin_matches(model_df, results_dict, patient_idx=0, k=5):
    """
    Visualize a patient and their digital twins across key features.
    """
    
    neighbors_result = results_dict['neighbors_result']
    selected_features = results_dict['selected_features']
    
    # Get twin information
    indices_nn = neighbors_result['indices'][patient_idx]
    distances = neighbors_result['distances'][patient_idx]
    
    # Exclude self
    twin_indices = indices_nn[1:k+1]
    twin_distances = distances[1:k+1]
    
    # Key features to visualize
    viz_features = [f for f in ['age_num', 'apachescore', 'heartrate_mean_24h', 
                                  'systemicsystolic_mean_24h', 'sao2_mean_24h'] 
                    if f in selected_features or f in model_df.columns][:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for ax_idx, feat in enumerate(viz_features):
        ax = axes[ax_idx]
        
        if feat in model_df.columns:
            query_val = model_df.iloc[patient_idx][feat]
            twin_vals = model_df.iloc[twin_indices][feat].values
            
            ax.bar(range(len(twin_vals)), twin_vals, alpha=0.6, label='Twins')
            ax.axhline(query_val, color='r', linestyle='--', linewidth=2, label='Query Patient')
            ax.set_xlabel('Twin Rank')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} (Distance: {twin_distances[0]:.3f})')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('./digital_twin_visualization.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved: digital_twin_visualization.png")


if __name__ == '__main__':
    print("Digital Twin Pipeline Module")
    print("="*70)
    print("Import this module and run:")
    print("  results = run_complete_pipeline(model_df, n_features=20)")
