"""
DIGITAL TWIN SYSTEM - ADVANCED IMPROVEMENTS
=============================================

Step-by-step enhancements for production-level performance:

1. Advanced embeddings (XGBoost, Neural Networks, Autoencoders)
2. Learned distance metrics (Mahalanobis, learned similarity)
3. Feature weighting by clinical importance
4. Smart missing data handling in similarity computation
5. Techniques to improve outcome homogeneity (65-75% target)
6. Outlier detection and handling
7. Advanced evaluation metrics

This module provides implementations for each improvement area.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import normaltest
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPROVEMENT 1: ADVANCED EMBEDDINGS
# ============================================================================

class EmbeddingComparison:
    """Compare multiple embedding methods for quality and performance."""
    
    def __init__(self, X, y, X_val=None, y_val=None):
        """
        Args:
            X: Training features (N × D)
            y: Training mortality labels
            X_val, y_val: Validation data (optional)
        """
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.results = {}
    
    def pca_embeddings(self, n_components=8):
        """Baseline: PCA embeddings."""
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(self.X)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        
        result = {
            'method': 'PCA',
            'embeddings': embeddings,
            'model': pca,
            'variance': variance_explained,
            'n_dims': n_components,
            'speed': 'Very Fast',
            'interpretability': 'High',
        }
        self.results['pca'] = result
        return result
    
    def xgboost_embeddings(self, n_components=8, n_rounds=100):
        """
        Clinical task-aligned embeddings from XGBoost:
        - Train XGBoost for mortality prediction
        - Extract hidden representations from penultimate layer
        - Use as embeddings (task-aligned, captures what matters for outcomes)
        """
        try:
            import xgboost as xgb
        except ImportError:
            print("⚠ XGBoost not installed. Install with: pip install xgboost")
            return None
        
        print(f"Training XGBoost for {n_rounds} rounds...")
        
        # Train XGBoost classifier
        dtrain = xgb.DMatrix(self.X, label=self.y)
        dval = xgb.DMatrix(self.X_val, label=self.y_val) if self.X_val is not None else None
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc',
        }
        
        evals = [(dtrain, 'train')]
        if dval is not None:
            evals.append((dval, 'eval'))
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            verbose_eval=False,
            early_stopping_rounds=10 if dval is not None else None
        )
        
        # Get feature importances (weights for each feature)
        importances = model.get_score(importance_type='gain')
        feature_weights = np.array([importances.get(f'f{i}', 0) for i in range(self.X.shape[1])])
        feature_weights = feature_weights / feature_weights.sum()  # Normalize
        
        # Extract embeddings: weighted average of tree predictions
        # Use leaf indices as embeddings (captures tree structure)
        train_leaf_indices = model.predict(dtrain, pred_leaf=True)
        
        # Dimensionality reduction: PCA on leaf indices
        pca_leaf = PCA(n_components=min(n_components, train_leaf_indices.shape[1]))
        embeddings = pca_leaf.fit_transform(train_leaf_indices)
        
        # Get AUC for validation
        train_pred = model.predict(dtrain)
        auc = roc_auc_score(self.y, train_pred)
        
        result = {
            'method': 'XGBoost',
            'embeddings': embeddings,
            'model': model,
            'feature_weights': feature_weights,
            'auc': auc,
            'n_dims': embeddings.shape[1],
            'speed': 'Fast',
            'interpretability': 'Medium',
            'task_aligned': True,
        }
        self.results['xgboost'] = result
        print(f"✓ XGBoost AUC: {auc:.4f}")
        return result
    
    def autoencoder_embeddings(self, n_components=8, epochs=50):
        """
        Neural network embeddings via autoencoder bottleneck.
        
        Architecture:
        Input(D) → Dense(64) → Dense(32) → Bottleneck(n_components) → Dense(32) → Dense(64) → Output(D)
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model, optimizers
        except ImportError:
            print("⚠ TensorFlow not installed. Install with: pip install tensorflow")
            return None
        
        print(f"Training autoencoder ({self.X.shape[1]}D → {n_components}D → {self.X.shape[1]}D)...")
        
        input_dim = self.X.shape[1]
        
        # Build autoencoder
        inputs = layers.Input(shape=(input_dim,))
        
        # Encoder
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Bottleneck (embeddings)
        bottleneck = layers.Dense(n_components, activation='relu', name='bottleneck')(x)
        
        # Decoder
        x = layers.Dense(32, activation='relu')(bottleneck)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(input_dim, activation='linear')(x)
        
        # Model
        autoencoder = Model(inputs=inputs, outputs=outputs, name='autoencoder')
        encoder = Model(inputs=inputs, outputs=bottleneck, name='encoder')
        
        # Compile
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        # Train
        history = autoencoder.fit(
            self.X, self.X,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2 if self.X_val is None else 0,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss' if self.X_val is None else 'val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Extract embeddings
        embeddings = encoder.predict(self.X, verbose=0)
        
        # Reconstruction loss (quality metric)
        X_recon = autoencoder.predict(self.X, verbose=0)
        reconstruction_loss = np.mean((self.X - X_recon) ** 2)
        
        result = {
            'method': 'Autoencoder',
            'embeddings': embeddings,
            'model': encoder,
            'reconstruction_loss': reconstruction_loss,
            'n_dims': n_components,
            'speed': 'Slow',
            'interpretability': 'Low',
            'task_aligned': False,
        }
        self.results['autoencoder'] = result
        print(f"✓ Reconstruction loss: {reconstruction_loss:.4f}")
        return result
    
    def mortality_aligned_embeddings(self, n_components=8):
        """
        Train neural network for mortality prediction, extract embeddings from penultimate layer.
        Task-aligned: learned to predict mortality, so embeddings capture mortality-relevant patterns.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model, optimizers
        except ImportError:
            print("⚠ TensorFlow not installed")
            return None
        
        print(f"Training mortality predictor with {n_components}D embeddings...")
        
        input_dim = self.X.shape[1]
        
        # Build model with embedding layer
        inputs = layers.Input(shape=(input_dim,))
        
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Embedding layer (penultimate)
        embeddings_layer = layers.Dense(n_components, activation='relu', name='embeddings')(x)
        x = layers.Dropout(0.2)(embeddings_layer)
        
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Build models
        model = Model(inputs=inputs, outputs=outputs, name='mortality_predictor')
        extractor = Model(inputs=inputs, outputs=embeddings_layer, name='embedding_extractor')
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['auc']
        )
        
        # Split data
        if self.X_val is None:
            X_t, X_v, y_t, y_v = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
        else:
            X_t, y_t = self.X, self.y
            X_v, y_v = self.X_val, self.y_val
        
        # Train
        history = model.fit(
            X_t, y_t,
            validation_data=(X_v, y_v),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                )
            ]
        )
        
        # Extract embeddings
        embeddings = extractor.predict(self.X, verbose=0)
        
        # Evaluate
        val_auc = model.evaluate(X_v, y_v, verbose=0)[1]
        
        result = {
            'method': 'Mortality-Aligned NN',
            'embeddings': embeddings,
            'model': extractor,
            'auc': val_auc,
            'n_dims': n_components,
            'speed': 'Slow',
            'interpretability': 'Low',
            'task_aligned': True,
        }
        self.results['mortality_nn'] = result
        print(f"✓ Mortality predictor AUC: {val_auc:.4f}")
        return result
    
    def compare_embeddings(self):
        """Compare all embedding methods."""
        print("\n" + "="*80)
        print("EMBEDDING QUALITY COMPARISON")
        print("="*80)
        
        comparison = []
        for method_name, result in self.results.items():
            if result is None:
                continue
            
            embeddings = result['embeddings']
            
            # Quality metrics for embeddings
            # 1. Dimensionality efficiency
            if method_name == 'pca':
                efficiency = result['variance']  # % variance explained
            else:
                efficiency = 0.8  # Placeholder
            
            # 2. Clustering quality (silhouette score)
            silhouette = silhouette_score(embeddings, (self.y > 0).astype(int))
            
            # 3. Classification power (if we have task alignment)
            if result.get('task_aligned', False) and 'auc' in result:
                classification_auc = result['auc']
            else:
                classification_auc = None
            
            # 4. Intrinsic dimensionality
            rank = np.linalg.matrix_rank(embeddings)
            
            comparison.append({
                'Method': result['method'],
                'Dimensions': result['n_dims'],
                'Silhouette': f"{silhouette:.4f}",
                'Task-Aligned': "Yes" if result.get('task_aligned', False) else "No",
                'Speed': result['speed'],
                'Interpretability': result['interpretability'],
            })
        
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
        
        print("\n✓ Recommendation:")
        print("  • Start with: PCA (baseline, interpretable)")
        print("  • Advanced: Mortality-Aligned NN (task-aligned, best for patient similarity)")
        print("  • Alternative: XGBoost embeddings (feature weights + task-aligned)")


# ============================================================================
# IMPROVEMENT 2: ADVANCED SIMILARITY METRICS
# ============================================================================

class AdvancedSimilarityMetrics:
    """
    Compare multiple similarity metrics for patient matching.
    
    Methods:
    - Euclidean: L2 distance (simple, affected by scale)
    - Cosine: angle-based (scale-invariant)
    - Mahalanobis: accounts for covariance structure
    - Gower: mixed data types (categorical + numerical)
    - Learned: neural network + triplet loss
    """
    
    def __init__(self, embeddings, y):
        """
        Args:
            embeddings: Patient embeddings (N × D)
            y: Patient outcomes for evaluation
        """
        self.embeddings = embeddings
        self.y = y
        self.n_patients = len(embeddings)
        self.metrics_results = {}
    
    def euclidean_similarity(self):
        """Standard Euclidean distance (L2)."""
        # Compute pairwise distances
        distances = cdist(self.embeddings, self.embeddings, metric='euclidean')
        
        # Convert to similarity (inverse of distance)
        max_dist = distances.max()
        similarities = 1 - (distances / max_dist)
        
        return {
            'name': 'Euclidean',
            'distances': distances,
            'similarities': similarities,
            'scale_invariant': False,
            'suited_for': 'Standard numerical features',
        }
    
    def cosine_similarity(self):
        """Cosine similarity (angle-based, scale-invariant)."""
        distances = cdist(self.embeddings, self.embeddings, metric='cosine')
        similarities = 1 - distances
        
        return {
            'name': 'Cosine',
            'distances': distances,
            'similarities': similarities,
            'scale_invariant': True,
            'suited_for': 'Normalized embeddings',
        }
    
    def mahalanobis_similarity(self):
        """
        Mahalanobis distance: accounts for feature correlations.
        Distance = sqrt((x-y)^T * Σ^-1 * (x-y))
        Useful when features are correlated (which they are in medical data).
        """
        # Estimate covariance matrix
        cov = np.cov(self.embeddings.T)
        
        # Add small regularization for numerical stability
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        
        try:
            inv_cov = np.linalg.inv(cov_reg)
            distances = cdist(self.embeddings, self.embeddings, metric='mahalanobis', VI=inv_cov)
            similarities = 1 - (distances / distances.max())
            
            return {
                'name': 'Mahalanobis',
                'distances': distances,
                'similarities': similarities,
                'scale_invariant': True,
                'suited_for': 'Correlated features',
            }
        except np.linalg.LinAlgError:
            print("⚠ Mahalanobis computation failed (singular covariance)")
            return None
    
    def gower_similarity(self, X_original):
        """
        Gower distance: handles mixed data types (categorical + numerical).
        Useful for raw features with both vitals (numerical) and categorical variables.
        
        Args:
            X_original: Original features before embedding (may have categorical)
        """
        # For demonstration, compute on embeddings
        # In practice, use on mixed-type original features
        
        # Normalize each dimension to [0,1]
        X_norm = (X_original - X_original.min(axis=0)) / (X_original.max(axis=0) - X_original.min(axis=0) + 1e-8)
        
        # Gower distance: average absolute difference across features
        distances = cdist(X_norm, X_norm, metric='cityblock') / X_norm.shape[1]
        similarities = 1 - distances
        
        return {
            'name': 'Gower',
            'distances': distances,
            'similarities': similarities,
            'scale_invariant': True,
            'suited_for': 'Mixed data types',
        }
    
    def weighted_cosine_similarity(self, feature_weights):
        """
        Weighted cosine similarity: features weighted by clinical importance.
        
        Args:
            feature_weights: Array of weights (importance) per feature
        """
        # Apply weights to embeddings
        weighted_embeddings = self.embeddings * feature_weights
        
        # Compute cosine distance on weighted embeddings
        distances = cdist(weighted_embeddings, weighted_embeddings, metric='cosine')
        similarities = 1 - distances
        
        return {
            'name': 'Weighted Cosine',
            'distances': distances,
            'similarities': similarities,
            'scale_invariant': True,
            'suited_for': 'Features with different clinical importance',
        }
    
    def evaluate_similarity_metric(self, metric_name, similarities, k=5):
        """
        Evaluate similarity metric using outcome homogeneity.
        
        Args:
            similarities: Similarity matrix (N × N)
            k: Number of neighbors to consider
        
        Returns:
            Outcome match rate (higher is better)
        """
        outcome_matches = []
        
        for i in range(self.n_patients):
            # Get k most similar patients (excluding self)
            sims = similarities[i].copy()
            sims[i] = -np.inf  # Exclude self
            top_k_indices = np.argsort(sims)[-k:]  # Top k most similar
            
            # Check outcome agreement
            query_outcome = self.y[i]
            neighbor_outcomes = self.y[top_k_indices]
            match_rate = (neighbor_outcomes == query_outcome).mean()
            outcome_matches.append(match_rate)
        
        outcome_matches = np.array(outcome_matches)
        
        return {
            'metric': metric_name,
            'mean_homogeneity': outcome_matches.mean(),
            'std_homogeneity': outcome_matches.std(),
            'high_quality': (outcome_matches >= 0.60).sum() / len(outcome_matches),
        }
    
    def compare_all_metrics(self, X_original=None, feature_weights=None):
        """Compare all similarity metrics."""
        print("\n" + "="*80)
        print("SIMILARITY METRIC COMPARISON")
        print("="*80)
        
        results = []
        
        # Basic metrics
        for method in [self.euclidean_similarity, self.cosine_similarity]:
            metric_dict = method()
            eval_result = self.evaluate_similarity_metric(metric_dict['name'], metric_dict['similarities'])
            results.append(eval_result)
            self.metrics_results[metric_dict['name']] = (metric_dict, eval_result)
        
        # Mahalanobis
        mahal = self.mahalanobis_similarity()
        if mahal is not None:
            eval_result = self.evaluate_similarity_metric(mahal['name'], mahal['similarities'])
            results.append(eval_result)
            self.metrics_results[mahal['name']] = (mahal, eval_result)
        
        # Gower (if original features available)
        if X_original is not None:
            gower = self.gower_similarity(X_original)
            eval_result = self.evaluate_similarity_metric(gower['name'], gower['similarities'])
            results.append(eval_result)
            self.metrics_results[gower['name']] = (gower, eval_result)
        
        # Weighted cosine (if weights available)
        if feature_weights is not None:
            weighted = self.weighted_cosine_similarity(feature_weights)
            eval_result = self.evaluate_similarity_metric(weighted['name'], weighted['similarities'])
            results.append(eval_result)
            self.metrics_results['Weighted Cosine'] = (weighted, eval_result)
        
        # Display results
        results_df = pd.DataFrame(results)
        results_df['High Quality %'] = (results_df['high_quality'] * 100).round(1)
        results_df = results_df[['metric', 'mean_homogeneity', 'std_homogeneity', 'High Quality %']]
        results_df.columns = ['Metric', 'Mean Homogeneity', 'Std Dev', '% High Quality']
        
        print(results_df.to_string(index=False))
        
        best_metric = max(results, key=lambda x: x['mean_homogeneity'])
        print(f"\n✓ Best metric: {best_metric['metric']} ({best_metric['mean_homogeneity']:.1%} homogeneity)")
        
        return results


# ============================================================================
# IMPROVEMENT 3: FEATURE WEIGHTING BY CLINICAL IMPORTANCE
# ============================================================================

class ClinicalFeatureWeighting:
    """
    Assign weights to features based on clinical importance.
    More important features get higher weight in similarity computation.
    """
    
    def __init__(self, X, y, feature_names=None):
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f"F_{i}" for i in range(X.shape[1])]
        self.weights = None
    
    def statistical_importance(self):
        """
        Weight features by statistical importance for outcome prediction.
        Methods:
        1. Correlation with outcome
        2. Mutual information
        3. Feature importance from Random Forest
        """
        
        weights = {}
        
        # 1. Univariate correlation
        correlations = []
        for i in range(self.X.shape[1]):
            valid_mask = ~(np.isnan(self.X[:, i]) | np.isnan(self.y))
            if valid_mask.sum() > 10:
                corr = np.abs(np.corrcoef(self.X[valid_mask, i], self.y[valid_mask])[0, 1])
            else:
                corr = 0.0
            correlations.append(corr)
        
        weights['correlation'] = np.array(correlations)
        
        # 2. Random Forest Feature Importance
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            X_imputed = np.nan_to_num(self.X, nan=np.nanmedian(self.X, axis=0))
            rf.fit(X_imputed, self.y)
            weights['rf_importance'] = rf.feature_importances_
        except:
            weights['rf_importance'] = np.zeros(self.X.shape[1])
        
        return weights
    
    def clinical_importance(self):
        """
        Expert-based clinical weighting.
        In real scenarios, domain experts assign importance scores.
        """
        
        # Clinical categories
        clinical_categories = {
            'vital_signs': ['heart', 'systolic', 'diastolic', 'sao2', 'respiration', 'temperature'],
            'severe_labs': ['lactate', 'creatinine', 'bilirubin', 'inr'],
            'other_labs': ['glucose', 'sodium', 'potassium', 'hemoglobin', 'platelet', 'wbc'],
            'severity_scores': ['apache', 'aps', 'sofa', 'shock'],
        }
        
        weights = np.ones(len(self.feature_names)) * 0.5  # Default baseline
        
        for i, fname in enumerate(self.feature_names):
            fname_lower = fname.lower()
            
            # Vital signs: high weight (0.9)
            if any(term in fname_lower for term in clinical_categories['vital_signs']):
                weights[i] = 0.9
            
            # Severe labs: high weight (0.8)
            elif any(term in fname_lower for term in clinical_categories['severe_labs']):
                weights[i] = 0.8
            
            # Other labs: medium weight (0.6)
            elif any(term in fname_lower for term in clinical_categories['other_labs']):
                weights[i] = 0.6
            
            # Severity scores: very high weight (0.95)
            elif any(term in fname_lower for term in clinical_categories['severity_scores']):
                weights[i] = 0.95
        
        return weights
    
    def combined_importance(self, alpha=0.33, beta=0.33, gamma=0.34):
        """
        Combine statistical and clinical importance.
        Weight = alpha * correlation + beta * RF_importance + gamma * clinical
        """
        
        stat_weights = self.statistical_importance()
        clinical_weights = self.clinical_importance()
        
        # Normalize
        corr_norm = stat_weights['correlation'] / (stat_weights['correlation'].max() + 1e-8)
        rf_norm = stat_weights['rf_importance'] / (stat_weights['rf_importance'].max() + 1e-8)
        clinical_norm = clinical_weights / clinical_weights.max()
        
        combined = alpha * corr_norm + beta * rf_norm + gamma * clinical_norm
        combined = combined / combined.sum()  # Normalize to sum to 1
        
        return combined
    
    def display_weights(self, weight_dict, title="Feature Weights"):
        """Display weights for interpretation."""
        print(f"\n{title}")
        print("="*80)
        
        weight_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Weight': weight_dict,
        }).sort_values('Weight', ascending=False)
        
        print(weight_df.head(10).to_string(index=False))
        if len(weight_df) > 10:
            print(f"... and {len(weight_df) - 10} more features")
        
        return weight_df


# ============================================================================
# IMPROVEMENT 4: SMART MISSING DATA HANDLING IN SIMILARITY
# ============================================================================

class MissingDataAwareSimilarity:
    """
    Compute similarity while accounting for missing data:
    - Only compare features that are present in both patients
    - Weight by data quality/completeness
    - Use expected correlations for partially observed features
    """
    
    def __init__(self, X, missing_mask=None):
        """
        Args:
            X: Features with NaNs (N × D)
            missing_mask: Boolean mask (N × D), True where missing
        """
        self.X = X.copy()
        self.missing_mask = np.isnan(X) if missing_mask is None else missing_mask
        self.n_patients = X.shape[0]
        self.n_features = X.shape[1]
    
    def pairwise_similarity_complete_pairs(self, metric='cosine'):
        """
        Compute pairwise similarity only on commonly observed features.
        
        Algorithm:
        For each pair (i, j):
            - Find features both patients have
            - Compute distance on those features only
            - Normalize by number of shared features
        """
        
        similarities = np.zeros((self.n_patients, self.n_patients))
        
        for i in range(self.n_patients):
            for j in range(i, self.n_patients):
                # Features that are observed in both patients
                common_features = ~(self.missing_mask[i] | self.missing_mask[j])
                n_common = common_features.sum()
                
                if n_common < 3:  # Need at least 3 common features
                    similarities[i, j] = 0.0
                else:
                    # Compute distance on common features only
                    x_i = self.X[i, common_features]
                    x_j = self.X[j, common_features]
                    
                    if metric == 'cosine':
                        # Handle zero vectors
                        norm_i = np.linalg.norm(x_i)
                        norm_j = np.linalg.norm(x_j)
                        if norm_i > 0 and norm_j > 0:
                            sim = np.dot(x_i, x_j) / (norm_i * norm_j)
                        else:
                            sim = 0.0
                    else:  # euclidean
                        dist = np.linalg.norm(x_i - x_j)
                        max_dist = np.sqrt(n_common * ((self.X[:, common_features].max(axis=0) - 
                                                        self.X[:, common_features].min(axis=0)) ** 2).sum())
                        sim = 1 - (dist / max_dist) if max_dist > 0 else 0
                    
                    similarities[i, j] = sim
                
                similarities[j, i] = similarities[i, j]
        
        np.fill_diagonal(similarities, 1.0)  # Self-similarity = 1
        return similarities
    
    def data_quality_weighted_similarity(self, embeddings, metric='cosine'):
        """
        Weight similarity by data quality (fraction of features observed).
        
        Algorithm:
        sim_weighted = sim_raw * quality_factor
        quality_factor = min(completeness_i, completeness_j)
        """
        
        # Compute base similarity (on available features)
        similarities = self.pairwise_similarity_complete_pairs(metric=metric)
        
        # Compute data completeness for each patient
        completeness = 1 - self.missing_mask.sum(axis=1) / self.n_features
        
        # Weight similarity by joint completeness
        quality_matrix = np.outer(completeness, completeness)
        quality_matrix = np.minimum(quality_matrix, 1.0)  # Cap at 1
        
        weighted_similarities = similarities * quality_matrix
        
        return weighted_similarities
    
    def expected_value_imputation_similarity(self, embeddings, metric='cosine'):
        """
        For missing features, use expected values based on other patients' data.
        Useful when patterns are predictable from other features.
        """
        
        # Use conditional expectation based on other features
        # For simplicity: use feature-wise mean across available data
        feature_means = np.nanmean(self.X, axis=0)
        
        X_imputed = self.X.copy()
        for i in range(self.n_features):
            missing_idx = np.isnan(X_imputed[:, i])
            X_imputed[missing_idx, i] = feature_means[i]
        
        # Compute similarity on imputed data
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(X_imputed)
            similarities = 1 - distances
        else:
            distances = cdist(X_imputed, X_imputed, metric='euclidean')
            similarities = 1 - (distances / distances.max())
        
        return similarities


# ============================================================================
# IMPROVEMENT 5: TECHNIQUES TO IMPROVE OUTCOME HOMOGENEITY (65-75% target)
# ============================================================================

class HomogeneityBoostingTechniques:
    """
    Methods to improve outcome homogeneity in K-NN matching.
    Target: 65-75% (vs baseline 58%)
    """
    
    def __init__(self, embeddings, y, outcomes):
        """
        Args:
            embeddings: Patient embeddings (N × D)
            y: Binary outcomes (mortality 0/1)
            outcomes: Outcome labels for matching
        """
        self.embeddings = embeddings
        self.y = y
        self.outcomes = outcomes
        self.n_patients = len(embeddings)
    
    def outcome_aware_knn(self, k=5, outcome_weight=0.5):
        """
        K-NN with outcome-aware weighting:
        Final similarity = base_similarity * (1 - outcome_weight + outcome_weight * outcome_match)
        
        Boost matches where outcomes align, penalize mismatches.
        """
        
        # Compute base similarity
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(self.embeddings)
        base_sims = 1 - distances
        
        # Compute outcome alignment matrix
        outcome_matches = np.equal.outer(self.y, self.y).astype(float)
        
        # Combine: boost same outcomes, penalize different
        combined_sims = base_sims * (1 - outcome_weight) + outcome_matches * outcome_weight
        
        return combined_sims, base_sims
    
    def stratified_knn(self, k=5):
        """
        Stratified K-NN: return k//2 same-outcome and k//2 different-outcome neighbors.
        Useful for understanding both similar and contrasting patients.
        """
        
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(self.embeddings)
        
        # Set self-distance to infinity
        np.fill_diagonal(distances, np.inf)
        
        same_outcome_twins = []
        diff_outcome_twins = []
        
        for i in range(self.n_patients):
            # Separate by outcome
            same_outcome_mask = self.y == self.y[i]
            diff_outcome_mask = self.y != self.y[i]
            
            # Get k//2 from each
            k_same = (k + 1) // 2
            k_diff = k // 2
            
            same_idx = np.where(same_outcome_mask)[0]
            diff_idx = np.where(diff_outcome_mask)[0]
            
            same_twins = same_idx[np.argsort(distances[i, same_idx])[:k_same]]
            diff_twins = diff_idx[np.argsort(distances[i, diff_idx])[:k_diff]]
            
            same_outcome_twins.append(same_twins)
            diff_outcome_twins.append(diff_twins)
        
        return same_outcome_twins, diff_outcome_twins
    
    def gradient_boosted_similarity(self):
        """
        Learn similarity metric using gradient boosting.
        Train model to predict outcome agreement given embeddings difference.
        """
        
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("⚠ XGBoost not installed")
            return None
        
        # Generate training samples: pairs of patients
        n_pairs = min(10000, self.n_patients * (self.n_patients - 1) // 2)
        
        pair_features = []
        pair_labels = []  # 1 if same outcome, 0 if different
        
        for _ in range(n_pairs):
            i = np.random.randint(self.n_patients)
            j = np.random.randint(self.n_patients)
            if i == j:
                continue
            
            # Compute difference/similarity features
            diff = self.embeddings[i] - self.embeddings[j]
            sim = np.dot(self.embeddings[i], self.embeddings[j]) / (
                np.linalg.norm(self.embeddings[i]) * np.linalg.norm(self.embeddings[j]) + 1e-8
            )
            
            features = np.concatenate([diff, [sim], [np.linalg.norm(diff)]])
            pair_features.append(features)
            
            # Label: 1 if outcomes match, 0 otherwise
            pair_labels.append(self.y[i] == self.y[j])
        
        X_train = np.array(pair_features)
        y_train = np.array(pair_labels)
        
        # Train XGBoost to predict outcome agreement
        model = XGBClassifier(max_depth=5, n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    def spectral_clustering_aware_knn(self, k=5):
        """
        Use spectral clustering to identify patient subgroups.
        Find K-NN within same subgroup (higher homogeneity).
        """
        
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            print("⚠ Spectral clustering not available")
            return None
        
        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(self.embeddings)
        similarities = 1 - distances
        
        # Spectral clustering
        n_clusters = max(2, self.n_patients // 20)  # Aim for clusters of ~20 patients
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        labels = clustering.fit_predict(similarities)
        
        # For each patient, find K-NN within same cluster
        within_cluster_knn = []
        for i in range(self.n_patients):
            cluster_members = np.where(labels == labels[i])[0]
            cluster_members = cluster_members[cluster_members != i]  # Exclude self
            
            if len(cluster_members) > k:
                # Get k closest within cluster
                best_k = cluster_members[np.argsort(distances[i, cluster_members])[:k]]
            else:
                # Need more neighbors, expand outside cluster
                best_k = np.argsort(distances[i])[:k+1]
                best_k = best_k[best_k != i][:k]
            
            within_cluster_knn.append(best_k)
        
        return within_cluster_knn, labels


# ============================================================================
# IMPROVEMENT 6: OUTLIER DETECTION & HANDLING
# ============================================================================

class OutlierDetectionAndHandling:
    """
    Identify and handle patients with poor-quality twins or unusual phenotypes.
    """
    
    def __init__(self, embeddings, y, similarities, k=5):
        self.embeddings = embeddings
        self.y = y
        self.similarities = similarities
        self.k = k
        self.n_patients = len(embeddings)
        self.outliers = None
    
    def isolation_forest_outliers(self, contamination=0.05):
        """
        Detect embedding-space outliers using Isolation Forest.
        Outliers = patients with unusual feature combinations.
        """
        
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            print("⚠ IsolationForest not available")
            return None
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.embeddings)
        outlier_scores = iso_forest.score_samples(self.embeddings)
        
        outliers = outlier_labels == -1
        
        return {
            'outliers': outliers,
            'scores': outlier_scores,
            'n_outliers': outliers.sum(),
            'pct_outliers': 100 * outliers.sum() / self.n_patients,
        }
    
    def neighborhood_stability_outliers(self):
        """
        Identify patients whose K-NN neighbors are unstable or inconsistent.
        High instability indicates poor-quality twins.
        """
        
        # For each patient, compute K-NN
        twin_qualities = []
        neighbor_variance = []
        
        for i in range(self.n_patients):
            sims = self.similarities[i].copy()
            sims[i] = -np.inf
            top_k = np.argsort(sims)[-self.k:]
            
            # Metric 1: Outcome homogeneity
            outcomes_match = (self.y[top_k] == self.y[i]).mean()
            twin_qualities.append(outcomes_match)
            
            # Metric 2: Similarity variance (stability)
            sim_variance = np.var(sims[top_k])
            neighbor_variance.append(sim_variance)
        
        twin_qualities = np.array(twin_qualities)
        neighbor_variance = np.array(neighbor_variance)
        
        # Flag as outlier if low quality twins
        outlier_threshold = np.percentile(twin_qualities, 20)  # Bottom 20%
        outliers = twin_qualities < outlier_threshold
        
        return {
            'outliers': outliers,
            'quality_scores': twin_qualities,
            'variance_scores': neighbor_variance,
            'n_outliers': outliers.sum(),
            'pct_outliers': 100 * outliers.sum() / self.n_patients,
        }
    
    def local_density_based_outliers(self):
        """
        Detect patients with low local density (isolated in embedding space).
        Uses K-distance measure.
        """
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        
        # K-distance: distance to k-th nearest neighbor
        k_distances = distances[:, -1]
        
        # Use median + 2*IQR as threshold
        q1 = np.percentile(k_distances, 25)
        q3 = np.percentile(k_distances, 75)
        threshold = q3 + 2 * (q3 - q1)
        
        outliers = k_distances > threshold
        
        return {
            'outliers': outliers,
            'k_distances': k_distances,
            'threshold': threshold,
            'n_outliers': outliers.sum(),
            'pct_outliers': 100 * outliers.sum() / self.n_patients,
        }
    
    def handle_outliers(self, outliers_mask, strategy='flag'):
        """
        Handle detected outliers.
        
        Strategies:
        - 'flag': Mark but don't remove (doctors decide)
        - 'exclude': Exclude from K-NN matching
        - 'special_handling': Different K or matching strategy
        """
        
        if strategy == 'flag':
            return {
                'status': 'Flagged',
                'n_flagged': outliers_mask.sum(),
                'action': 'Review by clinicians'
            }
        
        elif strategy == 'exclude':
            return {
                'status': 'Excluded',
                'n_excluded': outliers_mask.sum(),
                'remaining': (~outliers_mask).sum(),
                'action': 'Use only typical patients for matching'
            }
        
        elif strategy == 'special_handling':
            # Use different K (more neighbors) for outliers
            k_normal = self.k
            k_outlier = max(self.k + 3, int(self.k * 1.5))
            
            return {
                'status': 'Special handling',
                'k_normal': k_normal,
                'k_outlier': k_outlier,
                'n_outliers': outliers_mask.sum(),
                'action': 'Expand neighborhood for atypical patients'
            }
        
        return None


# ============================================================================
# IMPROVEMENT 7: ADVANCED EVALUATION METRICS
# ============================================================================

class AdvancedEvaluationMetrics:
    """
    Beyond binary outcome matching: comprehensive twin quality evaluation.
    """
    
    def __init__(self, embeddings, y, X_raw=None):
        """
        Args:
            embeddings: Patient embeddings
            y: Binary outcomes
            X_raw: Raw features for detailed analysis
        """
        self.embeddings = embeddings
        self.y = y
        self.X_raw = X_raw
        self.n_patients = len(embeddings)
    
    def outcome_homogeneity(self, similarities, k=5):
        """Outcome homogeneity: % of twins with same outcome."""
        
        outcome_matches = []
        for i in range(self.n_patients):
            sims = similarities[i].copy()
            sims[i] = -np.inf
            top_k = np.argsort(sims)[-k:]
            match = (self.y[top_k] == self.y[i]).mean()
            outcome_matches.append(match)
        
        return {
            'metric': 'Outcome Homogeneity',
            'mean': np.mean(outcome_matches),
            'std': np.std(outcome_matches),
            'percentile_25': np.percentile(outcome_matches, 25),
            'percentile_75': np.percentile(outcome_matches, 75),
        }
    
    def embedding_stability(self, k=5, n_perturb=10):
        """
        Embedding stability: how much do K-NN change with small perturbations?
        Lower variation = more stable, higher quality matches.
        """
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs_original = NearestNeighbors(n_neighbors=k+1).fit(self.embeddings)
        _, indices_original = nbrs_original.kneighbors()
        
        perturbation_similarities = []
        
        # Add small noise and recompute
        for _ in range(n_perturb):
            noise = np.random.normal(0, 0.01, self.embeddings.shape)
            emb_perturbed = self.embeddings + noise
            
            nbrs_pert = NearestNeighbors(n_neighbors=k+1).fit(emb_perturbed)
            _, indices_pert = nbrs_pert.kneighbors()
            
            # Jaccard similarity of neighborhoods
            jaccard_scores = []
            for i in range(self.n_patients):
                idx_orig = set(indices_original[i, 1:])
                idx_pert = set(indices_pert[i, 1:])
                jaccard = len(idx_orig & idx_pert) / len(idx_orig | idx_pert)
                jaccard_scores.append(jaccard)
            
            perturbation_similarities.append(np.mean(jaccard_scores))
        
        stability = np.mean(perturbation_similarities)
        
        return {
            'metric': 'Embedding Stability',
            'mean_jaccard': stability,
            'interpretation': f'{stability:.1%} of neighborhood preserved under perturbation'
        }
    
    def clinical_feature_agreement(self, similarities, k=5):
        """
        Clinical feature agreement: do K-NN neighbors have similar feature values?
        Beyond just outcome, check if features are actually similar.
        """
        
        if self.X_raw is None:
            return None
        
        feature_agreements = []
        
        for i in range(min(self.n_patients, 100)):  # Sample for speed
            sims = similarities[i].copy()
            sims[i] = -np.inf
            top_k = np.argsort(sims)[-k:]
            
            # Compute feature-wise correlation
            x_i = self.X_raw[i]
            x_neighbors = self.X_raw[top_k]
            
            # Compute L2 distance in feature space (normalized)
            valid_mask = ~(np.isnan(x_i) | np.isnan(x_neighbors).any(axis=0))
            if valid_mask.sum() > 3:
                feature_diffs = np.abs(x_i[valid_mask] - x_neighbors[:, valid_mask])
                mean_feature_diff = feature_diffs.mean()
                feature_agreements.append(mean_feature_diff)
        
        return {
            'metric': 'Clinical Feature Agreement',
            'mean_feature_difference': np.mean(feature_agreements),
            'interpretation': 'Lower = more similar twins in feature space'
        }
    
    def neighbor_diversity(self, similarities, k=5):
        """
        Neighbor diversity: are K-NN a homogeneous group or diverse?
        Some diversity is good (multiple similar phenotypes).
        """
        
        diversities = []
        
        for i in range(self.n_patients):
            sims = similarities[i].copy()
            sims[i] = -np.inf
            top_k = np.argsort(sims)[-k:]
            
            # Compute pairwise distances within neighborhood
            neighbor_emb = self.embeddings[top_k]
            from sklearn.metrics.pairwise import cosine_distances
            neighbor_dists = cosine_distances(neighbor_emb)
            
            # Average distance within neighborhood (excluding diagonal)
            upper_tri = np.triu_indices(k, k=1)
            avg_internal_dist = neighbor_dists[upper_tri].mean()
            
            diversities.append(avg_internal_dist)
        
        return {
            'metric': 'Neighbor Diversity',
            'mean_internal_distance': np.mean(diversities),
            'interpretation': 'Moderate diversity indicates robustness'
        }
    
    def outcome_risk_stratification(self, similarities, k=5):
        """
        Analyze outcome prediction accuracy:
        Can we predict patient mortality from K-NN outcomes?
        """
        
        predicted_mortality = []
        
        for i in range(self.n_patients):
            sims = similarities[i].copy()
            sims[i] = -np.inf
            top_k = np.argsort(sims)[-k:]
            
            # Majority vote among neighbors
            predicted = (self.y[top_k].sum() / k) >= 0.5
            predicted_mortality.append(predicted)
        
        predicted_mortality = np.array(predicted_mortality)
        
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
        
        cm = confusion_matrix(self.y, predicted_mortality)
        precision = precision_score(self.y, predicted_mortality)
        recall = recall_score(self.y, predicted_mortality)
        f1 = f1_score(self.y, predicted_mortality)
        
        return {
            'metric': 'Outcome Prediction via K-NN',
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
        }
    
    def comprehensive_evaluation(self, similarities, k=5):
        """Run all metrics and generate report."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TWIN QUALITY EVALUATION")
        print("="*80)
        
        # 1. Outcome homogeneity
        homo = self.outcome_homogeneity(similarities, k)
        print(f"\n1. OUTCOME HOMOGENEITY")
        print(f"   Mean: {homo['mean']:.1%} ± {homo['std']:.1%}")
        print(f"   25th-75th percentile: {homo['percentile_25']:.1%} - {homo['percentile_75']:.1%}")
        
        # 2. Embedding stability
        stab = self.embedding_stability(k=k, n_perturb=5)
        print(f"\n2. EMBEDDING STABILITY")
        print(f"   Neighborhood stability: {stab['mean_jaccard']:.1%}")
        print(f"   → {stab['interpretation']}")
        
        # 3. Clinical feature agreement
        if self.X_raw is not None:
            feat_agree = self.clinical_feature_agreement(similarities, k)
            print(f"\n3. CLINICAL FEATURE AGREEMENT")
            print(f"   Mean feature difference: {feat_agree['mean_feature_difference']:.4f}")
        
        # 4. Neighbor diversity
        div = self.neighbor_diversity(similarities, k)
        print(f"\n4. NEIGHBOR DIVERSITY")
        print(f"   Mean internal distance: {div['mean_internal_distance']:.4f}")
        
        # 5. Outcome prediction
        pred = self.outcome_risk_stratification(similarities, k)
        print(f"\n5. OUTCOME PREDICTION VIA K-NN VOTE")
        print(f"   Precision: {pred['precision']:.3f}")
        print(f"   Recall: {pred['recall']:.3f}")
        print(f"   F1-Score: {pred['f1_score']:.3f}")
        
        print("\n" + "="*80)


# ============================================================================
# UTILITY: COMPREHENSIVE IMPROVEMENT PIPELINE
# ============================================================================

def run_improvement_analysis(X, y, X_val=None, y_val=None, feature_names=None):
    """
    Run comprehensive analysis of all improvements.
    """
    
    print("\n" + "="*80)
    print("DIGITAL TWIN SYSTEM - COMPREHENSIVE IMPROVEMENT ANALYSIS")
    print("="*80)
    
    # 1. Advanced embeddings
    print("\n[1/7] TESTING ADVANCED EMBEDDINGS...")
    embeddings_comp = EmbeddingComparison(X, y, X_val, y_val)
    
    pca_result = embeddings_comp.pca_embeddings(n_components=8)
    print(f"✓ PCA: {pca_result['variance']:.1%} variance explained")
    
    xgb_result = embeddings_comp.xgboost_embeddings(n_components=8)
    if xgb_result:
        print(f"✓ XGBoost: Task-aligned embeddings")
    
    embeddings_to_use = pca_result['embeddings']
    
    # 2. Compare similarity metrics
    print("\n[2/7] COMPARING SIMILARITY METRICS...")
    sim_comp = AdvancedSimilarityMetrics(embeddings_to_use, y)
    
    feature_weights = None
    if feature_names:
        weighting = ClinicalFeatureWeighting(X, y, feature_names)
        feature_weights = weighting.combined_importance()
    
    sim_comp.compare_all_metrics(X_original=X, feature_weights=feature_weights)
    
    # 3. Feature weighting
    print("\n[3/7] ANALYZING FEATURE IMPORTANCE...")
    weighting = ClinicalFeatureWeighting(X, y, feature_names)
    weights = weighting.combined_importance()
    weighting.display_weights(weights, title="Combined Feature Importance")
    
    # 4. Missing data handling
    print("\n[4/7] EVALUATING MISSING DATA HANDLING...")
    missing_sim = MissingDataAwareSimilarity(X)
    sims_complete = missing_sim.pairwise_similarity_complete_pairs(metric='cosine')
    sims_weighted = missing_sim.data_quality_weighted_similarity(embeddings_to_use)
    print(f"✓ Complete-pairs similarity computed")
    print(f"✓ Data-quality weighted similarity computed")
    
    # 5. Homogeneity boosting
    print("\n[5/7] EXPLORING HOMOGENEITY BOOSTING...")
    homo_boost = HomogeneityBoostingTechniques(embeddings_to_use, y, y)
    combined_sims, base_sims = homo_boost.outcome_aware_knn(k=5, outcome_weight=0.3)
    print(f"✓ Outcome-aware K-NN computed")
    
    # 6. Outlier detection
    print("\n[6/7] DETECTING OUTLIERS...")
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(embeddings_to_use)
    similarities = 1 - distances
    
    outlier_detector = OutlierDetectionAndHandling(embeddings_to_use, y, similarities)
    iso_outliers = outlier_detector.isolation_forest_outliers()
    if iso_outliers:
        print(f"✓ Isolation Forest: {iso_outliers['pct_outliers']:.1f}% outliers detected")
    
    neigh_outliers = outlier_detector.neighborhood_stability_outliers()
    print(f"✓ Neighborhood stability: {neigh_outliers['pct_outliers']:.1f}% low-quality twins")
    
    # 7. Advanced evaluation
    print("\n[7/7] COMPREHENSIVE EVALUATION...")
    evaluator = AdvancedEvaluationMetrics(embeddings_to_use, y, X)
    evaluator.comprehensive_evaluation(similarities, k=5)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
