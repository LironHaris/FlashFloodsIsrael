import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def run_pca_workflow(input_path, output_dir, variance_threshold):
    """
    Perform standardization and PCA on cleaned attributes to explore feature variance.
    Filters components dynamically to reach the cumulative variance threshold from config.
    """
    df = pd.read_csv(input_path)
    ids = df['gauge_id']
    features = df.drop(columns=['gauge_id'])
    
    # 1. Standardize features (Z-score normalization)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    # 2. Fit PCA using the specified variance threshold from config
    pca = PCA(n_components=variance_threshold)
    projected = pca.fit_transform(scaled_data)
    
    n_components_found = pca.n_components_
    pca_cols = [f'PC{i+1}' for i in range(n_components_found)]
    
    # 3. Save Variance Summary Report
    variance_summary = pd.DataFrame({
        'Component': pca_cols,
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_summary.to_csv(os.path.join(output_dir, 'pca_variance_summary.csv'), index=False)
    
    # 4. Save Loadings Report (Feature weights per Principal Component)
    loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=pca_cols)
    loadings.to_csv(os.path.join(output_dir, 'pca_loadings.csv'))
    
    # 5. Save Projected Data (The dataset transformed into PCA space)
    projected_df = pd.DataFrame(projected, columns=pca_cols)
    projected_df.insert(0, 'gauge_id', ids.values)
    projected_df.to_csv(os.path.join(output_dir, 'pca_projected_data.csv'), index=False)
    
    print(f"[INFO] PCA completed. {n_components_found} components explain {variance_threshold*100}% of the variance.")

def main(config):
    """
    Load clean static attributes and configuration parameters to run the PCA analysis.
    """
    clean_data_path = config['static_attributes_file']
    pca_out_dir = config['pca_output_dir']
    
    # Extract variance threshold from configuration, default to 0.95 if not found
    variance_threshold = config.get('pca_variance_threshold', 0.95)
    
    os.makedirs(pca_out_dir, exist_ok=True)
    
    if os.path.exists(clean_data_path):
        run_pca_workflow(clean_data_path, pca_out_dir, variance_threshold)
    else:
        print(f"[ERROR] Clean static attributes file not found at: {clean_data_path}")

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yml"
    yaml_config = load_config(CONFIG_PATH)
    
    main(yaml_config)