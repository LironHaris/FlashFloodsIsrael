import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def run_pca_workflow(input_path, output_dir, variance_threshold=0.95):
    """
    Perform standardization and PCA on cleaned attributes to identify key features and variance.
    Filters components to reach the specified variance threshold.
    """
    df = pd.read_csv(input_path)
    ids = df['gauge_id']
    features = df.drop(columns=['gauge_id'])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    pca = PCA(n_components=variance_threshold)
    projected = pca.fit_transform(scaled_data)
    
   
    n_components_found = pca.n_components_
    pca_cols = [f'PC{i+1}' for i in range(n_components_found)]
    
    variance_summary = pd.DataFrame({
        'Component': pca_cols,
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_summary.to_csv(os.path.join(output_dir, 'pca_variance_summary.csv'), index=False)
    
    loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=pca_cols)
    loadings.to_csv(os.path.join(output_dir, 'pca_loadings.csv'))
    
    projected_df = pd.DataFrame(projected, columns=pca_cols)
    projected_df.insert(0, 'gauge_id', ids.values)
    projected_df.to_csv(os.path.join(output_dir, 'pca_projected_data.csv'), index=False)
    
    print(f"PCA completed. {n_components_found} components explain {variance_threshold*100}% of variance.")

def main():
    """
    Load processed attributes and generate PCA results.
    """
    PROCESSED_DIR = os.path.join('data', 'processed')
    PCA_OUT_DIR = os.path.join(PROCESSED_DIR, 'pca_results')
    os.makedirs(PCA_OUT_DIR, exist_ok=True)
    
    clean_data_path = os.path.join(PROCESSED_DIR, 'static_attributes_nh.csv')
    
    if os.path.exists(clean_data_path):
        run_pca_workflow(clean_data_path, PCA_OUT_DIR, variance_threshold=0.95)
    else:
        print(f"File not found: {clean_data_path}")

if __name__ == "__main__":
    main()