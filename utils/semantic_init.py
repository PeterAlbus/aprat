import torch
import clip
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import logging

def get_semantic_initialization(class_names, pool_size, target_dim, device='cuda'):
    """
    Implements Semantic-Anchored Initialization (SAI).
    
    Args:
        class_names (list): List of class names.
        pool_size (int): Number of clusters (M).
        target_dim (int): Dimension of the visual features (e.g., 768).
        device (str): Device to run CLIP on.
        
    Returns:
        torch.Tensor: Initialized Adapter Keys of shape (pool_size, target_dim).
    """
    logging.info("Generating Semantic-Anchored Initialization (SAI)...")
    
    # Check if class_names is valid
    if not class_names:
        logging.warning("No class names provided for SAI. Falling back to random initialization.")
        return None

    # 1. Text Embedding Extraction (文本特征提取)
    # Load CLIP model
    try:
        model, preprocess = clip.load("ViT-B/16", device=device)
    except Exception as e:
        logging.error(f"Failed to load CLIP: {e}")
        return None
        
    model.eval()
    
    # Tokenize class names
    # Handle potentially large number of classes by batching if necessary, but typically ok.
    # Add simple prompt template
    prompts = [f"a photo of a {c}" for c in class_names]
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        
    # Normalize text features (Generated text features need to be Normalized)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features_np = text_features.cpu().numpy()
    
    # 2. Semantic Clustering (语义聚类)
    logging.info(f"Clustering {len(class_names)} classes into {pool_size} centroids.")
    kmeans = KMeans(n_clusters=pool_size, random_state=42)
    kmeans.fit(text_features_np)
    centroids = kmeans.cluster_centers_ # Shape: (pool_size, clip_dim)
    
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32).to(device)
    
    # 3. Dimension Alignment (维度对齐)
    clip_dim = centroids_tensor.shape[1]
    
    if clip_dim != target_dim:
        logging.info(f"Aligning dimensions from {clip_dim} to {target_dim}...")
        # Use Padding as it preserves the original information without learning a projection
        if target_dim > clip_dim:
            # Padding with zeros
            padding = torch.zeros(pool_size, target_dim - clip_dim, device=device)
            centroids_aligned = torch.cat([centroids_tensor, padding], dim=1)
        else:
            # PCA (using sklearn) for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim)
            centroids_aligned_np = pca.fit_transform(centroids)
            centroids_aligned = torch.tensor(centroids_aligned_np, dtype=torch.float32).to(device)
    else:
        centroids_aligned = centroids_tensor
        
    # Normalize the centroids as well, as Keys are usually normalized in attention mechanisms,
    # though the prompt keys in APART might be used differently. 
    # The original initialization was uniform -1, 1.
    # Text features are normalized. Centroids might not be unit norm.
    # Let's keep them as is from KMeans/Padding, as the user didn't specify normalization of the final keys.
    # But text features were normalized before KMeans.
    
    return centroids_aligned.cpu()
