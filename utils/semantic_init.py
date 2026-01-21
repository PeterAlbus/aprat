import torch
import clip
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import logging
from sklearn.decomposition import PCA

def get_clip_text_features(class_names, device='cuda'):
    """
    Extracts normalized text features for a list of class names using CLIP.
    
    Args:
        class_names (list): List of class names.
        device (str): Device to run CLIP on.
        
    Returns:
        torch.Tensor: Normalized text features of shape (len(class_names), clip_dim).
        int: clip_dim (usually 512).
    """
    # Load CLIP model
    try:
        model, preprocess = clip.load("ViT-B/16", device=device)
    except Exception as e:
        logging.error(f"Failed to load CLIP: {e}")
        return None, 0
        
    model.eval()
    
    # Tokenize class names
    prompts = [f"a photo of a {c}" for c in class_names]
    # Handle batching if too many classes? usually ok.
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        
    # Normalize text features
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, text_features.shape[1]

def align_dimensions(features, target_dim, device='cuda'):
    """
    Aligns feature dimensions to target_dim using Padding or PCA.
    
    Args:
        features (torch.Tensor or np.ndarray): Features of shape (N, current_dim).
        target_dim (int): Target dimension.
        device (str): Device for output tensor.
        
    Returns:
        torch.Tensor: Aligned features of shape (N, target_dim).
    """
    if isinstance(features, np.ndarray):
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    else:
        features_tensor = features.to(device)
        
    current_dim = features_tensor.shape[1]
    
    if current_dim == target_dim:
        return features_tensor
        
    logging.info(f"Aligning dimensions from {current_dim} to {target_dim}...")
    
    if target_dim > current_dim:
        # Padding with zeros
        padding = torch.zeros(features_tensor.shape[0], target_dim - current_dim, device=device)
        features_aligned = torch.cat([features_tensor, padding], dim=1)
    else:
        # PCA for dimensionality reduction
        # Note: PCA is data dependent. In incremental settings, this might be unstable if not saved.
        # But for initialization it's fine.
        features_np = features_tensor.cpu().numpy()
        pca = PCA(n_components=target_dim)
        features_aligned_np = pca.fit_transform(features_np)
        features_aligned = torch.tensor(features_aligned_np, dtype=torch.float32).to(device)
        
    return features_aligned

def get_semantic_initialization(class_names, pool_size, target_dim, device='cuda'):
    """
    Implements Semantic-Anchored Initialization (SAI).
    """
    logging.info("Generating Semantic-Anchored Initialization (SAI)...")
    
    if not class_names:
        logging.warning("No class names provided for SAI. Falling back to random initialization.")
        return None

    # 1. Text Embedding Extraction
    text_features, clip_dim = get_clip_text_features(class_names, device)
    if text_features is None:
        return None
    
    text_features_np = text_features.cpu().numpy()
    
    # 2. Semantic Clustering
    logging.info(f"Clustering {len(class_names)} classes into {pool_size} centroids.")
    kmeans = KMeans(n_clusters=pool_size, random_state=42)
    kmeans.fit(text_features_np)
    centroids = kmeans.cluster_centers_ # Shape: (pool_size, clip_dim)
    
    # 3. Dimension Alignment
    centroids_aligned = align_dimensions(centroids, target_dim, device)
    
    return centroids_aligned.cpu()
