
import torch
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F
from utils.semantic_init import get_clip_text_features, align_dimensions

class SemanticGatekeeper:
    def __init__(self, threshold=0.5):
        """
        Semantic Gatekeeper for dynamic adapter expansion.
        
        Args:
            threshold (float): Semantic Tolerance Threshold (tau).
                               If min_distance > threshold, trigger expansion.
                               Distance metric: 1 - cosine_similarity.
        """
        self.threshold = threshold

    def process_new_classes(self, adapter_pool, new_class_names, device):
        """
        Checks new classes against existing adapter keys and triggers expansion if needed.
        Intelligently clusters new outliers to determine the number of adapters to add.
        
        Args:
            adapter_pool (AdapterPool): The adapter pool module.
            new_class_names (list): List of new class names.
            device (str): Device to run computations on.
            
        Returns:
            dict: Summary of actions taken (e.g., {'expanded': 3, 'reused': 5}).
        """
        logging.info("Semantic Gatekeeper: processing new classes...")
        
        if not new_class_names:
            return {'expanded': 0, 'reused': 0}

        # 1. Get Text Embeddings for New Classes
        # Note: We need the dimension of the adapter keys to align properly.
        # Adapter keys shape: (M, D)
        target_dim = adapter_pool.prompt_key.shape[1]
        
        text_features, clip_dim = get_clip_text_features(new_class_names, device=device)
        if text_features is None:
            logging.warning("Semantic Gatekeeper: Failed to extract text features. Skipping expansion check.")
            return {'expanded': 0, 'reused': 0, 'error': 'clip_failed'}
            
        # Align dimensions if necessary
        text_features_aligned = align_dimensions(text_features, target_dim, device=device)
        text_features_aligned = F.normalize(text_features_aligned, p=2, dim=1)
        
        # 2. Identify Outliers vs In-Distribution
        outlier_indices = []
        in_dist_indices = []
        
        # Snapshot of current keys (fixed for this batch of new classes)
        current_keys = adapter_pool.prompt_key.data # (M, D)
        current_keys_norm = F.normalize(current_keys, p=2, dim=1)
        
        for i, class_name in enumerate(new_class_names):
            t_new = text_features_aligned[i].unsqueeze(0) # (1, D)
            
            # Compute Cosine Similarity
            similarities = torch.mm(t_new, current_keys_norm.t())
            distances = 1 - similarities
            min_dist, min_idx = torch.min(distances, dim=1)
            min_dist_val = min_dist.item()
            
            logging.info(f"Class '{class_name}': Min Dist = {min_dist_val:.4f} (Nearest Adapter: {min_idx.item()})")
            
            if min_dist_val > self.threshold:
                outlier_indices.append(i)
            else:
                in_dist_indices.append(i)
        
        reused_count = len(in_dist_indices)
        expanded_count = 0
        
        # 3. Process Outliers with Clustering
        if outlier_indices:
            logging.info(f"Found {len(outlier_indices)} outliers. Clustering to determine new adapters...")
            
            outlier_embeddings = text_features_aligned[outlier_indices] # (N_out, D)
            
            if len(outlier_indices) == 1:
                # Only 1 outlier, just add it
                # Find nearest existing for warm start
                t_new = outlier_embeddings
                sim = torch.mm(t_new, current_keys_norm.t())
                dist = 1 - sim
                min_idx = torch.argmin(dist).item()
                
                logging.info(f"-> Single outlier. Expanding pool with 1 adapter. (Source: {min_idx})")
                adapter_pool.add_adapter(new_key_tensor=t_new, source_idx=min_idx)
                expanded_count = 1
                
            else:
                # Multiple outliers, use Agglomerative Clustering
                # Compute distance matrix between outliers
                sim_matrix = torch.mm(outlier_embeddings, outlier_embeddings.t())
                dist_matrix = 1 - sim_matrix
                dist_matrix = dist_matrix.cpu().numpy()
                dist_matrix[dist_matrix < 0] = 0 # Numerical stability
                
                # Use the same threshold for clustering
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.threshold,
                    metric='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(dist_matrix)
                n_clusters = len(set(labels))
                
                logging.info(f"-> Clustered {len(outlier_indices)} outliers into {n_clusters} groups.")
                
                for cluster_id in range(n_clusters):
                    member_mask = (labels == cluster_id)
                    
                    # Get class names for this cluster
                    cluster_indices = [outlier_indices[i] for i, is_member in enumerate(member_mask) if is_member]
                    cluster_class_names = [new_class_names[i] for i in cluster_indices]
                    
                    # Compute centroid
                    cluster_emb = outlier_embeddings[torch.tensor(member_mask)]
                    centroid = cluster_emb.mean(dim=0, keepdim=True)
                    centroid = F.normalize(centroid, p=2, dim=1)
                    
                    # Find nearest existing adapter for warm start
                    sim_to_existing = torch.mm(centroid, current_keys_norm.t())
                    dist_to_existing = 1 - sim_to_existing
                    min_idx = torch.argmin(dist_to_existing).item()
                    
                    logging.info(f"   -> Group {cluster_id}: New Adapter for classes {cluster_class_names}. Source: {min_idx}")
                    adapter_pool.add_adapter(new_key_tensor=centroid, source_idx=min_idx)
                    expanded_count += 1

        logging.info(f"Semantic Gatekeeper Summary: Expanded {expanded_count} adapters for {len(outlier_indices)} outliers. Reused {reused_count}.")
        return {'expanded': expanded_count, 'reused': reused_count}
