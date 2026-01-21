
import torch
import logging
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
        
        # 2. Compute Distances and Check for Expansion
        expanded_count = 0
        reused_count = 0
        
        # We iterate one by one to handle dynamic updates to the pool
        # (If we add a key, subsequent new classes might be close to the *new* key)
        
        for i, class_name in enumerate(new_class_names):
            t_new = text_features_aligned[i].unsqueeze(0) # (1, D)
            
            # Normalize t_new (it should be already normalized by get_clip_text_features, 
            # but align_dimensions (padding) preserves norm only if padding is zero and original was norm.
            # If padding is used, norm might change? No, adding zeros preserves L2 norm.
            # But let's ensure unit norm for cosine similarity.
            t_new = F.normalize(t_new, p=2, dim=1)
            
            # Get current keys
            current_keys = adapter_pool.prompt_key.data # (M, D)
            current_keys_norm = F.normalize(current_keys, p=2, dim=1)
            
            # Compute Cosine Similarity
            # sim = (1, D) @ (D, M) -> (1, M)
            similarities = torch.mm(t_new, current_keys_norm.t())
            
            # Distance = 1 - Cosine Similarity
            distances = 1 - similarities
            
            min_dist, min_idx = torch.min(distances, dim=1)
            min_dist_val = min_dist.item()
            nearest_idx = min_idx.item()
            
            logging.info(f"Class '{class_name}': Min Dist = {min_dist_val:.4f} (Nearest Adapter: {nearest_idx})")
            
            if min_dist_val > self.threshold:
                # Semantic Outlier -> Trigger Expansion
                logging.info(f"-> Outlier detected (Dist {min_dist_val:.4f} > {self.threshold}). Expanding pool.")
                
                # Adapter Fission
                # Key Initialization: k_M+1 <- t_new
                # Warm Start: Copy from nearest_idx
                adapter_pool.add_adapter(new_key_tensor=t_new, source_idx=nearest_idx)
                expanded_count += 1
            else:
                logging.info(f"-> In-distribution (Dist {min_dist_val:.4f} <= {self.threshold}). Reusing adapter {nearest_idx}.")
                reused_count += 1
                
        logging.info(f"Semantic Gatekeeper Summary: Expanded {expanded_count}, Reused {reused_count}")
        return {'expanded': expanded_count, 'reused': reused_count}
