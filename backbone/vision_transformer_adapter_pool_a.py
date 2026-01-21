# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import  Linear, LayerNorm
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model

import logging
import os
from collections import OrderedDict
import torch

class PoolAssigner(nn.Module):
    def __init__(self, dim_ins=16, dim_task=16, dim_cls=16) -> None:
        super().__init__()
        self.cls_emb = torch.nn.Embedding(501, dim_cls)
        self.ins_linear = torch.nn.Linear(768, dim_ins)
        dim_cat = dim_ins + dim_cls
        self.assign = torch.nn.Linear(dim_cat, 1)
        self.sg = torch.nn.Sigmoid()

    def forward(self, instance, weight):
        ins = self.ins_linear(instance)
        cls = self.cls_emb(weight)
        agg = torch.concat([ins, cls], dim=-1)
        res = self.assign(agg)
        res = self.sg(res)
        return res

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

    def apply_weight(self, down_m, down_b, up_m, up_b):
        self.down_proj_w= down_m
        self.up_proj_w = up_m
        self.down_proj_b = down_b
        self.up_proj_b = up_b


class AdapterPool(nn.Module):
    def __init__(self, n_emb, n_neck, pool_size=5, top_k = 1, embed_size=None, n_blocks = 12,
                config=None,
                prompt_init='uniform',
                prompt_key_init='uniform',
                prompt_key_init_tensor=None,
                ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.embed_size = embed_size
        self.config = config
        self.n_emb = n_emb
        self.n_neck = n_neck
        self.n_block = n_blocks
        self.embedding_key = "cls"
        self.top_k = top_k
        self.batchwise_prompt = config.batchwise_prompt
        self.prompt_init = 'uniform'
        self.prompt_key_init = prompt_key_init
    
        self.n_adapter = n_blocks * pool_size
        self.pool = nn.ModuleList([Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    )
                                for _ in range(self.n_adapter)
        ])

        key_shape = (pool_size, self.n_emb)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
        elif prompt_key_init == 'semantic':
            if prompt_key_init_tensor is None:
                raise ValueError("prompt_key_init is 'semantic' but prompt_key_init_tensor is None")
            if prompt_key_init_tensor.shape != key_shape:
                 raise ValueError(f"Shape mismatch for semantic init: {prompt_key_init_tensor.shape} vs {key_shape}")
            self.prompt_key = nn.Parameter(prompt_key_init_tensor)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm     

    def add_adapter(self, new_key_tensor, source_idx=None):
        """
        Dynamically adds a new adapter to the pool.
        
        Args:
            new_key_tensor (torch.Tensor): The new key to add. Shape (1, n_emb).
            source_idx (int, optional): The index of the adapter to copy weights from (Knowledge Inheritance).
        """
        device = self.prompt_key.device
        new_key_tensor = new_key_tensor.to(device)
        
        # 1. Update Key
        new_prompt_key = torch.cat([self.prompt_key.data, new_key_tensor], dim=0)
        self.prompt_key = nn.Parameter(new_prompt_key) # This resets grad requirement, handled by optimizer re-init
        
        # 2. Add Adapters (one per block)
        # self.pool is a ModuleList of length n_blocks * pool_size
        # The new adapters should be appended.
        # The indexing logic in forward_features (blk.adapt_list.append(POOL.pool[idx_group * POOL.n_block + idx - self.n_global]))
        # implies that adapters are grouped by adapter index, then block index.
        # So adapter k, block b is at index k * n_block + b.
        # Appending new adapters means they will be at indices (pool_size) * n_block + b.
        
        import logging
        logging.info(f"Expanding AdapterPool from {self.pool_size} to {self.pool_size + 1}...")
        
        for i in range(self.n_block):
            # Create new adapter
            new_adapter = Adapter(self.config, dropout=0.1, bottleneck=self.config.ffn_num,
                                    init_option=self.config.ffn_adapter_init_option,
                                    adapter_scalar=self.config.ffn_adapter_scalar,
                                    adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
                                    ).to(device)
            
            # Warm Start / Knowledge Inheritance
            if source_idx is not None:
                # Calculate index of source adapter for this block
                src_adapter_idx = source_idx * self.n_block + i
                src_adapter = self.pool[src_adapter_idx]
                new_adapter.load_state_dict(src_adapter.state_dict())
                logging.debug(f"Block {i}: Inherited weights from adapter {source_idx}")
            
            self.pool.append(new_adapter)
            
        self.pool_size += 1
        self.n_adapter += self.n_block
        logging.info(f"AdapterPool expansion complete. New pool size: {self.pool_size}")

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if 1:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # calculate similarity
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            # match, get idx
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            
            out['prompt_idx'] = idx  

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
            # logging.info(f"select {idx.squeeze(1)} , loss {reduce_sim}")
        else:
            if self.prompt_init == 'zero':
                self.down = nn.Parameter(torch.zeros(self.n_block, self.n_emb+1, self.n_neck))
                self.up = nn.Parameter(torch.zeros(self.n_block, self.n_neck+1, self.n_emb))
            elif self.prompt_init == 'uniform':
                self.down = nn.Parameter(torch.zeros(self.n_block, self.n_emb+1, self.n_neck))
                self.up = nn.Parameter(torch.zeros(self.n_block, self.n_neck+1, self.n_emb))
                nn.init.uniform_(self.down)
                nn.init.uniform_(self.up)
            batched_down_raw = self.down.unsqueeze(0).expand(x_embed.shape[0], -1, -1, -1)
            batched_up_raw = self.up.unsqueeze(0).expand(x_embed.shape[0], -1, -1, -1)
        
        return out
    


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None,
                 adapt_size=16,
                 shared=True):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        self.shared = shared
        if self.config.ffn_adapt:
            if self.shared:
                self.adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                        )
            else:
                self.adapt_list = []

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            if self.shared:
                adapt_x = self.adapter(x, add_residual=False)
            else:
                adapt_x = torch.cat([
                    self.adapt_list[i](x[i].unsqueeze(0), add_residual=False) 
                    for i in range(x.shape[0])
                ])

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt :
            if self.config.ffn_option == 'sequential' :
                if self.shared:
                    adapt_x = self.adapter(x, add_residual=False)
                else:
                    adapt_x = torch.cat([
                        self.adapt_list[i](x[i].unsqueeze(0), add_residual=False) 
                        for i in range(x.shape[0])
                    ])
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None,pool_size=10, bs=-1):
        super().__init__()


        print("I'm using ViT with adapter [POOL].")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.tuning_config = tuning_config

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.n_global = 0 

        blocks_ = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, adapt_size=bs, 
                shared= True
            )
            for i in range(self.n_global)
            ]
        for i in range(self.n_global, depth):
            blocks_.append(Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[depth-1], norm_layer=norm_layer, act_layer=act_layer,
                    config=tuning_config, layer_id=i, adapt_size=bs, 
                    shared= False
                ))
        assert len(blocks_) == 12, len(blocks_)
        self.blocks = nn.Sequential(*blocks_)
        self.norm = norm_layer(embed_dim)

        if self.tuning_config["ffn_adapt"]:
            prompt_key_init = getattr(tuning_config, 'prompt_key_init', 'uniform')
            prompt_key_init_tensor = getattr(tuning_config, 'prompt_key_init_tensor', None)
            
            self.pool = AdapterPool(pool_size=pool_size , n_blocks= 12-self.n_global, n_emb=embed_dim, n_neck=tuning_config.ffn_num, config=tuning_config,
                                    prompt_key_init=prompt_key_init, prompt_key_init_tensor=prompt_key_init_tensor)
            self.pool_few = AdapterPool(pool_size=pool_size , n_blocks= 12-self.n_global,n_emb=embed_dim, n_neck=tuning_config.ffn_num, config=tuning_config,
                                        prompt_key_init=prompt_key_init, prompt_key_init_tensor=prompt_key_init_tensor)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_few = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)
        
        ######## Assign Pool #########
        if self.tuning_config["ffn_adapt"]:
            self.assigner = PoolAssigner()

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, cls_features=None, adapter_id=-1, few=0):
        B = x.shape[0]
        x = self.patch_embed(x) 
        sim = 0.
        matched_idx = torch.zeros((B, 1))
        if self.tuning_config["ffn_adapt"]:
            if few==0:
                POOL = self.pool_few
            elif few==1:
                POOL = self.pool
        if self.tuning_config["ffn_adapt"]:
            if adapter_id != -1:
                matched_idx = torch.ones(B, dtype=int) * adapter_id
                matched_idx = matched_idx.unsqueeze(-1)
            else:
                res = POOL(x, cls_features=cls_features)
                matched_idx = res["prompt_idx"]  #shape [bs,1]
                sim = res['reduce_sim']
        
        if self.tuning_config["ffn_adapt"]:
            for idx, blk in enumerate(self.blocks): 
                if not blk.shared:
                    blk.adapt_list = []
                    for idx_group in matched_idx:
                        blk.adapt_list.append(POOL.pool[idx_group * POOL.n_block + idx - self.n_global])
                    assert len(blk.adapt_list) == B
    
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # return cls token [bs, dim]

        return outcome, sim, matched_idx

    def forward_pool(self, cls_features, weight):
        weight = weight.long()
        pool_id = self.assigner(cls_features, weight)
        return pool_id
    
    def forward(self, x, task_id=-1, cls_features=None, train=False, adapter_id=-1, weight=None):
        if self.tuning_config["ffn_adapt"]:
            x_few, sim_few, matched_few = self.forward_features(x, cls_features=cls_features, adapter_id=adapter_id, few=1)

        x, sim, matched = self.forward_features(x, cls_features=cls_features, adapter_id=adapter_id)
        
        out = {"pre_logits": x}  # cls token 
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)

        out["logits"] = x
        out['reduce_sim'] = sim
        out['prompt_idx'] = matched
        
        if self.tuning_config["ffn_adapt"]:
            pool_id=-1
            if weight != None:
                pool_id = self.forward_pool(cls_features, weight)
            out["pool_id"] = pool_id
            x_few = self.head_few(x_few)
            out["logits_few"] = x_few
            out['reduce_sim_few'] = sim_few
            out['prompt_idx_few'] = matched_few
        return out


def vit_base_patch16_224_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
        elif 'mlp.f1' in key:
            f1 = state_dict.pop(key)
            state_dict[key.replace('mlp.f1', 'f1')] = f1
        elif 'mlp.f2' in key:
            f1 = state_dict.pop(key)
            state_dict[key.replace('mlp.f2', 'f2')] = f1
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    # import clip
    # import logging
    # logging.info("Loading CLIP-ViT-B/16 pretrained weights...")
    # clip_model, _ = clip.load("ViT-B/16", device='cpu')
    # clip_state_dict = clip_model.visual.state_dict()
    # state_dict = {}

    # # Map CLIP weights to VisionTransformer
    # state_dict['patch_embed.proj.weight'] = clip_state_dict['conv1.weight']
    # state_dict['cls_token'] = clip_state_dict['class_embedding'].view(1, 1, -1)
    # state_dict['pos_embed'] = clip_state_dict['positional_embedding'].unsqueeze(0)
    
    # if 'ln_post.weight' in clip_state_dict:
    #     state_dict['norm.weight'] = clip_state_dict['ln_post.weight']
    #     state_dict['norm.bias'] = clip_state_dict['ln_post.bias']

    # for key, val in clip_state_dict.items():
    #     if 'transformer.resblocks' in key:
    #         # key example: transformer.resblocks.0.attn.in_proj_weight
    #         parts = key.split('.')
    #         block_idx = parts[2]
    #         module_name = parts[3]
    #         target_prefix = f"blocks.{block_idx}"
            
    #         if module_name == 'attn':
    #             sub_module = parts[4]
    #             suffix = parts[-1] # weight or bias
    #             if 'in_proj' in sub_module:
    #                 # split q, k, v
    #                 q, k, v = val.chunk(3, dim=0)
    #                 param_type = suffix.split('_')[-1] # weight or bias
    #                 state_dict[f"{target_prefix}.attn.q_proj.{param_type}"] = q
    #                 state_dict[f"{target_prefix}.attn.k_proj.{param_type}"] = k
    #                 state_dict[f"{target_prefix}.attn.v_proj.{param_type}"] = v
    #             elif 'out_proj' in sub_module:
    #                 state_dict[f"{target_prefix}.attn.proj.{suffix}"] = val
    #         elif module_name == 'ln_1':
    #             suffix = parts[-1]
    #             state_dict[f"{target_prefix}.norm1.{suffix}"] = val
    #         elif module_name == 'ln_2':
    #             suffix = parts[-1]
    #             state_dict[f"{target_prefix}.norm2.{suffix}"] = val
    #         elif module_name == 'mlp':
    #             sub_module = parts[4]
    #             suffix = parts[-1]
    #             if 'c_fc' in sub_module:
    #                 state_dict[f"{target_prefix}.fc1.{suffix}"] = val
    #             elif 'c_proj' in sub_module:
    #                 state_dict[f"{target_prefix}.fc2.{suffix}"] = val

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    print("OK!")


    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model




def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
        elif 'mlp.f1' in key:
            f1 = state_dict.pop(key)
            state_dict[key.replace('mlp.f1', 'f1')] = f1
        elif 'mlp.f2' in key:
            f1 = state_dict.pop(key)
            state_dict[key.replace('mlp.f2', 'f2')] = f1
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # s=model.state_dict()
    # # print the keys in s
    # for key in s.keys():
    #     print(key)
    # # print the keys in checkpoint_model
    # for key in state_dict.keys():
    #     if key in s.keys():
    #         print(key, 'yes')
    #     else:
    #         print(key, 'NOOOOOOOOOOOOOOOOOOO')

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model
