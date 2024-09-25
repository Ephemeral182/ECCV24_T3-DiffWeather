import torch
import torch.nn as nn

class Attention_cross(nn.Module):
    def __init__(self, dim,text_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(text_dim, 2*dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,guidance=None):
        #B, C, H, W = x.shape
        #x = x.reshape(B,C,H*W).permute(0,2,1)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(guidance)
        B_, N_, C_ = kv.shape
        kv = kv.reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads // 2).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #x = x.permute(0,2,1).reshape(B,C,H,W)
        return x #, attn

class Prompt(nn.Module):
    def __init__(self, length=5,length_g=20, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',use_g_prompt=False):
        super().__init__()
        
        self.cross_attn = Attention_cross(dim=embed_dim,text_dim=embed_dim)
        #self.attn = Attention(dim=embed_dim)
        self.length_g = length_g
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.use_g_prompt = use_g_prompt
        if self.use_g_prompt:
            self.g_prompt = nn.Parameter(torch.randn(1, length_g, embed_dim))
            if prompt_init == 'zero':
                nn.init.zeros_(self.g_prompt)
            elif prompt_init == 'uniform':
                nn.init.xavier_uniform_(self.g_prompt)
                # nn.init.uniform_(self.g_prompt, -1, 1)
            if prompt_key_init == 'zero':
                self.prompt_key_g = nn.Parameter(torch.zeros((1,embed_dim)))
            elif prompt_key_init == 'uniform':
                self.prompt_key_g = nn.Parameter(torch.randn((1,embed_dim)))
                nn.init.uniform_(self.prompt_key_g, -1, 1)
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            self.prompt_frequency_table = torch.zeros((pool_size))
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    def update_frequency_table(self, selected_idx):
    # Increment the frequency table based on selected indices
        for idx in selected_idx:
            self.prompt_frequency_table[idx] += 1

    def penalize_frequent_prompts(self, similarity_scores):
    # Penalize prompts based on frequency by subtracting the normalized frequency
    # from the similarity scores
        self.prompt_frequency_table = self.prompt_frequency_table.to(similarity_scores.device)
        penalties = self.prompt_frequency_table / self.prompt_frequency_table.max()
        adjusted_scores = similarity_scores - penalties.unsqueeze(0)  # Broadcasting the penalties
        return adjusted_scores

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None,depth_feature=torch.randn(1,256,256).cuda(),reverse=False):
        if self.use_g_prompt:
            # g_prompt
            g_prompt = self.g_prompt.expand(x_embed.size(0), -1, -1)
            self.prompt_key_g.expand(x_embed.size(0), -1)
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
                depth_feature_mean = torch.mean(depth_feature, dim=1)
                g_prompt_mean = torch.mean(g_prompt, dim=1) 
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
                depth_feature_mean = torch.max(depth_feature, dim=1)
                g_prompt_mean = torch.max(g_prompt, dim=1) 
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
                depth_feature_mean = torch.max(depth_feature, dim=1)[0] + 2 * torch.mean(depth_feature, dim=1)
                g_prompt_mean = torch.max(g_prompt, dim=1)[0] + 2 * torch.mean(g_prompt, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                    depth_feature_mean = torch.max(depth_feature, dim=1)[0] # B, C
                    g_prompt_mean = torch.max(g_prompt, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
                    depth_feature_mean = cls_features
                    g_prompt_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
            g_prompt_norm = self.l2_normalize(self.prompt_key_g, dim=1) # B,C
            depth_feature_norm = self.l2_normalize(depth_feature_mean, dim=1)

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                similarity = self.penalize_frequent_prompts(similarity)
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                # reverse
                _, idx_reverse = torch.topk(similarity, k=self.top_k, dim=1, largest=False)
                self.update_frequency_table(idx.view(-1))
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
                    # reverse
                    idx_reverse = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            if reverse == True:
                idx = idx_reverse
            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
            
            if self.use_g_prompt:
                batched_prompt = torch.cat([(self.cross_attn(g_prompt,depth_feature)), batched_prompt], dim=1)

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            self.l2_normalize(self.prompt_key, dim=1)
            g_prompt_norm = g_prompt_norm.unsqueeze(1)
            depth_feature_norm = depth_feature_norm.unsqueeze(1)
            sim_g_e_pool = torch.matmul(batched_key_norm, g_prompt_norm.transpose(-1,-2)) # B, topk , C
            sim_g_dep_pool = torch.matmul(g_prompt_norm, depth_feature_norm.transpose(-1,-2))
            reduce_sim = ((torch.sum((1 - sim_g_dep_pool).square())) - torch.sum((1 - sim_g_e_pool).square())) / x_embed.shape[0] # Scalar
            out['prompt_loss'] = reduce_sim
        else:
            if not self.use_g_prompt:
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                    nn.init.uniform_(self.prompt)
                batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
            else:
                batched_prompt = g_prompt
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding_e'] = batched_prompt[:,self.length_g:,:]
        out['prompted_embedding_g'] = batched_prompt[:,:self.length_g,:]
        return out
    
# from ptflops import get_model_complexity_info
# import numpy as np
# model_prompt = Prompt(length=32,length_g=128,embed_dim=256,prompt_pool=True,prompt_key=True,pool_size=10,top_k=5,batchwise_prompt=True,use_g_prompt=True).cuda()
# x = torch.randn([1,256,256]).cuda()
# flops_t, params_t = get_model_complexity_info(model_prompt,(256,256) , as_strings=True, print_per_layer_stat=True)
# import time
# print(f"net flops:{flops_t} parameters:{params_t}")
# # model = nn.DataParallel(model)

# steps=25
# # print(b)
# time_avgs=[]
# memory_avgs=[]
# with torch.no_grad():
#     for step in range(steps):
        
#         torch.cuda.synchronize()
#         start = time.time()
#         result = model_prompt(x)
#         torch.cuda.synchronize()
#         time_interval = time.time() - start
#         memory = torch.cuda.max_memory_allocated()
#         if step>5:
#             time_avgs.append(time_interval)
#         #print('run time:',time_interval)
#             memory_avgs.append(memory)
# print('avg time:',np.mean(time_avgs),'fps:',(1/np.mean(time_avgs)),'memory:',(1/np.mean(memory_avgs)))