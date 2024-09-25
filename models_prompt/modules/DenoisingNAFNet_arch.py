# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from .module_util import LayerNorm
from models_prompt.modules.prompt import Prompt

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

    def forward(self, x,guidance):
        B, C, H, W = x.shape
        x = x.reshape(B,C,H*W).permute(0,2,1)
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
        x = x.permute(0,2,1).reshape(B,C,H,W)
        return x 

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x).contiguous()
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time



from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
class NAFNet(nn.Module):

    def __init__(self, img_channel=3,out_channel=3,text_dim=512, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1, trans_out=False,is_prompt_pool=False):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        time_pos_emb = Timesteps(fourier_dim,flip_sin_to_cos=True,downscale_freq_shift=0.) 
        time_dim = width * 4
        self.trans_out=trans_out
        self.is_prompt_pool = is_prompt_pool
        self.time_mlp = nn.Sequential(
            time_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        if self.trans_out:
            self.ending_trans = nn.Sequential(
                            nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True),
                            nn.Sigmoid()
                            )
        if self.is_prompt_pool:
            self.atten_list = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)],
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )
        if self.is_prompt_pool:
            self.prompt_pool = Prompt(length=64,length_g=256,embed_dim=chan,prompt_pool=True
                                      ,prompt_key=True,pool_size=20,top_k=5,batchwise_prompt=True,
                                      use_g_prompt=True)
            self.depth_transform = nn.Linear(384,chan)
            for i in range(2):
                self.atten_list.append(Attention_cross(chan,chan))
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** (len(self.encoders))

    def DarkChannelMap(self,img,patch_size=11):
        B,C,H,W = img.size()
        maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
        dc = maxpool(0-img[:, None, :, :, :]).view(B,1,H,W)
        return -dc

    def forward(self, inp,time=1.,depth_feature = torch.ones([1,1028,384]).cuda()):

        timesteps = time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=inp.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(inp.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(inp.shape[0], dtype=timesteps.dtype, device=timesteps.device)


        x = inp 
    
        t = self.time_mlp(timesteps)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)
        
        encs = [x]
        for idx, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        b,c,h,w = x.shape
        x, _ = self.middle_blks([x, t])
        if self.is_prompt_pool is True:
                # prompt conditions
                depth_feature = self.depth_transform(depth_feature)
                prompt = self.prompt_pool(x.reshape(b,c,h*w).permute(0,2,1),depth_feature=depth_feature)
                prompt_e = prompt['prompted_embedding_e']
                prompt_g = prompt['prompted_embedding_g']
                x_e = self.atten_list[0](x,prompt_e)
                x = self.atten_list[1](x_e,prompt_g)
                prompt_loss = prompt['prompt_loss']

        idx_ = idx + 1
        for idx, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])
        if self.trans_out:
            trans_map = self.ending_trans(x + encs[0])
            trans_map = trans_map[..., :H, :W]
            
        x = self.ending(x + encs[0])
        x = x[..., :H, :W]
        if self.trans_out:
            return x,trans_map
        return [x,prompt_loss]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# from ptflops import get_model_complexity_info
# import numpy as np
# model =NAFNet(
#             img_channel=6,
#             out_channel=3,
#             width=64,
#             enc_blk_nums=[1, 1, 1, 18],
#             middle_blk_num=1,
#             dec_blk_nums=[1, 1, 1, 1],
#             trans_out=False,
#             is_prompt_pool=True,
#         ).cuda()
# H,W=256,256
# flops_t, params_t = get_model_complexity_info(model, (6, H,W), as_strings=True, print_per_layer_stat=True)
# import time
# print(f"net flops:{flops_t} parameters:{params_t}")
# # model = nn.DataParallel(model)
# condition = torch.ones(1,1,512).cuda()
# x = torch.ones([1,6,H,W]).cuda()
# depth_feature = torch.ones([1,1028,384]).cuda()
# steps=25
# # print(b)
# time_avgs=[]
# memory_avgs=[]
# with torch.no_grad():
#     for step in range(steps):
        
#         torch.cuda.synchronize()
#         start = time.time()
#         result = model(x)
#         torch.cuda.synchronize()
#         time_interval = time.time() - start
#         memory = torch.cuda.max_memory_allocated()
#         if step>5:
#             time_avgs.append(time_interval)
#         #print('run time:',time_interval)
#             memory_avgs.append(memory)
# print('avg time:',np.mean(time_avgs),'fps:',(1/np.mean(time_avgs)),'memory:',(1/np.mean(memory_avgs)),' size:',H,W)