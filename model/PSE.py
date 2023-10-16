

import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import os
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.count = 0

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q,k.transpose(-2, -1)) * self.scale
        # if self.count == 0:
            # for param in self.qkv.parameters():
            #     print(param)
            # print('-----------------------------------------')
            # print('the number of 0 of q',torch.sum(torch.sum(torch.eq(q,torch.zeros_like(q)))))
            # print('the number of 0 of k',torch.sum(torch.sum(torch.eq(k,torch.zeros_like(k)))))
            # print('the number of 0 of attn',torch.sum(torch.sum(torch.eq(attn,torch.zeros_like(attn)))))
            # if(os.path.exists('/home/chenguangyan/data/q.pth') == False):
            #     torch.save(x,'/home/chenguangyan/data/attnx.pth')
            #     torch.save(q,'/home/chenguangyan/data/q.pth')
            #     torch.save(k,'/home/chenguangyan/data/k.pth')
            #     count = 0
            #     for param in self.qkv.parameters():
            #         torch.save(param,'/home/chenguangyan/data/param-%d.pth'%count)
            #         count += 1
            # print('q',q)
            # print('k.shape',q.shape)
            # print('k',k)
            # print('attn',attn)
            # self.count += 1
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn,v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 0',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 0',torch.all(torch.eq(judge_1,judge)))
        x = self.norm1(x)
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 1',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 1',torch.all(torch.eq(judge_1,judge)))
        x = self.attn(x)
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 2',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 2',torch.all(torch.eq(judge_1,judge)))
        x = x + self.drop_path(x)
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 3',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 3',torch.all(torch.eq(judge_1,judge)))
        x = self.norm2(x)
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 4',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 4',torch.all(torch.eq(judge_1,judge)))
        x = self.mlp(x)
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 5',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 5',torch.all(torch.eq(judge_1,judge)))
        x = x + self.drop_path(x)
        # judge = torch.isnan(x)
        # judge_1 = torch.zeros_like(x)
        # print('[-3] - 6',torch.all(torch.eq(judge_1,x))==False)
        # print('[-3] - 6',torch.all(torch.eq(judge_1,judge)))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = norm(in_dim,eps=0.0001)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        v = v.transpose(1,2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = v.squeeze(1) + x

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)
        self.dim = dim
        self.count = 0

    def forward(self, x):
        y = self.norm1(x)
        x = self.attn(y)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))
        return x


def knn(x, k):
    x = x.transpose(1,2)
    distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
    idx = distance.topk(k=k, dim=-1)[1]
    return idx

def get_local_feature(x, refer_idx):

    x = x.view(*x.size()[:3])

    batch_size, num_points, k = refer_idx.size()

    idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1) * num_points

    idx = refer_idx + idx_base

    idx = idx.view(-1)

    _, _, num_dims = x.size()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k*num_dims)

    return feature

class LFI(nn.Module):
    def __init__(self):
        super(LFI,self).__init__()
    def forward(self,x,refer_idx):
        x = get_local_feature(x,refer_idx)
        return x

class PSE_module(nn.Module):
    def __init__(self, num_points=5, in_chans=3, embed_dim=256, token_dim=64,norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_points = num_points
        self.LFI0 = LFI()
        self.LFI1 = LFI()
        self.LFI2 = LFI()
        self.LFI3 = LFI()
        self.LFI4 = LFI()
        self.LFI5 = LFI()
        self.LFI6 = LFI()
        self.LFI7 = LFI()

        self.attention1 = Token_transformer(dim=512 * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention2 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention3 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention4 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention5 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention6 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention7 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention8 = Token_transformer(dim=token_dim * 5, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        
        self.project = nn.Linear(token_dim * 4, embed_dim)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        refer_idx = knn(x,self.num_points)
        refer_idx = refer_idx.to('cuda')

        x = x.transpose(1,2).contiguous()
        x = self.LFI0(x,refer_idx)
        x1 = self.attention1(x)
        
        x = self.LFI1(x1,refer_idx)
        x2 = self.attention2(x)

        x = self.LFI2(x2,refer_idx)
        x3 = self.attention3(x)

        x = self.LFI3(x3,refer_idx)
        x4 = self.attention4(x)

        x = self.LFI4(x4,refer_idx)
        x5 = self.attention5(x)
        
        x = self.LFI5(x5,refer_idx)
        x6 = self.attention6(x)

        x = self.LFI6(x6,refer_idx)
        x7 = self.attention7(x)

        x = self.LFI7(x7,refer_idx)
        x8 = self.attention8(x)

        x = F.leaky_relu(torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=-1))

        x = self.norm(x)
        x = x.transpose(-1,-2).contiguous()
        return x
