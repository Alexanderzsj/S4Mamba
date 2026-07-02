import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


class spa_similarity(nn.Module):
    # 空间相似度计算，支持基于置信度图的锚点引导
    def __init__(self, in_channels):
        super(spa_similarity, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, conf_map=None):
        batch_size, c, h, w = x.size()
        a, b = self.conv(x).chunk(2, dim=1)
        a_spa = a.view(batch_size, c, -1).permute(0, 2, 1) 
        b_spa = b.view(batch_size, c, -1).permute(0, 2, 1) 

        if conf_map is not None:
            conf_flat = conf_map.view(batch_size, -1) 
            anchor_idx = torch.argmax(conf_flat, dim=-1) 
        else:
            anchor_idx = torch.full((batch_size,), (h // 2) * w + (w // 2), device=x.device)
            
        selective_vec = torch.gather(a_spa, 1, anchor_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, c)) 
        
        sim_spa = F.cosine_similarity(selective_vec, b_spa, dim=2) 
        atten_spa = self.softmax(torch.pow(sim_spa, 2)).unsqueeze(2) 
        
        spa_x = torch.mul(atten_spa, b_spa).permute(0, 2, 1).contiguous().view(batch_size, c, h, w)
        return spa_x


class spe_similarity(nn.Module):
    # 光谱特征聚合，提取光谱序列的统计量（均值与方差）并生成动态门控
    def __init__(self, in_channels):
        super(spe_similarity, self).__init__()
    
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, conf_map=None):
        batch_size, c, h, w = x.size()
        x_flat = x.view(batch_size, c, -1) 

        if conf_map is not None:
            weights = conf_map.view(batch_size, 1, -1) 
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = torch.ones(batch_size, 1, h * w, device=x.device) / (h * w)

        mu = torch.bmm(x_flat, weights.transpose(1, 2)) 

        diff_sq = (x_flat - mu) ** 2 
        var = torch.bmm(diff_sq, weights.transpose(1, 2)) 
        sigma = torch.sqrt(var + 1e-8) 

        stats = torch.cat([mu.squeeze(-1).unsqueeze(1), sigma.squeeze(-1).unsqueeze(1)], dim=1) 
        gate = self.fuse_conv(stats)

        gate = gate.transpose(1, 2) 
        spe_vector = mu * gate 
        
        spe_vector = self.proj(spe_vector).unsqueeze(-1)

        return spe_vector


class SpaMamba(nn.Module):
    # 空间 Mamba 分支，处理空间维度特征
    def __init__(self, channels, group_num=4):
        super(SpaMamba, self).__init__()
        self.spa_sim = spa_similarity(channels)
        self.dpe = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=3, expand=2)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels))

    def forward(self, x, conf_map=None):
        x_pos = self.spa_sim(x, conf_map) + self.dpe(x)
        B, C, H, W = x_pos.shape
        x_flat = x_pos.view(B, C, H * W).permute(0, 2, 1).contiguous() 
        x_flat = x_flat + self.mamba(self.norm1(x_flat))
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.permute(0, 2, 1).contiguous().view(B, C, H, W) + x


class SpeMamba(nn.Module):
    # 光谱 Mamba 分支，学习通道/光谱维度的长序列依赖
    def __init__(self, channels, d_inner=64):
        super(SpeMamba, self).__init__()
        
        self.spe_agg = SpeAggregation(channels) 
        self.up_proj = nn.Linear(1, d_inner)
        self.norm1 = nn.LayerNorm(d_inner)
        self.mamba_fwd = Mamba(d_model=d_inner, d_state=16, d_conv=3, expand=2)
        self.mamba_bwd = Mamba(d_model=d_inner, d_state=16, d_conv=3, expand=2)
        self.fusion_conv = nn.Linear(d_inner * 2, d_inner)
        self.norm2 = nn.LayerNorm(d_inner)
        self.mlp = nn.Sequential(nn.Linear(d_inner, d_inner * 2), nn.SiLU(), nn.Linear(d_inner * 2, d_inner))
        self.down_proj = nn.Linear(d_inner, 1)

    def forward(self, x, conf_map=None):
        x_sim = self.spe_agg(x, conf_map) 
        B, C, _, _ = x_sim.shape
        
        x_emb = self.up_proj(x_sim.view(B, C, 1))
        x_norm1 = self.norm1(x_emb)
        out_fwd = self.mamba_fwd(x_norm1)
        out_bwd = torch.flip(self.mamba_bwd(torch.flip(x_norm1, dims=[1])), dims=[1])
        x_emb = x_emb + self.fusion_conv(torch.cat([out_fwd, out_bwd], dim=-1)) 
        x_emb = x_emb + self.mlp(self.norm2(x_emb)) 
        
        refined_spe = x_sim + self.down_proj(x_emb).view(B, C, 1, 1) 
        return x + refined_spe


class FusionEncoder(nn.Module):
    # 基于 O(N) 线性注意力的空间与光谱多头特征融合
    def __init__(self, channels, heads=8):
        super().__init__()
        self.heads = heads
        
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)

    def forward(self, t1, t2):
        B, C, H, W = t1.shape

        t1_flat = rearrange(t1, 'b c h w -> b (h w) c')
        t2_flat = rearrange(t2, 'b c h w -> b (h w) c')

        q = rearrange(self.to_q(t2_flat), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(t1_flat), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(t1_flat), 'b n (h d) -> b h n d', h=self.heads)

        q = q.softmax(dim=-1)  
        k = k.softmax(dim=-2)  

        context = torch.matmul(k.transpose(-1, -2), v) 
        out_flat = torch.matmul(q, context)
        
        out_flat = rearrange(out_flat, 'b h n d -> b n (h d)')
        out = rearrange(out_flat, 'b (h w) c -> b c h w', h=H, w=W)

        return out


class S4Mamba(nn.Module):
    # 主网络架构：采用两阶段（盲看与置信度引导）的空间-光谱联合建模
    def __init__(self, in_channels=128, num_classes=10, hidden_dim=64, group_num=4):
        super(S4Mamba, self).__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1), 
            nn.GroupNorm(group_num, hidden_dim), 
            nn.SiLU()
        )
      
        self.spa_branch = SpaMamba(hidden_dim, group_num)
        self.spe_branch = SpeMamba(hidden_dim)
        
        self.fusion = FusionEncoder(channels=hidden_dim, num_classes=num_classes)
        
        self.feature_to_dense = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1), 
            nn.GroupNorm(group_num, hidden_dim), 
            nn.SiLU(),
            nn.Conv2d(hidden_dim, num_classes, 1) 
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.theta = nn.Parameter(torch.tensor([1.0, 1.0, 3.0]))

    def _get_logits(self, t1, t2, t_cross):
            dense1 = self.feature_to_dense(t1)
            dense2 = self.feature_to_dense(t2)
            dense_cross = self.feature_to_dense(t_cross)
            
            w1 = torch.sigmoid(self.theta[0])
            w2 = torch.sigmoid(self.theta[1])
            
            dense_fused = dense_cross + w1 * dense1 + w2 * dense2
            return dense_fused

    def _calc_confidence_map(self, dense_logits):
        # 利用香农熵计算特征级置信度，为第二阶段提供引导
        probs = F.softmax(dense_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True) 
        
        B = dense_logits.size(0)
        entropy_flat = entropy.view(B, -1)
        min_e = entropy_flat.amin(dim=1).view(B, 1, 1, 1)
        max_e = entropy_flat.amax(dim=1).view(B, 1, 1, 1)
        
        conf_map = 1.0 - (entropy - min_e) / (max_e - min_e + 1e-8)
        return conf_map

    def forward(self, x):
        x = self.patch_embedding(x.permute(0, 3, 1, 2))
        
        # 第一遍：盲看 (Initial Selection)
        spa_1 = self.spa_branch(x, conf_map=None)
        spe_1 = self.spe_branch(x, conf_map=None)
        t1, t2, t_cross = self.fusion(spa_1, spe_1)
        
        dense_logits_1 = self._get_logits(t1, t2, t_cross) 
        logits1 = self.pool(dense_logits_1).flatten(1) 
        
        with torch.no_grad(): 
            conf_map = self._calc_confidence_map(dense_logits_1) 
        
        # 第二遍：指引看 (Guided Selection，分支受 conf_map 优化)
        spa_2 = self.spa_branch(x, conf_map=conf_map)
        spe_2 = self.spe_branch(x, conf_map=conf_map)
        
        t1_2, t2_2, t_cross_2 = self.fusion(spa_2, spe_2)
        dense_logits_2 = self._get_logits(t1_2, t2_2, t_cross_2)
        logits2 = self.pool(dense_logits_2).flatten(1)
        
        if self.training:
            return logits1, logits2
        else:
            return logits2


if __name__=="__main__":
    model = S4Mamba(in_channels=128, num_classes=10, hidden_dim=64)
    model.train() 
    input_tensor = torch.randn(2, 128, 16, 16) 
    out1, out2 = model(input_tensor)
    print(f"第一轮分类输出形状: {out1.shape}, 最终引导输出形状: {out2.shape}")
