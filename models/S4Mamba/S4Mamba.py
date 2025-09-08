import math
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch import cosine_similarity


class spa_spe_similarity(nn.Module):
    def __init__(self, in_channels):
        super(spa_spe_similarity, self).__init__()
        self.to_ab = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.eps = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    
    def _cal_spa_entropy(self, x_spa):
        probs = self.softmax(x_spa)
        shannon_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * shannon_probs, dim=-1)
        return entropy


    def forward(self, x):
        batch_size, c, h, w = x.size()
        a, b = self.to_ab(x).chunk(2, dim=1)

        a_spa = a.view(batch_size, c, -1).permute(0, 2, 1)
        b_spa = b.view(batch_size, c, -1).permute(0, 2, 1)

        entropy_spa = self._cal_spa_entropy(a_spa)
        max_entropy_spa = torch.argmax(entropy_spa, dim=-1)
        selective_vec = torch.gather(a_spa, 1, max_entropy_spa.unsqueeze(1).unsqueeze(2).expand(-1, -1, c))
        sim_spa = F.cosine_similarity(selective_vec, b_spa, dim=2)
        atten_spa = self.softmax(torch.pow(sim_spa, 2)) # (batch_size, h*w)
        atten_spa = atten_spa.unsqueeze(2) # (batch_size, h*w, 1)

        spa_x = torch.mul(atten_spa, b_spa)
        spa_x = spa_x.permute(0, 2, 1).contiguous().view(batch_size, c, h, w) + x

        ###################################################################

        a_spe = a.view(batch_size, c, h * w)

        sim_spe = torch.bmm(a_spe, a_spe.permute(0, 2, 1))
        norm_spe = torch.norm(a_spe, p=2, dim=2, keepdim=True)
        norm_mat = torch.bmm(norm_spe, norm_spe.permute(0, 2, 1))

        atten_spe = self.softmax(sim_spe / (norm_mat + self.eps))
        spe_weight = torch.bmm(atten_spe, a_spe)
        spe_x = spe_weight.view(batch_size, c, h, w)

        return spa_x, spe_x


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(d_model=self.group_channel_num, d_state=16, d_conv=4, expand=2, )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), 
            nn.ReLU(), 
            nn.Linear(channels * 2, channels)
        )
        
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        B, C, H, W = x.shape # x shape: (B, C, H, W)
        
        x_re = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_re.view(B, H * W, C)
        
        x_mlp_out = self.mlp(x_flat) # (B, H*W, C)        
        x_mamba_out = self.mamba(x_mlp_out) # (B, H*W, C)

        
        x_recon = x_mamba_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # restore original shape (B, C, H, W)
        
        if self.use_proj:
            x_recon = self.proj(x_recon)

        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual, num_layers=4, group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        self.num_layers = num_layers
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels,use_residual=use_residual,group_num=group_num)
        self.spe_mamba = SpeMamba(channels,token_num=token_num,use_residual=use_residual,group_num=group_num)

        self.Conv_spa = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels), 
                                      nn.LeakyReLU(),)
        self.Conv_spe = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels), 
                                      nn.LeakyReLU(),)

    def forward(self,x, x_spa, x_spe):
        
        for i in range(self.num_layers):
            spa_x = self.spa_mamba(x_spa)
            spe_x = self.spe_mamba(x_spe)

        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = self.Conv_spa(spa_x) * weights[0] + self.Conv_spe(spe_x) * weights[1]
        else:
            fusion_x = spa_x + spe_x

        # if self.use_residual: 
        #     return fusion_x
        # else: 
        #     return fusion_x
        return fusion_x


class S4Mamba(nn.Module):
    def __init__(self,in_channels=128, num_classes=10, hidden_dim=64, num_layers=6, use_residual=True, token_num=8, group_num=4, use_att=True):
        super(S4Mamba, self).__init__()

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                            nn.GroupNorm(group_num,hidden_dim),
                                            nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=1,padding=1),
                                            nn.SiLU())
      
        self.spa_spe_similarity = spa_spe_similarity(hidden_dim)

        self.mamba = BothMamba(hidden_dim,token_num,use_residual, num_layers, group_num,use_att)


        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,hidden_dim),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=hidden_dim,out_channels=num_classes,kernel_size=1,stride=1,padding=0),
                                      nn.AdaptiveAvgPool2d(1),
                                      )

    def forward(self,x):
        x = x.permute(0, 3, 1, 2)
        x = self.patch_embedding(x)
        x_spa, x_spe = self.spa_spe_similarity(x)
        x = self.mamba(x, x_spa, x_spe)
        logits = self.cls_head(x)
        logits = logits.view(logits.size(0), -1)
        return logits



if __name__=='__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 9, 9, 103).to(device)
    model = S4Mamba(in_channels=103, num_classes=10).to(device)
    out = model(x).to(device)
    print(out.shape)