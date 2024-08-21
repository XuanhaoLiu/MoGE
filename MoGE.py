import torch
from torch import nn, einsum
from einops import rearrange, repeat

######################################################### MoGE #########################################################
# Name : Sparse Mixture of Graph Experts
# Time : 2024.08.20 (Black Myth: Wukong published)
# Author : Xuanhao Liu
# Affiliation : Shanghai Jiao Tong University
# Conference : BIBM 2024

# in_channels     : the number of EEG channels
# hidden_channels : the number of hidden layers channels
# num_points      : the hidden tensor's dimmension
# time_window     : the window size of the slice of EEG signals
# num_layers      : the number of MoGE blocks, same as the L
# heads           : multi-head self-attention
# dim_head        : each head's dimmsion
# num_classes     : the number of categories
# num_experts     : the number of experts
# pool            : 'cls' is using the class token for classification,
#                   'mean' is using the average token for classification
# self.A          : the adjacency matrix learned from input features, all experts share the same adjacency matrix
#######################################################################################################################

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def norm_adjacency(self, A):
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)

    def forward(self, X, A):
        support = torch.einsum('mtik,kj->mtij', X, self.weight)
        norm_A = self.norm_adjacency(A)
        output = torch.einsum('ki,mtij->mtkj', norm_A, support)
        if self.use_bias:
            output += self.bias
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim)
            #nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Moe_GCN_Transformer_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, heads, dim_head, num_experts):
        super().__init__()
        self.num_points = num_points
        self.num_experts = num_experts
        
        self.gate = nn.Linear(num_points, num_experts)
        
        self.GCNs = nn.ModuleList([
            GCN(in_channels, out_channels)
            for _ in range(num_experts)])
        
        self.GCN = GCN(in_channels, out_channels)
        
        if in_channels != out_channels:
            self.res = lambda x: 0
        else:
            self.res = lambda x: x
        self.squ = nn.Sequential(
            nn.LayerNorm(out_channels * num_points),
            #nn.BatchNorm1d(out_channels * num_points),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.attention = nn.Sequential(
            Attention(in_channels * num_points, heads, dim_head, dropout=0.2),
            nn.LayerNorm(in_channels * num_points),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        
        self.pad = nn.ZeroPad2d((2, 0, 0, 0))
        self.causal_conv = nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1)

    def forward(self, x, A):
        y = self.attention(x) + x
        y = rearrange(y, 'N T (V C) -> N T V C', V=self.num_points)
        
        y = rearrange(y, 'N T V C -> N T C V')
        # print("y.shape = ", y.shape)
        prob = self.gate(y)
        # print("prob.shape = ", prob.shape)
        idx = torch.argmax(prob, dim=3)
        z = []
        for e in range(self.num_experts):
            mask = (idx==e).float()
            z.append(torch.mul(y, mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)))
        
        y = rearrange(y, 'N T V C -> N T C V')
        y = self.res(y)
        for e in range(self.num_experts):
            yg = self.GCNs[e](rearrange(z[e], 'N T V C -> N T C V'), A)
            yg = rearrange(yg, 'N T V C -> N T C V')
            mask = (idx==e).float()
            yg = torch.mul(yg, mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1))
            y += rearrange(yg, 'N T V C -> N T C V')
            z[e] = rearrange(yg, 'N T V C -> N T C V')
        
        y = rearrange(y, 'N T V C -> (N V) C T')
        
        y = self.pad(y)       
        y = y.unsqueeze(1)
        y = self.causal_conv(y)
        y = y.squeeze(1)

        y = rearrange(y, '(N V) C T -> N V C T', V=5)
        y = rearrange(y, 'N V C T -> N T (V C)')
        
        y = self.squ(y)
        
        return y

class Sparse_MoGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_points, time_window, num_layers, heads, dim_head, num_classes, num_experts=6, pool='cls'):
        super().__init__()
        self.num_points = num_points
        self.pool = pool
        dim = in_channels * num_points
        self.pos_embedding = nn.Parameter(torch.randn(1, time_window + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ln = nn.LayerNorm(in_channels * num_points)

        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            Moe_GCN_Transformer_unit(in_channels, hidden_channels, num_points, heads, dim_head, num_experts)
            for _ in range(num_layers) ])
        
        self.fc = nn.Linear(hidden_channels * num_points, num_classes)
        
        self.xs, self.ys = torch.tril_indices(self.num_points, self.num_points, offset=-1)
        adjacency = torch.Tensor(self.num_points, self.num_points)
        nn.init.uniform_(adjacency)
        self.A = nn.Parameter(adjacency[self.xs, self.ys], requires_grad=True)

    def forward(self, x):
        N, T, V, C = x.shape
        x = x.view(N, T, V * C)
        cls_tokens = repeat(self.cls_token, '() T D -> N T D', N=N)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(T + 1)]
        T += 1
        x = x.view(N * T, V * C)
        x = self.ln(x)
        x = x.view(N, T, V * C)

        adacency = torch.zeros((self.num_points, self.num_points)).to(x.device)
        adacency[self.xs, self.ys] = self.A
        adacency = adacency + adacency.T + torch.eye(self.num_points).to(x.device)
        for layer in self.layers:
            x = layer(x, adacency)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        out = self.fc(x)
        return out

if __name__ == "__main__":
    model = Sparse_MoGE(in_channels=62, hidden_channels=62, num_points=5, time_window=5, num_layers=4, heads=2, dim_head=4, num_classes=3, num_experts=3, pool='cls')
    x = torch.rand(size=(32, 5, 62, 5))
    print(x.shape)
    y = model(x)
    print(y.shape)