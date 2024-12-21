import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

class AttentionHead(nn.Module):
    def __init__(self, embed_size):
        super(AttentionHead, self).__init__()
        self.embed_size = embed_size
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.scale_factor = math.sqrt(embed_size)
    
    def forward(self, x):
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"
        
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        N, seq_length, embed_size = x.shape
        
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        
        query = query.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        
        out = out.transpose(1, 2).contiguous().view(N, seq_length, self.embed_size)
        out = self.fc_out(out)
        
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers):
        super(SelfAttentionBlock, self).__init__()
        self.layers = nn.ModuleList(
            [MultiHeadSelfAttention(embed_size, num_heads) for _ in range(num_layers)]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(num_layers)])
    
    def _param_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for multihead_layer, norm_layer in zip(self.layers, self.norm_layers):
            x = norm_layer(x + multihead_layer(x))
        return x

if __name__ == "__main__":
    embed_size = 128
    num_heads = 8
    num_layers = 6
    x = torch.randn(3, 10, embed_size)  # 示例输入 (batch_size, sequence_length, embed_dim)

    multi_layer_attention_model = SelfAttentionBlock(embed_size, num_heads, num_layers)
    output = multi_layer_attention_model(x)

    print(output.shape)  # Expected output shape should be (3, 10, 128)