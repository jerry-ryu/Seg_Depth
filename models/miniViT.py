import torch
import torch.nn as nn

from .layers import PatchTransformerEncoder, PixelWiseDotProduct


class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear', seg_class = 37):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.seg_class = seg_class +1
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)
        x_seg = self.conv1x1(x.clone()) # (B, 128, h, w)

        regression_head, queries, seg_queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...], tgt[self.n_query_channels + 1:self.n_query_channels + 1 + self.seg_class, ...]
        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2) # (128,2,128) -> (B,128,128) 
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w
        
        seg_queries = seg_queries.permute(1, 0, 2) # (38,2,128) -> (B,38,128)
        seg_range_attention_maps = self.dot_product_layer(x_seg, seg_queries)  # .shape = n, self.seg_class, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps, seg_range_attention_maps
