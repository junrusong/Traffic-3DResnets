import torch



# transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)


encoder_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
src = torch.rand(11, 32, 256)
print(src.size())
out = encoder_layer(src)
print(out.size())