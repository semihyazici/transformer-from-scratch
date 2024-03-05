import torch
import torch.nn as nn
import math
from .config import TransformerConfig

class TransformerMultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        if not(d_model % n_heads == 0):
            raise Exception(f"d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.per_head_d = self.d_model // self.n_heads
        self.scale_by = math.sqrt(self.d_model)
        self.softmax = nn.Softmax(dim=-1)

        self.w_q = nn.Linear(in_features=self.d_model, out_features=self.d_model,bias=False)
        self.w_k = nn.Linear(in_features=self.d_model, out_features=self.d_model,bias=False)
        self.w_v = nn.Linear(in_features=self.d_model, out_features=self.d_model,bias=False)
        self.projection = nn.Linear(in_features=d_model, out_features=self.d_model)

    def scaled_dot_prod(self,q, k, v, mask ):

        attn_matrix = torch.matmul(q, k.transpose(-2,-1)) / self.scale_by

        if mask != None:
            attn_matrix = attn_matrix.masked_fill(mask == 0, -1e10)

        attn_matrix = self.softmax(attn_matrix)
        result = torch.matmul(attn_matrix, v )
        return result, attn_matrix

    def forward(self,q, k ,v, mask = None):
        batch_size, n_token_src, dim = q.size()
        batch_size, n_token_tgt, dim = v.size()
        
        q = self.w_q(q).view(batch_size, n_token_src, self.n_heads, self.per_head_d).transpose(1,2)
        k = self.w_k(k).view(batch_size, n_token_tgt, self.n_heads, self.per_head_d).transpose(1,2)
        v = self.w_v(v).view(batch_size, n_token_tgt, self.n_heads, self.per_head_d).transpose(1,2)

        attention_result ,attention_map = self.scaled_dot_prod(q=q, k=k, v=v ,mask=mask)
        attention_result = attention_result.transpose(1,2).contiguous().view(batch_size , n_token_src, self.d_model )
        result = self.projection(attention_result)
        return result, attention_map


class TransformerPositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class TransformerPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model,d_hidden, n_heads):
        super().__init__()
        self.multiheadattention = TransformerMultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.feedforward = TransformerPositionWiseFeedForward(d_model=d_model, d_hidden=d_hidden)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src , src_mask, ):
        attention_result, attention_map = self.multiheadattention(src, src, src, src_mask)
        src = self.layernorm_1( src + self.dropout(attention_result))
        out = self.feedforward(src)
        src = self.layernorm_2( src + self.dropout(out))    
        return src, attention_map

class TransformerEncoder(nn.Module):

    def __init__(self, d_model,d_hidden, n_heads, n_encoder,save_attention_maps=False):
        super().__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads) for _ in range(n_encoder)])
        self.save_attention_maps = save_attention_maps
    def forward(self, src, src_mask,):
        out = src
        if self.save_attention_maps:
            self.attetion_maps = []

        for enc in self.encoder_layers:
            out,encoder_attention_map = enc(out, src_mask)
            if self.save_attention_maps:
                self.attetion_maps.append(encoder_attention_map)
        
        return out,encoder_attention_map

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model , d_hidden, n_heads) :
        super().__init__()
        self.multiheadattention_1 = TransformerMultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.multiheadattention_2 = TransformerMultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.feedforward = TransformerPositionWiseFeedForward(d_model=d_model, d_hidden=d_hidden)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.layernorm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src, tgt, tgt_mask):
        """
            src -> encoder outs(memory) 
            tgt -> decoder inputs(targets)
        """
        attention_result, decoder_attention_map = self.multiheadattention_1(tgt, tgt, tgt, tgt_mask)
        tgt = self.layernorm_1( tgt + self.dropout(attention_result))
        cross_attention_result, cross_attention_map = self.multiheadattention_2(tgt, src , src, None)
        tgt = self.layernorm_2( tgt + self.dropout(cross_attention_result))
        out = self.feedforward(tgt)
        tgt = self.layernorm_3( tgt + self.dropout(out))
        return tgt, decoder_attention_map, cross_attention_map

class TransformerDecoder(nn.Module):

    def __init__(self, d_model,d_hidden, n_heads, n_decoder,save_attention_maps=False):
        super().__init__()
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads) for _ in range(n_decoder)])
        self.save_attention_maps = save_attention_maps
    def forward(self, src, tgt , tgt_mask):
        out = src
        if self.save_attention_maps:
            self.attetion_maps = []

        for dec in self.decoder_layers:
            tgt,decoder_attention_map, cross_attention_map = dec(out,tgt, tgt_mask )
            if self.save_attention_maps:
                self.attetion_maps.append([decoder_attention_map,cross_attention_map])
        
        return tgt,decoder_attention_map,cross_attention_map

class TransformerModel(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, n_blocks, vocab_size_src, vocab_size_tgt,max_seq_length =256,device ='cuda') :
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=vocab_size_src,embedding_dim=d_model)
        self.tgt_embedding = nn.Embedding(num_embeddings=vocab_size_tgt,embedding_dim=d_model)


        self.encoder = TransformerEncoder(d_model=d_model, d_hidden=d_hidden,
                                           n_heads=n_heads ,n_encoder=n_blocks)
        self.decoder = TransformerDecoder(d_model=d_model, d_hidden=d_hidden,
                                           n_heads=n_heads ,n_decoder=n_blocks)
        
        self.generator = nn.Linear(in_features=d_model, out_features= vocab_size_tgt)
        self.positional_encoding = TransformerPositionalEncoding(d_model=d_model,max_seq_length=max_seq_length )

        self.device = device



    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length,device=self.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def create_embeddings(self, src, tgt ):
        src_embed = self.positional_encoding(self.src_embedding(src))
        tgt_embed = self.positional_encoding(self.tgt_embedding(tgt))
        return src_embed, tgt_embed
    

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src,tgt)
        src_embed, tgt_embed = self.create_embeddings(src, tgt)
        
        memory,enc_attention_map = self.encoder(src_embed,src_mask)
        decoder_outs ,decoder_attention_map, cross_attention_map = self.decoder(memory,tgt_embed,tgt_mask)
        
        logits = self.generator(decoder_outs)
        return {'logits':logits ,
                'encoder_attention_map':enc_attention_map,
                'decoder_attention_map':decoder_attention_map,
                'cross_attention_map':cross_attention_map}


def get_model(config:TransformerConfig,vocab_size_src, vocab_size_tgt,device,path = None):
    d_model,d_hidden,n_heads, n_blocks,max_seq_len = TransformerConfig.d_model,TransformerConfig.d_hidden,TransformerConfig.n_heads,TransformerConfig.n_blocks,TransformerConfig.max_seq_len
    
    model = TransformerModel(d_model=d_model,d_hidden=d_hidden,
                              n_blocks=n_blocks, n_heads=n_heads,
                              max_seq_length=max_seq_len,vocab_size_src=vocab_size_src,
                                vocab_size_tgt=vocab_size_tgt ,device=device)
    if path is not None:
        model.load_state_dict(torch.load(path))

    return model.to(device)