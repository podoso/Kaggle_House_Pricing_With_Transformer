import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from typing import List, Optional, Tuple
import copy
import math
import warnings

Tensor = torch.Tensor
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    用一个大的权重参数矩阵进行线性变换

    参数:
        q, k, v: 对自注意来说，三者都是src；对于seq2seq模型，k和v是一致的tensor。
                 但它们的最后一维(num_features或者叫做embed_dim)都必须保持一致。
        w: 用以线性变换的大矩阵，按照q,k,v的顺序压在一个tensor里面。
        b: 用以线性变换的偏置，按照q,k,v的顺序压在一个tensor里面。

    形状:
        输入:
        - q: shape:`(..., E)`，E是词嵌入的维度（下面出现的E均为此意）。
        - k: shape:`(..., E)`
        - v: shape:`(..., E)`
        - w: shape:`(E * 3, E)`
        - b: shape:`E * 3` 

        输出:
        - 输出列表 :`[q', k', v']`，q,k,v经过线性变换前后的形状都一致。
    """
    E = q.size(-1)
    # 若为自注意，则q = k = v = src，因此它们的引用变量都是src
    # 即k is v和q is k结果均为True
    # 若为seq2seq，k = v，因而k is v的结果是True
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # seq2seq模型
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

# q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r'''
    在query, key, value上计算点积注意力，若有注意力遮盖则使用，并且应用一个概率为dropout_p的dropout

    参数：
        - q: shape:`(B, Nt, E)` B代表batch size， Nt是目标语言序列长度，E是嵌入后的特征维度
        - key: shape:`(B, Ns, E)` Ns是源语言序列长度
        - value: shape:`(B, Ns, E)`与key形状一样
        - attn_mask: 要么是3D的tensor，形状为:`(B, Nt, Ns)`或者2D的tensor，形状如:`(Nt, Ns)`

        - Output: attention values: shape:`(B, Nt, E)`，与q的形状一致;attention weights: shape:`(B, Nt, Ns)`
    
    例子：
        >>> q = torch.randn((2,3,6))
        >>> k = torch.randn((2,4,6))
        >>> v = torch.randn((2,4,6))
        >>> out = scaled_dot_product_attention(q, k, v)
        >>> out[0].shape, out[1].shape
        >>> torch.Size([2, 3, 6]) torch.Size([2, 3, 4])
    '''
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2,-1))
    if attn_mask is not None:
        attn += attn_mask 
    # attn意味着目标序列的每个词对源语言序列做注意力
    attn = F.softmax(attn, dim=-1)
    if dropout_p:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn 

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_seperate_proj_weight = None,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r'''
    形状：
        输入：
        - query：`(L, N, E)`
        - key: `(S, N, E)`
        - value: `(S, N, E)`
        - key_padding_mask: `(N, S)`
        - attn_mask: `(L, S)` or `(N * num_heads, L, S)`
        输出：
        - attn_output:`(L, N, E)`
        - attn_output_weights:`(N, L, S)`
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"

        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)
    
    # reshape q,k,v将Batch放在第一维以适合点积注意力
    # 同时为多头机制，将不同的头拼在一起组成一层
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
    # 若attn_mask值是布尔值，则将mask转换为float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # 若training为True时才应用dropout
    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.to(device)
    attn_output_weights = attn_output_weights.to(device)
    out_proj_weight = out_proj_weight.to(device)
    out_proj_bias = out_proj_bias.to(device)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # 打印调试信息
        # print(f"bsz: {bsz}, num_heads: {num_heads}, tgt_len: {tgt_len}, src_len: {src_len}")
        # print(f"attn_output_weights shape before view: {attn_output_weights.shape}")
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
    
class MultiheadAttention(nn.Module):
    r'''
    参数：
        embed_dim: 词嵌入的维度
        num_heads: 平行头的数量
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)
    
    例子：
        >>> multihead_attn = MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    '''
    def __init__(self, embed_dim, num_heads, device = 'cuda',dropout=0., bias=True,
                 kdim=None, vdim=None, batch_first=False) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim))).to(device)
            # self.in_proj_weight = self.in_proj_weight.to(device)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim)).to(device)
            # self.in_proj_bias = self.in_proj_bias.to(device)
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)



    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
    
class My_EncoderLayer(nn.Module):
    r'''
    参数：
        d_model: 词嵌入的维度（必备）
        nhead: 多头注意力中平行头的数目（必备）
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率（Default = 0.1）
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0（Default = 1e-5）
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)（Default：False）

    例子：
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.randn((32, 10, 512))
        >>> out = encoder_layer(src)
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(My_EncoderLayer, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.self_attn = self.self_attn.to(device)
        self.linear1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model).to(device)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        self.activation = activation


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # src = positional_encoding(src, src.shape[-1])  #为什么每一行都加一个positional embedding 应该是初始时加的这个东西
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        # 调试代码
        # print(src.shape)
        # print(src.device)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        # 调试代码
        return src
    
class My_Encoder(nn.Module):
    r'''
    参数：
        encoder_layer（必备）
        num_layers： encoder_layer的层数（必备）
        norm: 归一化的选择（可选）
    
    例子：
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.randn((10, 32, 512))
        >>> out = transformer_encoder(src)
    '''

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_Encoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # 不需要位置编码
        # output = positional_encoding(src, src.shape[-1])
        if src is None:
            raise ValueError("Input 'src' cannot be None")
        output = copy.deepcopy(src)
        # print(output)
        # print("The output1 shape is:",output.shape)
        for i in range(self.num_layers):
            output = output
            # 调试代码
            # print("The current loop is: ",i+1)
            # print(output)
            # print(type(output))
            # print(output.shape)

            output = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
    
class My_DecoderLayer(nn.Module):
    r'''
    参数：
        d_model: 词嵌入的维度（必备）
        nhead: 多头注意力中平行头的数目（必备）
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率（Default = 0.1）
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0（Default = 1e-5）
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)（Default：False）
    
    例子：
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.randn((10, 32, 512))
        >>> tgt = torch.randn((20, 32, 512))
        >>> out = decoder_layer(tgt, memory)
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(My_DecoderLayer, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first).to(device)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first).to(device)

        self.linear1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model).to(device)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        self.dropout3 = nn.Dropout(dropout).to(device)

        self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r'''
        参数：
            tgt: 目标语言序列（必备）
            memory: 从最后一个encoder_layer跑出的句子（必备）
            tgt_mask: 目标语言序列的mask（可选）
            memory_mask（可选）
            tgt_key_padding_mask（可选）
            memory_key_padding_mask（可选）
        '''
        # print("The shape of tgt is: ",tgt.shape)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
class My_Decoder(nn.Module):
    r'''
    参数：
        decoder_layer（必备）
        num_layers: decoder_layer的层数（必备）
        norm: 归一化选择
    
    例子：
        >>> decoder_layer =TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    '''
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(My_Decoder, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layer = decoder_layer.to(device)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for _ in range(self.num_layers):
            output = self.layer(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output

class My_Transformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False) -> None:
        super(My_Transformer, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder_layer = My_EncoderLayer(d_model, nhead).to(device)
        self.encoder = My_Encoder(encoder_layer, num_encoder_layers).to(device)
        decoder_layer = My_DecoderLayer(d_model, nhead).to(device)
        self.decoder = My_Decoder(decoder_layer, num_decoder_layers).to(device)
        self.linear = nn.Linear(d_model, 1).to(device)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first


    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r'''
        参数：
            src: 源语言序列（送入Encoder）（必备）
            tgt: 目标语言序列（送入Decoder）（必备）
            src_mask: （可选)
            tgt_mask: （可选）
            memory_mask: （可选）
            src_key_padding_mask: （可选）
            tgt_key_padding_mask: （可选）
            memory_key_padding_mask: （可选）
        
        形状：
            - src: shape:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: shape:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: shape:`(S, S)`.
            - tgt_mask: shape:`(T, T)`.
            - memory_mask: shape:`(T, S)`.
            - src_key_padding_mask: shape:`(N, S)`.
            - tgt_key_padding_mask: shape:`(N, T)`.
            - memory_key_padding_mask: shape:`(N, S)`.

            [src/tgt/memory]_mask确保有些位置不被看到，如做decode的时候，只能看该位置及其以前的，而不能看后面的。
            若为ByteTensor，非0的位置会被忽略不做注意力；若为BoolTensor，True对应的位置会被忽略；
            若为数值，则会直接加到attn_weights

            [src/tgt/memory]_key_padding_mask 使得key里面的某些元素不参与attention计算，三种情况同上

            - output: shape:`(T, N, E)`, `(N, T, E)` if batch_first.

        注意：
            src和tgt的最后一维需要等于d_model，batch的那一维需要相等
            
        例子:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        '''
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        out = self.linear(output)
        # 调试代码
        # print("The shape of model output is:",out.shape)
        # out1 = self.encoder(src)
        # out = self.linear(out1)
        return out
    
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r'''产生关于序列的mask，被遮住的区域赋值`-inf`，未被遮住的区域赋值为`0`'''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r'''用正态分布初始化参数'''
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
