"""***********************************************************************
FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition

-------------------------------------------------------------------------
File: mysql.py
- This file includes the main model configurations.

Version: 1.0
***********************************************************************"""


import torch
import torch.nn as nn
from torch.fft import rfft
import torch.nn.functional as F


class EventFilter(nn.Module):
    def __init__(self, num_filter, F, N, T, is_complex):
        """
        :param num_filter: out depth dimension
        :param F: in depth dimension
        :param N: num assets
        :param T: temporal dimension
        :param is_complex: use complex weights
        """
        super(EventFilter, self).__init__()
        self.num_filters = num_filter
        self.F, self.N, self.T = F, N, T

        if is_complex:
            self.weights = nn.Parameter(torch.randn(self.num_filters, self.F, 1, self.T, dtype=torch.cfloat))
        else:
            self.weights = nn.Parameter(torch.rand(self.num_filters, self.F, 1, self.T))

    def forward(self, x):
        x = x.unsqueeze(1) * self.weights.repeat(1, 1, self.N, 1)
        return x.view(-1, self.num_filters * self.F, self.N, self.T)


class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        """
        :param normalized_shape: The normalization target shape. i.e., (B, N, F) -> (F)
        """
        super(ComplexLayerNorm, self).__init__()
        self.axes = 2
        self.weight = nn.Parameter(torch.ones(1, 1, normalized_shape, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, 1, normalized_shape, dtype=torch.cfloat))

    def forward(self, x):
        x_mean = x.mean(dim=self.axes)
        x_var = x.var(dim=self.axes)

        expand_dim = (*x.shape[:2], 1)
        x_centered = x - x_mean.reshape(expand_dim)
        x_whitened = (1 / (torch.sqrt(x_var))).reshape(expand_dim) * x_centered

        x_normed = self.weight * x_whitened + self.bias
        return x_normed


class RealComplexTransformerEncoder(nn.Module):
    """
    Replacement Transformer for the ablation study during FQ-CTE or FQ-FRE-CTE.
    It embeds real and imag part respectively in real-domain instead of operating in complex-domain.
    """
    def __init__(self, in_features: int, embed_dim: int, num_heads: int):
        super(RealComplexTransformerEncoder, self).__init__()
        self.real_transformer = \
            nn.TransformerEncoderLayer(d_model=in_features, nhead=num_heads, dim_feedforward=embed_dim, dropout=0,
                                       batch_first=True)
        self.imag_transformer = \
            nn.TransformerEncoderLayer(d_model=in_features, nhead=num_heads, dim_feedforward=embed_dim, dropout=0,
                                       batch_first=True)

    def forward(self, x):
        real_part_out = self.real_transformer(x.real)
        imag_part_out = self.imag_transformer(x.imag)
        x = torch.complex(real_part_out, imag_part_out)
        return x, None


class ComplexTransformerEncoder(nn.Module):
    """
    The main module Complex Transformer Encoder (CTE) consists of Complex-MHA and
    Complex-LN (Same as original LN, except that it works in complex-domain).
    """
    def __init__(self, in_features: int, embed_dim: int, num_heads: int):
        super(ComplexTransformerEncoder, self).__init__()
        self.complex_multihead_attn = ComplexMultiheadAttention(in_features=in_features, embed_dim=embed_dim,
                                                                num_heads=num_heads)
        self.layer_norm_1 = ComplexLayerNorm(normalized_shape=in_features)
        self.c_ffn = nn.Linear(in_features=in_features, out_features=in_features, dtype=torch.cfloat)
        self.layer_norm_2 = ComplexLayerNorm(normalized_shape=in_features)

    def forward(self, x):
        # MHA
        _x, attention_score = self.complex_multihead_attn(x)
        # Add & Norm
        x = self.layer_norm_1(_x + x)

        # FFN
        _x = self.c_ffn(x)
        # Add & Norm
        x = self.layer_norm_2(_x + x)

        return x, attention_score


class ComplexMultiheadAttention(nn.Module):
    def __init__(self, in_features: int, embed_dim: int, num_heads: int):
        super(ComplexMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, f'Invalid num_heads: {num_heads} for the given embed_dim: {embed_dim}'
        self.embed_dim = embed_dim  # E
        self.num_heads = num_heads  # H
        self.chunk_size = int(self.embed_dim / self.num_heads)  # C
        self.divisor = torch.sqrt(torch.tensor([self.chunk_size * 2]))

        self.linear_q = nn.Linear(in_features=in_features, out_features=embed_dim, dtype=torch.cfloat)
        self.linear_k = nn.Linear(in_features=in_features, out_features=embed_dim, dtype=torch.cfloat)
        self.linear_v = nn.Linear(in_features=in_features, out_features=embed_dim, dtype=torch.cfloat)
        self.linear_cat = nn.Linear(in_features=embed_dim, out_features=in_features, dtype=torch.cfloat)

    def split_emb(self, x):
        split = x.split(self.chunk_size, dim=2)
        stack = torch.stack(split)
        return stack.transpose(0, 1)

    def forward(self, x):
        q = self.split_emb(self.linear_q(x))
        k = self.split_emb(self.linear_k(x))
        v = self.split_emb(self.linear_v(x))

        # Scaled Dot(Hermitian Inner)-Product Attention & Concatenation
        similarity_matrix = hermitian_inner_product_matmul(x=q, y=k.transpose(2, 3))
        similarity_matrix = similarity_matrix / self.divisor.to(similarity_matrix.device)
        attention_score = complex_abs_softmax(similarity_matrix)
        attention_value = hermitian_inner_product_matmul(x=attention_score, y=v)
        concat = attention_value.transpose(1, 2).reshape(-1, x.shape[1], self.embed_dim)

        # Linear out
        return self.linear_cat(concat), attention_score


class TemporalPatternExtractor(nn.Module):
    def __init__(self, N, T, pw_dim, num_ef, is_complex_filter, embed_dim, norm_=False, use_FRE=True, use_CTE=True):
        super(TemporalPatternExtractor, self).__init__()
        self.norm_ = norm_
        self.use_FRE = use_FRE
        self.use_CTE = use_CTE
        self.N, self.T = N, T
        self.spec_T = int(self.T/2 + 1) - 1

        # The Multi-Event Fusion Network consists of two consecutive blocks of event-filter and convolution layer
        self.event_filter_1 = EventFilter(num_filter=num_ef, F=pw_dim, N=self.N, T=self.spec_T,
                                          is_complex=is_complex_filter)
        self.context_dim = pw_dim * num_ef
        self.x_centered, self.ef_1_o, self.conv_1_o, self.ef_2_o = None, None, None, None
        self.conv_layer_1 = nn.Conv2d(in_channels=self.context_dim, out_channels=16, kernel_size=(1, 3),
                                      stride=(1, 2), bias=False, dtype=torch.cfloat)
        self.spec_T_conv_1 = \
            int((self.spec_T - self.conv_layer_1.kernel_size[1]) / self.conv_layer_1.stride[1] + 1)
        self.event_filter_2 = EventFilter(num_filter=1, F=self.conv_layer_1.out_channels,
                                          N=self.N, T=self.spec_T_conv_1, is_complex=is_complex_filter)
        self.conv_layer_2 = nn.Conv2d(in_channels=self.conv_layer_1.out_channels, out_channels=16,
                                      kernel_size=(1, 3), stride=(1, 2), bias=False, dtype=torch.cfloat)
        self.spec_T_conv_2 = \
            int((self.spec_T_conv_1 - self.conv_layer_2.kernel_size[1]) / self.conv_layer_2.stride[1] + 1)

        if self.use_FRE:
            if self.use_CTE:
                self.frequency_relation_encoder = \
                    ComplexTransformerEncoder(in_features=self.conv_layer_2.out_channels,
                                              embed_dim=embed_dim,
                                              num_heads=4)
            else:
                self.frequency_relation_encoder = \
                    RealComplexTransformerEncoder(in_features=self.conv_layer_2.out_channels,
                                                  embed_dim=embed_dim,
                                                  num_heads=4)
        self.spec_feature_dim = self.spec_T_conv_2 * self.conv_layer_2.out_channels

    def forward(self, x, mean_: torch.tensor = None, var_: torch.tensor = None):
        x = rfft(x, dim=3)
        x = x[:, :, :, :-1]

        if self.norm_:
            if mean_ is None and var_ is None:
                mean_ = x.mean(dim=2, keepdim=True)
                var_ = x.var(dim=2, keepdim=True)
                var_[var_ == 0] = 1e-09

            x = (1 / (torch.sqrt(var_))) * (x - mean_)
            self.x_centered = x

        ef_1_o = self.event_filter_1(x)
        conv_1_o = self.conv_layer_1(ef_1_o)
        ef_2_o = self.event_filter_2(conv_1_o)
        freq_emb = self.conv_layer_2(ef_2_o)
        self.ef_1_o, self.conv_1_o, self.ef_2_o = ef_1_o, conv_1_o, ef_2_o

        x = freq_emb.transpose(1, 2).transpose(2, 3).reshape(-1, self.spec_T_conv_2,
                                                             self.conv_layer_2.out_channels)
        if self.use_FRE:
            x, temp_attn_score = self.frequency_relation_encoder(x)
        else:
            x, temp_attn_score = x, None
        return x.reshape(-1, self.N, self.spec_feature_dim), freq_emb, temp_attn_score, mean_, var_


class FrequencyStateEncoder(nn.Module):
    def __init__(self, N, T, pw_dim, num_ef, embed_dim, is_complex_filter=False, use_FRE=True, use_CTE=True):
        super(FrequencyStateEncoder, self).__init__()
        self.N, self.T = N, T
        self.spec_T = int(self.T/2 + 1)

        self.tpe_X = TemporalPatternExtractor(N, T, pw_dim, num_ef, is_complex_filter, embed_dim=embed_dim,
                                              norm_=True, use_FRE=use_FRE, use_CTE=use_CTE)
        self.tpe_G = TemporalPatternExtractor(1, T, pw_dim, num_ef, is_complex_filter, embed_dim=embed_dim,
                                              norm_=True, use_FRE=use_FRE, use_CTE=use_CTE)
        self.cplx_linear = nn.Linear(in_features=self.tpe_X.spec_feature_dim, out_features=16, dtype=torch.cfloat)
        self.freq_emb, self.temp_attn_score = None, None  # Saves the Frequency Embedding and Temporal Attention Scores

    def forward(self, x, g):
        x, self.freq_emb, self.temp_attn_score, mean_, var_ = self.tpe_X(x, mean_=None, var_=None)
        g, _, _, _, _ = self.tpe_G(g, mean_=mean_, var_=var_)
        c = x + g.repeat(1, self.N, 1)
        c = self.cplx_linear(c)
        return c


class FQMuNet(nn.Module):
    def __init__(self, settings, net_settings, classifier_=True):
        super(FQMuNet, self).__init__()
        pw_dim, num_ef, dim1, dim3, var1 = net_settings
        self.classifier_ = classifier_
        self.state_space, self.action_space, self.use_short, self.use_glob, self.max_portfolio, \
            self.use_FRE, self.use_CTE = settings
        self.F, self.T = self.state_space[0], self.state_space[2]
        self.N = self.action_space[0]

        self.pw_conv_X = nn.Conv2d(in_channels=self.F, out_channels=pw_dim, kernel_size=(1, 1), bias=False)
        self.pw_conv_G = nn.Conv2d(in_channels=self.F, out_channels=pw_dim, kernel_size=(1, 1), bias=False)

        self.temp_mha_embed_dim = dim1
        self.num_fse_block = var1
        self.multi_fse_blocks = nn.ModuleList([FrequencyStateEncoder(N=self.N, T=int(self.T / 2 ** i),
                                                                     pw_dim=pw_dim,
                                                                     num_ef=num_ef,
                                                                     is_complex_filter=True,
                                                                     use_FRE=self.use_FRE,
                                                                     use_CTE=self.use_CTE,
                                                                     embed_dim=self.temp_mha_embed_dim) for i in
                                               range(self.num_fse_block)])

        self.context_dim = 0
        for i in range(self.num_fse_block):
            self.context_dim += self.multi_fse_blocks[i].cplx_linear.out_features

        self.asset_mha_embed_dim = dim3
        self.asset_attn_score = None
        self.c = None

        if self.use_CTE:
            self.asset_relation_encoder = ComplexTransformerEncoder(in_features=self.context_dim,
                                                                    num_heads=4,
                                                                    embed_dim=self.asset_mha_embed_dim)
        else:
            self.asset_relation_encoder = RealComplexTransformerEncoder(in_features=self.context_dim,
                                                                        num_heads=4,
                                                                        embed_dim=self.asset_mha_embed_dim)
        self.cash_weights = \
            nn.Parameter(torch.randn(1, 1, self.context_dim, dtype=torch.cfloat))
        self.linear_out = nn.Linear(in_features=self.context_dim+1, out_features=1, dtype=torch.cfloat)
        self.activation = nn.ReLU()

    def get_reg_term(self, prior_periodicity, prior_glob=False):
        def _func_(x, n):
            _common = torch.exp(2j * torch.pi * x / n)
            _numerator = -1 + _common
            _denominator = -2 + _common
            return (_numerator / _denominator).real

        def _glob_(x, n):
            return torch.tanh(x/n)

        regularization_term = None
        num_glob_filters = 5
        num_prior_filters = len(prior_periodicity) + num_glob_filters if prior_glob else len(prior_periodicity)

        for _enc in self.multi_fse_blocks:
            num_freq = _enc.spec_T - 1
            window_size = _enc.T
            periodicity_indices = [int(window_size/_p) for _p in prior_periodicity]
            reg_target_weights = [_enc.tpe_X.event_filter_1.weights.view(-1, num_freq),
                                  _enc.tpe_G.event_filter_1.weights.view(-1, num_freq)]

            for _reg_target_weight in reg_target_weights:
                for _idx, _filter in enumerate(_reg_target_weight[-num_prior_filters:, :]):
                    if _idx < len(prior_periodicity):
                        if periodicity_indices[_idx] >= len(_filter) or periodicity_indices[_idx] == 0:
                            continue
                        else:
                            _mask = _func_(x=torch.arange(len(_filter), device=self.cash_weights.device),
                                           n=periodicity_indices[_idx])
                    else:
                        _glob_idx = _idx - len(prior_periodicity)
                        n = len(_filter) / (2.0 + 1.5 * _glob_idx)
                        _mask = _glob_(x=torch.arange(len(_filter), device=self.cash_weights.device), n=n)

                    if regularization_term is None:
                        regularization_term = torch.linalg.vector_norm(torch.abs(_filter) * _mask)
                    else:
                        regularization_term = regularization_term + \
                                              torch.linalg.vector_norm(torch.abs(_filter) * _mask)

        return regularization_term

    def forward(self, x, w, g):
        x = x.contiguous()
        g = g.contiguous()

        x = self.activation(self.pw_conv_X(x))
        g = self.activation(self.pw_conv_G(g))

        c = torch.cat(
            [self.multi_fse_blocks[i](x[:, :, :, -int(self.T / 2 ** i):], g[:, :, :, -int(self.T / 2 ** i):])
             for i in range(self.num_fse_block)], dim=2)
        self.c = c

        attn, self.asset_attn_score = self.asset_relation_encoder(c)
        attn = torch.cat([self.cash_weights.repeat(attn.shape[0], 1, 1), attn], dim=1)
        attn = torch.cat([attn, w.view(-1, self.N + 1, 1)], dim=2)
        out = self.linear_out(attn).squeeze(2)
        out = out.real

        if not self.classifier_:
            return out
        else:
            if self.max_portfolio:
                out = asset_selection(out, self.max_portfolio)
            if self.use_short:
                return torch.div(out, torch.sum(torch.abs(out)))
            else:
                return F.softmax(out, dim=1)


class FQQNet(nn.Module):
    def __init__(self, settings, net_settings):
        super(FQQNet, self).__init__()
        self.state_space, self.action_space, self.use_short, self.use_glob, self.max_portfolio, _, _ = settings
        self.F, self.T = self.state_space[0], self.state_space[2]
        self.N = self.action_space[0]

        self.mu = FQMuNet(settings, net_settings, classifier_=False)
        self.linear_q_val = nn.Linear(in_features=2*(self.N+1), out_features=20)
        self.linear_q_val_out = nn.Linear(in_features=self.linear_q_val.out_features, out_features=1)
        self.activation = nn.ReLU()

    def forward(self, x, w, g, a):
        a = a.contiguous()
        out = self.mu(x, w, g)
        out = torch.cat([out, a], dim=1)
        out = self.activation(self.linear_q_val(out))
        q = self.linear_q_val_out(out)
        return q


def asset_selection(portfolio: torch.tensor, max_portfolio: int):
    """
    Selects the Top-|G| confidence scoring assets
    """
    assert max_portfolio > 0, 'Number of portfolio asset should be larger than 0'
    num_batch, num_asset = portfolio.size()

    indices = torch.sort(torch.abs(portfolio), dim=1)[1]
    # Selects the Top-|G| confident assets and deletes the rest
    del_indices = indices[:, :-max_portfolio]
    for _b in range(0, num_batch):
        portfolio[_b, del_indices[_b]] = 0.0

    return portfolio


def hermitian_inner_product_matmul(x, y):
    """
    Hermitian Inner Product based matrix multiplication for Multihead Attention.
    Input tensor x and y's shape must be (B, H, 2, 3)
    B = batch-size
    H = num-heads
    2 = sequence-length or chunk-size
    3 = chunk-size or sequence-length
    Note that, 2 and 3 dimensions are negligible.
    """
    assert x.shape[3] == y.shape[2], f'4th dimension of x should be equal to the 3rd dimension of y ' \
                                     f'{x.shape[3]} != {y.shape[2]}'
    out_shape = (x.shape[0], x.shape[1], x.shape[2], y.shape[3])
    x = x.reshape(-1, x.shape[2], x.shape[3])
    y = y.reshape(-1, y.shape[2], y.shape[3])
    return torch.bmm(x, torch.conj(y)).view(out_shape)


def complex_abs_softmax(x):
    """
    Takes complex tensor x and compute row-wise softmax. Change to complex before return.
    :returns a complex-valued row-wise softmax
    """
    return F.softmax(torch.abs(x), dim=3).type(dtype=torch.cfloat)
