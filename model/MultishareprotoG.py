
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.FITS import Model as FITS
from model.freq_proto import (
    FrequencyRouter, FrequencyEmbedding, FreqCrossAttention, PrototypeMemory
)
from model.csim import CSIM


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


class AdaptiveFusionGate(nn.Module):

    def __init__(self, num_scales, enc_in, hidden_dim=128):
        super().__init__()
        self.num_scales = num_scales
        feat_dim = enc_in * 2 * num_scales
        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, scale_outputs):
        gate_features = []
        for out in scale_outputs:
            gate_features.append(out.mean(dim=1))
            gate_features.append(out.std(dim=1))
        gate_input = torch.cat(gate_features, dim=-1)
        logits = self.gate_net(gate_input)
        weights = torch.softmax(logits, dim=-1)
        return weights


class ProtoFusionGate(nn.Module):

    def __init__(self, num_scales, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_scales, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, proto_dists):
        logits = self.gate(proto_dists)
        weights = torch.softmax(logits, dim=-1)
        fused_dist = (proto_dists * weights).sum(dim=-1)
        return fused_dist, weights


class SharedFreProG(nn.Module):

    def __init__(self, seq_len, channels, d_model=64,
                 topk_freq=10, n_queries=8, query_len=None,
                 gumbel_temperature=1.0, n_heads=4, attn_dropout=0.1,
                 shared_memory=None):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.d_model = d_model

        self.router = FrequencyRouter(
            seq_len=seq_len, channels=channels, d_model=d_model,
            topk_freq=topk_freq, n_queries=n_queries,
            query_len=query_len, gumbel_temperature=gumbel_temperature,
        )

        self.embedding = FrequencyEmbedding(
            seq_len=seq_len, channels=channels, d_model=d_model,
        )

        self.cross_attn = FreqCrossAttention(
            d_model=d_model, n_heads=n_heads, dropout=attn_dropout,
        )

        self.csim = CSIM(
            channels=channels,
            hidden_dim=32,
            freq_agg='mean',
            temperature=1.0,
            diag_bias=1.0,
            use_layer_norm=False
        )

        self.memory = shared_memory

    def forward(self, x_down, gumbel_temp=None, hard=False, update_proto=True):
        X_f = torch.fft.rfft(x_down, dim=1)
        X_f_csim, W_csim = self.csim(X_f, return_details=False)
        Q, route_weights = self.router(X_f_csim, temperature=gumbel_temp, hard=hard)
        K, V = self.embedding(X_f_csim)
        R, attn_weights = self.cross_attn(Q, K, V)
        
        G_selected, max_sim, proto_dist, match_idx, balance_loss = \
            self.memory(R, update=update_proto)

        return proto_dist, max_sim, match_idx, route_weights, attn_weights, balance_loss, G_selected

    def get_buffer_size(self):
        return self.memory.get_buffer_size()


class MultishareprotoG(nn.Module):

    def __init__(self, win_size, enc_in, individual, cut_freq,
                 scales=[2, 4, 8],
                 d_model=64, topk_freq=10, n_queries=8,
                 n_heads=4, attn_dropout=0.1,
                 n_prototypes=16, ema_decay=0.95,
                 gumbel_temperature=1.0,
                 max_buffer_size=50000,
                 n_basis=8, basis_type='mixed'):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.win_size = win_size
        self.enc_in = enc_in

        self.global_memory = PrototypeMemory(
            d_model=d_model,
            n_prototypes=n_prototypes,
            decay=ema_decay,
            max_buffer_size=max_buffer_size,
        )

        self.fits_models = nn.ModuleList()
        self.freprog_models = nn.ModuleList()

        for dsr in scales:
            seq_len = win_size // dsr
            pred_len = win_size - seq_len
            scale_cut_freq = min(cut_freq, seq_len // 2)

            cfg = DotDict({
                'seq_len': seq_len,
                'pred_len': pred_len,
                'enc_in': enc_in,
                'individual': individual,
                'cut_freq': scale_cut_freq,
                'use_basis': True,
                'd_model': d_model,
                'n_basis': n_basis,
                'basis_type': basis_type,
            })
            fits = FITS(cfg)
            self.fits_models.append(fits)

            query_len = max(seq_len // 4, 4)
            scale_topk = min(topk_freq, seq_len // 2)
            self.freprog_models.append(SharedFreProG(
                seq_len=seq_len,
                channels=enc_in,
                d_model=d_model,
                topk_freq=scale_topk,
                n_queries=n_queries,
                query_len=query_len,
                gumbel_temperature=gumbel_temperature,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                shared_memory=self.global_memory,
            ))

        self.recon_fusion = AdaptiveFusionGate(
            num_scales=self.num_scales,
            enc_in=enc_in,
            hidden_dim=128,
        )
        self.proto_fusion = ProtoFusionGate(
            num_scales=self.num_scales,
            hidden_dim=64,
        )

    def forward(self, x_full, gumbel_temp=None, hard=False, update_proto=True):
        B, T, C = x_full.shape
        scale_outputs = []
        all_proto_dists = []
        all_max_sims = []
        all_match_idxs = []
        all_route_weights = []
        total_balance_loss = torch.tensor(0.0, device=x_full.device)
        total_basis_loss = torch.tensor(0.0, device=x_full.device)

        for i, dsr in enumerate(self.scales):
            x_down = x_full[:, ::dsr, :]

            (proto_dist, max_sim, match_idx, route_w,
             attn_w, balance_loss, G_selected) = \
                self.freprog_models[i](
                    x_down,
                    gumbel_temp=gumbel_temp,
                    hard=hard,
                    update_proto=update_proto,
                )

            out, _ = self.fits_models[i](x_down, G_selected=G_selected)
            scale_outputs.append(out)

            basis_loss = self.fits_models[i].get_basis_loss()

            all_proto_dists.append(proto_dist)
            all_max_sims.append(max_sim)
            all_match_idxs.append(match_idx)
            all_route_weights.append(route_w)
            total_balance_loss = total_balance_loss + balance_loss
            total_basis_loss = total_basis_loss + basis_loss

        recon_weights = self.recon_fusion(scale_outputs)
        stacked = torch.stack(scale_outputs, dim=-1)
        w = recon_weights.unsqueeze(1).unsqueeze(2)
        fused_output = (stacked * w).sum(dim=-1)

        scale_proto_dists = torch.stack(all_proto_dists, dim=-1)
        fused_proto_dist, proto_weights = self.proto_fusion(scale_proto_dists)

        proto_details = {
            'max_sims': torch.stack(all_max_sims, dim=-1),
            'match_idxs': torch.stack(all_match_idxs, dim=-1),
            'route_weights': all_route_weights,
            'scale_proto_dists': scale_proto_dists,
        }

        return (fused_output, scale_outputs, recon_weights,
                fused_proto_dist, scale_proto_dists, proto_weights,
                total_balance_loss, total_basis_loss, proto_details)

    def set_proto_phase(self, phase):
        self.global_memory.set_phase(phase)

    def cluster_and_initialize_all(self, device=None, verbose=True):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Clustering SHARED prototypes ...")
            print(f"{'=' * 60}")
        
        buf_size = self.global_memory.get_buffer_size()
        if verbose:
            print(f"  Global buffer size: {buf_size}")
        
        self.global_memory.cluster_and_initialize(device=device, verbose=verbose)

    def all_protos_initialized(self):
        return self.global_memory.is_initialized()

    def get_all_proto_stats(self):
        return {'Global': self.global_memory.get_stats()}

    def get_all_basis_diagnostics(self):
        diags = {}
        for i, dsr in enumerate(self.scales):
            diags[f'DSR={dsr}'] = self.fits_models[i].get_basis_diagnostics()
        return diags

    def get_multiscale_loss(self, x_full, gumbel_temp=None, hard=False, update_proto=True):
        (fused_output, scale_outputs, recon_weights,
         fused_proto_dist, scale_proto_dists, proto_weights,
         total_balance_loss, total_basis_loss, proto_details) = self.forward(
            x_full, gumbel_temp=gumbel_temp, hard=hard, update_proto=update_proto
        )
        
        recon_loss = F.mse_loss(fused_output, x_full)
        
        scale_losses = {}
        for i, scale_output in enumerate(scale_outputs):
            scale_loss = F.mse_loss(scale_output, x_full)
            scale_losses[f'loss_scale_{self.scales[i]}'] = scale_loss.item()
        
        total_loss = recon_loss + 0.01 * total_balance_loss + 0.001 * total_basis_loss
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'balance_loss': total_balance_loss.item(),
            'basis_loss': total_basis_loss.item(),
            **scale_losses
        }
        
        return total_loss, loss_dict
