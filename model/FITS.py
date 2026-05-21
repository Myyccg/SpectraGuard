import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.target_freq_bins = int(self.dominance_freq * self.length_ratio)


        self.use_basis = getattr(configs, 'use_basis', False)

        if self.use_basis:
            from model.basis_upsampler import BasisGuidedUpsampler
            d_model = getattr(configs, 'd_model', 64)
            n_basis = getattr(configs, 'n_basis', 8)
            basis_type = getattr(configs, 'basis_type', 'mixed')

            self.basis_upsampler = BasisGuidedUpsampler(
                F_in=self.dominance_freq,
                F_out=self.target_freq_bins,
                d_model=d_model,
                n_basis=n_basis,
                basis_type=basis_type,
                use_residual=True,
            )
        else:

            if self.individual:
                self.freq_upsampler = nn.ModuleList()
                for i in range(self.channels):
                    self.freq_upsampler.append(
                        nn.Linear(self.dominance_freq, self.target_freq_bins).to(torch.cfloat)
                    )
            else:
                self.freq_upsampler = nn.Linear(
                    self.dominance_freq, self.target_freq_bins
                ).to(torch.cfloat)

    def forward(self, x, G_selected=None):

        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)


        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0
        low_specx = low_specx[:, 0:self.dominance_freq, :]


        if self.use_basis and G_selected is not None:

            C = low_specx.size(-1)
            low_specx_ri = torch.cat([low_specx.real, low_specx.imag], dim=-1)


            low_specxy_ri = self.basis_upsampler(low_specx_ri, G_selected)


            low_specxy_ = torch.complex(
                low_specxy_ri[:, :, :C],
                low_specxy_ri[:, :, C:]
            )


        else:

            if self.individual:
                low_specxy_ = torch.zeros(
                    [low_specx.size(0), self.target_freq_bins, low_specx.size(2)],
                    dtype=low_specx.dtype
                ).to(low_specx.device)
                for i in range(self.channels):
                    low_specxy_[:, :, i] = self.freq_upsampler[i](
                        low_specx[:, :, i].permute(0, 1)
                    ).permute(0, 1)
            else:
                low_specxy_ = self.freq_upsampler(
                    low_specx.permute(0, 2, 1)
                ).permute(0, 2, 1)


        low_specxy = torch.zeros(
            [low_specxy_.size(0),
             int((self.seq_len + self.pred_len) / 2 + 1),
             low_specxy_.size(2)],
            dtype=low_specxy_.dtype
        ).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_

        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio


        xy = low_xy * torch.sqrt(x_var) + x_mean

        return xy, low_xy * torch.sqrt(x_var)

    def get_basis_loss(self):
        if self.use_basis and hasattr(self, 'basis_upsampler'):
            return self.basis_upsampler.basis_loss()
        return torch.tensor(0.0)

    def get_basis_diagnostics(self):
        if self.use_basis and hasattr(self, 'basis_upsampler'):
            return self.basis_upsampler.get_diagnostics()
        return {}
        

