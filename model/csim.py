
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CSIM(nn.Module):
    def __init__(self, channels, hidden_dim=32, freq_agg='mean', temperature=1.0, diag_bias=1.0, use_layer_norm=False):
        super().__init__()
        
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.freq_agg = freq_agg
        self.use_layer_norm = use_layer_norm
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.diag_bias = nn.Parameter(torch.tensor(diag_bias))
        
        self.rho_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.beta_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.phi_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.nu_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if use_layer_norm:
            self.ln = nn.LayerNorm(1)
        
        self.freq_weight_net = None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.rho_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        
        for m in self.beta_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)
        
        for m in self.phi_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        
        for m in self.nu_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, X_f, return_details=False):
       
        B, F_s, C = X_f.shape
        
        if not torch.is_complex(X_f):
            X_f = X_f.to(torch.cfloat)
        
        h_i = self._aggregate_frequency(X_f)
        
        if self.use_layer_norm:
            h_i = self.ln(h_i.unsqueeze(-1)).squeeze(-1)
        
        h_i_input = h_i.unsqueeze(-1)
        
        rho_h = self.rho_proj(h_i_input)
        beta_h = self.beta_proj(h_i_input)
        exp_beta_h = torch.exp(beta_h)
        
        S1 = torch.bmm(rho_h, exp_beta_h.transpose(1, 2))
        
        phi_h = self.phi_proj(h_i_input)
        nu_h = self.nu_proj(h_i_input)
        
        S2 = torch.bmm(phi_h, nu_h.transpose(1, 2))
        
        S = S1 + S2
        
        diag_mask = torch.eye(C, device=X_f.device, dtype=S.dtype)
        S_biased = S + self.diag_bias * diag_mask.unsqueeze(0)
        
        tau = torch.clamp(self.temperature, min=0.1)
        W = F.softmax(S_biased / tau, dim=-1)
        
        W_complex = W.to(X_f.dtype)
        X_f_modulated = torch.bmm(X_f, W_complex.transpose(1, 2))
        
        if return_details:
            details = {
                'h_i': h_i,
                'S1': S1,
                'S2': S2,
                'S': S,
                'S_biased': S_biased,
                'W': W,
                'rho_h': rho_h,
                'beta_h': beta_h,
                'phi_h': phi_h,
                'nu_h': nu_h,
            }
            return X_f_modulated, W, details
        else:
            return X_f_modulated, W
    
    def _aggregate_frequency(self, X_f):
        X_f_abs = torch.abs(X_f)
        
        if self.freq_agg == 'mean':
            h_i = X_f_abs.mean(dim=1)
        
        elif self.freq_agg == 'max':
            h_i = X_f_abs.max(dim=1)[0]
        
        elif self.freq_agg == 'weighted':
            if self.freq_weight_net is None:
                F_s = X_f.size(1)
                self.freq_weight_net = nn.Sequential(
                    nn.Linear(F_s, F_s // 2),
                    nn.ReLU(),
                    nn.Linear(F_s // 2, F_s),
                    nn.Softmax(dim=-1)
                ).to(X_f.device)
            
            weights = self.freq_weight_net(X_f_abs.transpose(1, 2))
            h_i = (X_f_abs * weights.transpose(1, 2)).sum(dim=1)
        
        else:
            h_i = X_f_abs.mean(dim=1)
        
        return h_i
    
    def get_relation_matrix(self, X_f):
        _, W = self.forward(X_f, return_details=False)
        return W
    
    def check_asymmetry(self, W):
        W_T = W.transpose(1, 2)
        diff = torch.abs(W - W_T)
        
        C = W.size(1)
        mask = 1 - torch.eye(C, device=W.device).unsqueeze(0)
        diff_off_diag = diff * mask
        
        asymmetry_score = diff_off_diag.mean()
        max_diff = diff_off_diag.max()
        
        return asymmetry_score.item(), max_diff.item()


class CSIM_MultiScale(nn.Module):
    def __init__(self, channels, scales, hidden_dim=32, shared_params=False, **kwargs):
        super().__init__()
        
        self.channels = channels
        self.scales = scales
        self.shared_params = shared_params
        
        if shared_params:
            self.csim = CSIM(channels, hidden_dim, **kwargs)
        else:
            self.csim_modules = nn.ModuleDict({
                f'scale_{s}': CSIM(channels, hidden_dim, **kwargs)
                for s in scales
            })
    
    def forward(self, X_f_dict, return_details=False):
        X_f_modulated_dict = {}
        W_dict = {}
        details_dict = {} if return_details else None
        
        for scale, X_f in X_f_dict.items():
            if self.shared_params:
                csim = self.csim
            else:
                csim = self.csim_modules[f'scale_{scale}']
            
            if return_details:
                X_f_mod, W, details = csim(X_f, return_details=True)
                details_dict[scale] = details
            else:
                X_f_mod, W = csim(X_f, return_details=False)
            
            X_f_modulated_dict[scale] = X_f_mod
            W_dict[scale] = W
        
        if return_details:
            return X_f_modulated_dict, W_dict, details_dict
        else:
            return X_f_modulated_dict, W_dict


def visualize_relation_matrix(W, channel_names=None, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if W.dim() == 3:
        W = W[0]
    
    W_np = W.detach().cpu().numpy()
    C = W_np.shape[0]
    
    asymmetry = np.abs(W_np - W_np.T)
    off_diag_mask = 1 - np.eye(C)
    asymmetry_score = (asymmetry * off_diag_mask).mean()
    
    diag_mean = np.diag(W_np).mean()
    off_diag_mean = (W_np * off_diag_mask).sum() / (C * (C - 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(W_np, ax=axes[0], cmap='RdYlBu_r', center=0,
                xticklabels=channel_names or range(C),
                yticklabels=channel_names or range(C),
                cbar_kws={'label': 'Relation Strength'})
    axes[0].set_title(f'Channel Relation Matrix W\n'
                     f'Diag: {diag_mean:.3f}, Off-diag: {off_diag_mean:.3f}')
    axes[0].set_xlabel('Channel j (source)')
    axes[0].set_ylabel('Channel i (target)')
    
    sns.heatmap(asymmetry, ax=axes[1], cmap='Reds',
                xticklabels=channel_names or range(C),
                yticklabels=channel_names or range(C),
                cbar_kws={'label': '|W_ij - W_ji|'})
    axes[1].set_title(f'Asymmetry Matrix\n'
                     f'Mean asymmetry: {asymmetry_score:.4f}')
    axes[1].set_xlabel('Channel j')
    axes[1].set_ylabel('Channel i')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return asymmetry_score
