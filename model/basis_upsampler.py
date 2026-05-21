
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasisGuidedUpsampler(nn.Module):

    def __init__(self, F_in, F_out, d_model, n_basis=8,
                 basis_type='mixed', use_residual=True,
                 init_residual_weight=0.5):
        super().__init__()

        self.F_in = F_in
        self.F_out = F_out
        self.d_model = d_model
        self.n_basis = n_basis
        self.use_residual = use_residual


        self.basis_networks = nn.ModuleList()
        for i in range(n_basis):
            if basis_type == 'linear':
                basis = nn.Linear(F_in, F_out)
            elif basis_type == 'spline':
                basis = nn.Sequential(
                    nn.Linear(F_in, F_out),
                    nn.Tanh(),
                    nn.Linear(F_out, F_out),
                )
            elif basis_type == 'mixed':


                if i % 2 == 0:
                    basis = nn.Linear(F_in, F_out)
                else:
                    basis = nn.Sequential(
                        nn.Linear(F_in, F_out),
                        nn.Tanh(),
                        nn.Linear(F_out, F_out),
                    )
            else:
                basis = nn.Linear(F_in, F_out)
            self.basis_networks.append(basis)



        self.semantic_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, n_basis),
        )

        self.alpha_temperature = nn.Parameter(torch.tensor(1.0))


        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(init_residual_weight))


        self._init_weights()


        self._last_alpha = None

    def _init_weights(self):
        for i, basis in enumerate(self.basis_networks):
            for module in (basis if isinstance(basis, nn.Sequential) else [basis]):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
                    if module.bias is not None:

                        nn.init.constant_(module.bias, 0.01 * (i - self.n_basis / 2))

    def forward(self, X_f_ri, G_selected):
        B, F_in, C2 = X_f_ri.shape


        z = self.semantic_proj(G_selected)
        alpha_logits = self.alpha_net(z)
        temp = torch.clamp(self.alpha_temperature, min=0.1)
        alpha = F.softmax(alpha_logits / temp, dim=-1)
        self._last_alpha = alpha


        X_t = X_f_ri.transpose(1, 2)


        basis_outputs = []
        for basis_net in self.basis_networks:
            out = basis_net(X_t)
            basis_outputs.append(out)


        basis_stack = torch.stack(basis_outputs, dim=1)
        alpha_w = alpha.view(B, self.n_basis, 1, 1)
        X_out_t = (alpha_w * basis_stack).sum(dim=1)


        X_out = X_out_t.transpose(1, 2)


        if self.use_residual:
            X_baseline = F.interpolate(
                X_t, size=self.F_out, mode='linear', align_corners=False
            ).transpose(1, 2)
            lam = torch.sigmoid(self.residual_weight)
            X_out = (1 - lam) * X_baseline + lam * X_out

        return X_out



    def orthogonal_loss(self):
        weights = []
        for basis in self.basis_networks:
            if isinstance(basis, nn.Linear):
                weights.append(basis.weight.flatten())
            elif isinstance(basis, nn.Sequential):
                for layer in basis:
                    if isinstance(layer, nn.Linear):
                        weights.append(layer.weight.flatten())
                        break

        if len(weights) < 2:
            return torch.tensor(0.0, device=weights[0].device)

        W = torch.stack(weights)
        W_norm = F.normalize(W, dim=1)
        sim = W_norm @ W_norm.T
        eye = torch.eye(self.n_basis, device=sim.device)
        return ((sim - eye) ** 2).mean()

    def entropy_loss(self):
        if self._last_alpha is None:
            return torch.tensor(0.0)
        alpha = self._last_alpha
        entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(self.n_basis), device=alpha.device))
        return -entropy / (max_entropy + 1e-8)

    def basis_loss(self):
        return self.orthogonal_loss() + 0.1 * self.entropy_loss()

    def get_diagnostics(self):
        diag = {}
        if self.use_residual:
            diag['residual_ratio'] = torch.sigmoid(self.residual_weight).item()
        if self._last_alpha is not None:
            alpha = self._last_alpha.detach()
            diag['alpha_mean'] = alpha.mean(dim=0).cpu().numpy()
            ent = -(alpha * (alpha + 1e-8).log()).sum(dim=-1).mean().item()
            max_ent = np.log(self.n_basis)
            diag['alpha_entropy_ratio'] = ent / max_ent
        return diag
