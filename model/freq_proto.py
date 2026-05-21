
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter





class FrequencyRouter(nn.Module):

    def __init__(self, seq_len, channels, d_model,
                 topk_freq=10, n_queries=8, query_len=None,
                 gumbel_temperature=1.0):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.d_model = d_model
        self.topk_freq = topk_freq
        self.n_queries = n_queries
        self.query_len = query_len or max(seq_len // 4, 4)
        self.gumbel_temperature = gumbel_temperature
        self.freq_bins = seq_len // 2 + 1

        self.query_library = nn.Parameter(
            torch.randn(n_queries, self.query_len, d_model) * 0.02
        )


        self.router_mlp = nn.Sequential(
            nn.Linear(self.freq_bins, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_queries),
        )

        self.query_proj = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.topk_tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, X_f, temperature=None, hard=False):
        B = X_f.size(0)

        A = torch.abs(X_f)
        A_mean = A.mean(dim=-1)

        tau = torch.clamp(self.topk_tau, min=0.1)
        freq_weights = F.softmax(A_mean / tau, dim=1)

        X_f_masked = X_f * freq_weights.unsqueeze(-1)


        u = self.router_mlp(A_mean)

        temp = temperature if temperature is not None else self.gumbel_temperature
        weights = F.gumbel_softmax(u, tau=temp, hard=hard, dim=-1)

        w = weights.unsqueeze(-1).unsqueeze(-1)
        Q_lib = self.query_library.unsqueeze(0)
        Q = (w * Q_lib).sum(dim=1)

        Q = self.query_proj(Q)
        Q = self.norm_q(Q)

        return Q, weights





class FrequencyEmbedding(nn.Module):

    def __init__(self, seq_len, channels, d_model):
        super().__init__()
        self.freq_bins = seq_len // 2 + 1

        self.k_proj = nn.Linear(channels * 2, d_model)
        self.v_proj = nn.Linear(channels * 2, d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

    def forward(self, X_f):

        X_ri = torch.cat([X_f.real, X_f.imag], dim=-1)
        K = self.norm_k(self.k_proj(X_ri))
        V = self.norm_v(self.v_proj(X_ri))
        return K, V





class FreqCrossAttention(nn.Module):

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm_r = nn.LayerNorm(d_model)
        self.scale = math.sqrt(self.d_k)

    def forward(self, Q, K, V):
        B, L_q, _ = Q.shape
        F_bins = K.size(1)

        Q_ = self.w_q(Q).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        K_ = self.w_k(K).view(B, F_bins, self.n_heads, self.d_k).transpose(1, 2)
        V_ = self.w_v(V).view(B, F_bins, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        R = torch.matmul(attn, V_)
        R = R.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        R = self.norm_r(self.w_o(R))

        return R, attn





class PrototypeMemory(nn.Module):

    def __init__(self, d_model, n_prototypes=16, decay=0.95,
                 max_buffer_size=50000):
        super().__init__()
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        self.base_decay = decay
        self.max_buffer_size = max_buffer_size


        G_init = torch.randn(n_prototypes, d_model)
        G_init = F.normalize(G_init, dim=-1)
        self.register_buffer('G', G_init)


        self.register_buffer('G_count', torch.zeros(n_prototypes))


        self.register_buffer('G_initialized', torch.tensor(False))


        self._r_buffer = []


        self._phase = 'collect'

    @property
    def phase(self):
        return self._phase

    def set_phase(self, phase):
        assert phase in ('collect', 'train'), f"Unknown phase: {phase}"
        self._phase = phase

    def forward(self, R, update=True):
        B = R.size(0)


        r = R.mean(dim=1)
        r_hat = F.normalize(r, dim=-1)


        if self._phase == 'collect':
            self._collect(r_hat)

            with torch.no_grad():
                G_hat = F.normalize(self.G, dim=-1)
                sim = torch.matmul(r_hat, G_hat.T)
                max_sim, match_idx = sim.max(dim=1)
            proto_dist = 1.0 - max_sim
            G_selected = self.G[match_idx]
            balance_loss = torch.tensor(0.0, device=R.device)
            return G_selected, max_sim, proto_dist, match_idx, balance_loss


        G_hat = F.normalize(self.G, dim=-1)
        sim = torch.matmul(r_hat, G_hat.T)
        max_sim, match_idx = sim.max(dim=1)

        proto_dist = 1.0 - max_sim


        if update and self.training:
            self._ema_update(r_hat, match_idx, G_hat)

        G_selected = self.G[match_idx]


        balance_loss = self._compute_balance_loss(match_idx, B)

        return G_selected, max_sim, proto_dist, match_idx, balance_loss

    def _collect(self, r_hat):
        r_np = r_hat.detach().cpu().numpy()
        for i in range(r_np.shape[0]):
            if len(self._r_buffer) < self.max_buffer_size:
                self._r_buffer.append(r_np[i])

    def cluster_and_initialize(self, device=None, verbose=True):
        n_collected = len(self._r_buffer)
        if n_collected < self.n_prototypes * 2:
            if verbose:
                print(f"  ⚠️ Not enough samples: {n_collected} < {self.n_prototypes * 2}")
            self._phase = 'train'
            return None

        if device is None:
            device = self.G.device

        if verbose:
            print(f"  Clustering {n_collected} samples → {self.n_prototypes} prototypes ...")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize


        R_all = np.stack(self._r_buffer)
        R_all_norm = normalize(R_all, axis=1)


        kmeans = KMeans(
            n_clusters=self.n_prototypes,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        kmeans.fit(R_all_norm)


        if verbose:
            labels = kmeans.labels_
            counter = Counter(labels)

            print(f"  Cluster distribution:")
            for i in range(self.n_prototypes):
                cnt = counter.get(i, 0)
                pct = cnt / len(labels) * 100
                bar = '█' * max(1, int(pct / 2))
                print(f"    Proto {i:2d}: {cnt:5d} ({pct:5.1f}%) {bar}")


            usage = np.array([counter.get(i, 0) for i in range(self.n_prototypes)])
            usage = usage / usage.sum() + 1e-10
            entropy = -np.sum(usage * np.log(usage))
            max_entropy = np.log(self.n_prototypes)
            print(f"  Usage entropy: {entropy:.3f} / {max_entropy:.3f} "
                  f"({entropy / max_entropy * 100:.1f}%)")


            centers_norm = normalize(kmeans.cluster_centers_, axis=1)
            sim_mat = centers_norm @ centers_norm.T
            off_diag = sim_mat[~np.eye(self.n_prototypes, dtype=bool)]
            print(f"  Inter-prototype similarity: "
                  f"mean={off_diag.mean():.4f}, "
                  f"max={off_diag.max():.4f}, "
                  f"min={off_diag.min():.4f}")


        centers = normalize(kmeans.cluster_centers_, axis=1)
        G_new = torch.from_numpy(centers).float().to(device)
        self.G.data.copy_(G_new)
        self.G_initialized.fill_(True)
        self.G_count.zero_()


        self._r_buffer = []


        self._phase = 'train'

        if verbose:
            print(f"  ✅ G initialized. Phase → train.")

        return kmeans

    def _ema_update(self, r_hat, match_idx, G_hat):
        with torch.no_grad():
            for idx in range(self.n_prototypes):
                mask = (match_idx == idx)
                if mask.sum() == 0:
                    continue

                avg_r = r_hat[mask].mean(dim=0)
                avg_r = F.normalize(avg_r, dim=-1)


                count = self.G_count[idx].item()
                adaptive_decay = self.base_decay * min(1.0, count / 100.0 + 0.5)

                new_g = adaptive_decay * G_hat[idx] + (1 - adaptive_decay) * avg_r
                new_g = F.normalize(new_g, dim=-1)

                self.G[idx].copy_(new_g)
                self.G_count[idx] += mask.sum()

    def _compute_balance_loss(self, match_idx, batch_size):
        one_hot = F.one_hot(match_idx, self.n_prototypes).float()
        f = one_hot.mean(dim=0)
        balance_loss = self.n_prototypes * (f * f).sum()
        return balance_loss

    def get_stats(self):
        G_hat = F.normalize(self.G, dim=-1)
        sim_mat = torch.matmul(G_hat, G_hat.T)
        off_diag_mask = ~torch.eye(self.n_prototypes, dtype=bool, device=self.G.device)
        off_diag = sim_mat[off_diag_mask]

        return {
            'G': self.G.clone(),
            'G_count': self.G_count.clone(),
            'utilization': (self.G_count > 0).float().mean().item(),
            'inter_sim_mean': off_diag.mean().item(),
            'inter_sim_max': off_diag.max().item(),
            'initialized': self.G_initialized.item(),
            'phase': self._phase,
        }

    def get_buffer_size(self):
        return len(self._r_buffer)

    def is_initialized(self):
        return self.G_initialized.item()





class FreqGateGenerator(nn.Module):

    def __init__(self, d_model, target_freq_bins, channels=None, per_channel=False):
        super().__init__()
        self.target_freq_bins = target_freq_bins
        self.per_channel = per_channel

        if per_channel and channels is not None:
            out_dim = target_freq_bins * channels
            self.channels = channels
        else:
            out_dim = target_freq_bins
            self.channels = 1

        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )




        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

    def forward(self, G_selected, max_sim=None):
        B = G_selected.size(0)

        raw_gate = self.gate_net(G_selected)
        gate = torch.sigmoid(raw_gate)



        if max_sim is not None:


            strength = (1.0 - max_sim).unsqueeze(-1)
            gate = 1.0 - strength * (1.0 - gate)



        if self.per_channel:
            freq_gate = gate.view(B, self.target_freq_bins, self.channels)
        else:
            freq_gate = gate.view(B, self.target_freq_bins, 1)

        return freq_gate





class FreProG(nn.Module):

    def __init__(self, seq_len, channels, d_model=64,
                 topk_freq=10, n_queries=8, query_len=None,
                 gumbel_temperature=1.0, n_heads=4, attn_dropout=0.1,
                 n_prototypes=16, ema_decay=0.95, max_buffer_size=50000):
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

        self.memory = PrototypeMemory(
            d_model=d_model,
            n_prototypes=n_prototypes,
            decay=ema_decay,
            max_buffer_size=max_buffer_size,
        )

    def forward(self, x_down, gumbel_temp=None, hard=False, update_proto=True):
        X_f = torch.fft.rfft(x_down, dim=1)

        Q, route_weights = self.router(X_f, temperature=gumbel_temp, hard=hard)

        K, V = self.embedding(X_f)

        R, attn_weights = self.cross_attn(Q, K, V)

        G_selected, max_sim, proto_dist, match_idx, balance_loss = \
            self.memory(R, update=update_proto)


        return proto_dist, max_sim, match_idx, route_weights, attn_weights, balance_loss, G_selected


    def set_phase(self, phase):
        self.memory.set_phase(phase)

    def cluster_and_initialize(self, device=None, verbose=True):
        return self.memory.cluster_and_initialize(device=device, verbose=verbose)

    def get_proto_stats(self):
        return self.memory.get_stats()

    def get_buffer_size(self):
        return self.memory.get_buffer_size()

    def is_initialized(self):
        return self.memory.is_initialized()
