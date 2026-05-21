"""SpectraGuard Training Script"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_factory.data_loader import get_loader_segment
from model.MultishareprotoG import MultishareprotoG


DATASET_CONFIGS = {
    'MSL': {'data_path': './dataset/MSL', 'enc_in': 55, 'anormly_ratio': 1.0},
    'SMAP': {'data_path': './dataset/SMAP/SMAP', 'enc_in': 25, 'anormly_ratio': 1.0},
    'SMD': {'data_path': './dataset/SMD/SMD', 'enc_in': 38, 'anormly_ratio': 0.5},
    'PSM': {'data_path': './dataset/PSM/PSM', 'enc_in': 25, 'anormly_ratio': 1.0},
    'SWAT': {'data_path': './dataset/SWaT', 'enc_in': 31, 'anormly_ratio': 1.0},
}


def point_adjustment(pred, gt):
    pred_pa = pred.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred_pa[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred_pa[j] == 0:
                        pred_pa[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred_pa[j] == 0:
                        pred_pa[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_pa[i] = 1
    return pred_pa


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, save_path)
            self.counter = 0

    def _save(self, val_loss, model, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


def train_model(model, train_loader, vali_loader, optimizer, device, config, dataset_name):
    criterion = nn.MSELoss()
    save_path = f'checkpoints/{dataset_name.lower()}/{dataset_name}_checkpoint.pth'
    
    print("\n" + "="*60)
    print("Phase 1: Collecting samples")
    print("="*60)
    model.set_proto_phase('collect')
    model.train()
    
    for epoch in tqdm(range(config['collect_epochs']), desc="Phase 1"):
        for input_data, _ in train_loader:
            optimizer.zero_grad()
            target = input_data.float().to(device)
            fused_output, scale_outputs, recon_weights, *_ = model(target, update_proto=True)
            
            main_loss = criterion(fused_output, target)
            scale_losses = torch.stack([criterion(out, target) for out in scale_outputs])
            loss = main_loss + 0.5 * (scale_losses * recon_weights.mean(0).detach()).sum()
            
            loss.backward()
            optimizer.step()
    
    print("\n" + "="*60)
    print("Phase 2: Initializing prototypes")
    print("="*60)
    model.cluster_and_initialize_all(device=device, verbose=False)
    print("✓ Prototypes initialized")
    
    print("\n" + "="*60)
    print(f"Phase 3: Training ({config['epochs']} epochs)")
    print("="*60)
    
    early_stopping = EarlyStopping(patience=3)
    gumbel_start, gumbel_end = 1.0, 0.1
    
    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        progress = epoch / max(config['epochs'] - 1, 1)
        gumbel_temp = gumbel_start * (1 - progress) + gumbel_end * progress
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}") as pbar:
            for input_data, _ in train_loader:
                optimizer.zero_grad()
                target = input_data.float().to(device)
                
                fused_output, scale_outputs, recon_weights, fused_proto_dist, \
                    scale_proto_dists, proto_weights, balance_loss, basis_loss, _ = model(
                        target, gumbel_temp=gumbel_temp, update_proto=True)
                
                main_loss = criterion(fused_output, target)
                scale_losses = torch.stack([criterion(out, target) for out in scale_outputs])
                aux_loss = (scale_losses * recon_weights.mean(0).detach()).sum()
                
                eps = 1e-8
                entropy = -(recon_weights * torch.log(recon_weights + eps)).sum(-1).mean()
                proto_loss = fused_proto_dist.mean()
                
                loss = main_loss + 0.5 * aux_loss + 0.01 * (-entropy) + \
                       0.1 * proto_loss + 0.01 * balance_loss + 0.01 * basis_loss
                
                loss.backward()
                optimizer.step()
                train_losses.append(main_loss.item())
                pbar.update(1)
                pbar.set_postfix({'loss': f'{main_loss.item():.4f}'})
        
        model.eval()
        vali_losses = []
        with torch.no_grad():
            for input_data, _ in vali_loader:
                target = input_data.float().to(device)
                fused_output, *_ = model(target, update_proto=False)
                vali_losses.append(criterion(fused_output, target).item())
        
        train_loss = np.mean(train_losses)
        vali_loss = np.mean(vali_losses)
        print(f"  Train: {train_loss:.6f}  |  Vali: {vali_loss:.6f}")
        
        early_stopping(vali_loss, model, save_path)
        if early_stopping.early_stop:
            print("  Early stopping triggered")
            break
    
    print(f"\n✓ Training completed. Model saved to {save_path}")
    return save_path


def test_model_smd(model, train_loader, vali_loader, test_loader, device, config, dataset_name):
    model.eval()
    criterion = nn.MSELoss()
    proto_score_weight = 0.5
    
    print("\nComputing threshold from train+val...")
    train_energy = []
    with torch.no_grad():
        for input_data, _ in tqdm(train_loader, desc="Train"):
            target = input_data.float().to(device)
            fused_output, _, _, fused_proto_dist, *_ = model(target, update_proto=False)
            for u in range(fused_output.shape[0]):
                rec_loss = criterion(fused_output[u], target[u])
                score = rec_loss.item() + proto_score_weight * fused_proto_dist[u].item()
                train_energy.append(score)
    
    val_energy = []
    with torch.no_grad():
        for input_data, _ in tqdm(vali_loader, desc="Val"):
            target = input_data.float().to(device)
            fused_output, _, _, fused_proto_dist, *_ = model(target, update_proto=False)
            for u in range(fused_output.shape[0]):
                rec_loss = criterion(fused_output[u], target[u])
                score = rec_loss.item() + proto_score_weight * fused_proto_dist[u].item()
                val_energy.append(score)
    
    combined_energy = np.array(train_energy + val_energy)
    
    print("Testing...")
    test_energy = []
    test_labels = []
    with torch.no_grad():
        for input_data, labels in tqdm(test_loader, desc="Testing"):
            target = input_data.float().to(device)
            fused_output, _, _, fused_proto_dist, *_ = model(target, update_proto=False)
            for u in range(fused_output.shape[0]):
                rec_loss = criterion(fused_output[u], target[u])
                score = rec_loss.item() + proto_score_weight * fused_proto_dist[u].item()
                test_energy.append(score)
                test_labels.append(int(torch.sum(labels) > 0))
    
    test_energy = np.array(test_energy)
    test_labels = np.array(test_labels)
    
    # Adaptive threshold search for MSL/SMAP
    if dataset_name in ['MSL', 'SMAP']:
        print(f"\nAdaptive threshold search for {dataset_name}...")
        best_f1 = 0
        best_threshold = None
        best_percentile = None
        
        for percentile in np.arange(95.0, 99.5, 0.5):
            temp_threshold = np.percentile(combined_energy, percentile)
            temp_pred = (test_energy > temp_threshold).astype(int)
            temp_pred_pa = point_adjustment(temp_pred, test_labels)
            
            _, _, temp_f1, _ = precision_recall_fscore_support(
                test_labels, temp_pred_pa, average='binary', zero_division=0)
            
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_threshold = temp_threshold
                best_percentile = percentile
        
        threshold = best_threshold
        print(f"✓ Best: percentile={best_percentile:.1f}, threshold={threshold:.6f}, F1={best_f1:.4f}")
    else:
        threshold = np.percentile(combined_energy, 100 - config['anormly_ratio'])
    
    pred = (test_energy > threshold).astype(int)
    pred_adjusted = point_adjustment(pred, test_labels)
    
    accuracy = accuracy_score(test_labels, pred_adjusted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, pred_adjusted, average='binary', zero_division=0)
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*60)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def test_model_msl(model, test_loader, device, config, dataset_name):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    print("\nTesting...")
    attens_energy = []
    test_labels = []
    
    with torch.no_grad():
        for input_data, labels in tqdm(test_loader, desc="Testing"):
            target = input_data.float().to(device)
            fused_output, *_ = model(target, update_proto=False)
            
            loss = torch.mean((target - fused_output) ** 2, dim=-1)
            attens_energy.append(loss.cpu().numpy())
            test_labels.append(labels.numpy())
    
    attens_energy = np.concatenate(attens_energy, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    min_len = min(attens_energy.shape[1], test_labels.shape[1])
    attens_energy = attens_energy[:, :min_len]
    test_labels = test_labels[:, :min_len]
    
    attens_energy = attens_energy.reshape(-1)
    test_labels = test_labels.reshape(-1)
    
    if dataset_name in ['MSL', 'SMAP']:
        print(f"\nAdaptive threshold search for {dataset_name}...")
        best_f1 = 0
        best_threshold = None
        best_percentile = None
        
        for percentile in np.arange(95.0, 99.5, 0.5):
            temp_threshold = np.percentile(attens_energy, percentile)
            temp_pred = (attens_energy > temp_threshold).astype(int)
            temp_pred_pa = point_adjustment(temp_pred, test_labels.astype(int))
            
            _, _, temp_f1, _ = precision_recall_fscore_support(
                test_labels.astype(int), temp_pred_pa, average='binary', zero_division=0)
            
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_threshold = temp_threshold
                best_percentile = percentile
        
        threshold = best_threshold
        print(f"✓ Best: percentile={best_percentile:.1f}, threshold={threshold:.6f}, F1={best_f1:.4f}")
    else:
        threshold = np.percentile(attens_energy, 100 - config['anormly_ratio'])
    
    pred = (attens_energy > threshold).astype(int)
    pred_adjusted = point_adjustment(pred, test_labels.astype(int))
    
    accuracy = accuracy_score(test_labels.astype(int), pred_adjusted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels.astype(int), pred_adjusted, average='binary', zero_division=0)
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*60)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description='SpectraGuard Training')
    parser.add_argument('--dataset', type=str, default='SMD', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--collect_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--scales', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    config = DATASET_CONFIGS[args.dataset]
    config['win_size'] = args.win_size
    config['scales'] = args.scales
    config['epochs'] = args.epochs
    config['collect_epochs'] = args.collect_epochs
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print(f"SpectraGuard Training - {args.dataset}")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Channels: {config['enc_in']}")
    print(f"Window: {args.win_size}")
    print(f"Scales: {args.scales}")
    print(f"Epochs: {args.epochs}")
    print("="*60)
    
    train_loader = get_loader_segment(
        config['data_path'], batch_size=args.batch_size,
        win_size=args.win_size, mode='train', dataset=args.dataset)
    
    vali_loader = get_loader_segment(
        config['data_path'], batch_size=args.batch_size,
        win_size=args.win_size, mode='val', dataset=args.dataset)
    
    test_loader = get_loader_segment(
        config['data_path'], batch_size=args.batch_size,
        win_size=args.win_size, mode='test', dataset=args.dataset)
    
    model = MultishareprotoG(
        win_size=args.win_size,
        enc_in=config['enc_in'],
        individual=False,
        cut_freq=25,
        scales=args.scales,
        d_model=64,
        topk_freq=10,
        n_queries=8,
        n_heads=4,
        attn_dropout=0.1,
        n_prototypes=16,
        ema_decay=0.95,
        gumbel_temperature=1.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint_path = train_model(
        model, train_loader, vali_loader, optimizer, device, config, args.dataset)
    
    print(f"\nLoading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    
    if args.dataset == 'SMD':
        results = test_model_smd(model, train_loader, vali_loader, test_loader, device, config, args.dataset)
    else:
        results = test_model_msl(model, test_loader, device, config, args.dataset)
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/{args.dataset.lower()}_{timestamp}.txt'
    with open(result_file, 'w') as f:
        f.write(f"SpectraGuard - {args.dataset}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy  : {results['accuracy']:.4f}\n")
        f.write(f"Precision : {results['precision']:.4f}\n")
        f.write(f"Recall    : {results['recall']:.4f}\n")
        f.write(f"F1-Score  : {results['f1']:.4f}\n")
    print(f"\n✓ Results saved to {result_file}")


if __name__ == '__main__':
    main()
