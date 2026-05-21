"""
简化的Affiliation指标实现
用于异常检测的段级别评估
"""
import numpy as np


def calculate_affiliation_metrics(gt, pred):
    """
    计算Affiliation指标
    
    Args:
        gt: ground truth labels (0/1)
        pred: predicted labels (0/1)
    
    Returns:
        dict: 包含各种affiliation指标
    """
    # 简化实现：返回基本指标
    gt = np.array(gt).astype(int)
    pred = np.array(pred).astype(int)
    
    # 找到所有异常段
    gt_segments = _find_segments(gt)
    pred_segments = _find_segments(pred)
    
    if len(gt_segments) == 0 or len(pred_segments) == 0:
        return {
            'affiliation_precision': 0.0,
            'affiliation_recall': 0.0,
            'affiliation_f1': 0.0,
            'existence_affiliation': 0.0,
            'range_affiliation': 0.0
        }
    
    # 计算precision: 预测段与真实段的重叠
    precision_scores = []
    for pred_seg in pred_segments:
        max_overlap = 0
        for gt_seg in gt_segments:
            overlap = _segment_overlap(pred_seg, gt_seg)
            max_overlap = max(max_overlap, overlap)
        precision_scores.append(max_overlap)
    
    affiliation_precision = np.mean(precision_scores) if precision_scores else 0.0
    
    # 计算recall: 真实段被预测段覆盖的程度
    recall_scores = []
    for gt_seg in gt_segments:
        max_overlap = 0
        for pred_seg in pred_segments:
            overlap = _segment_overlap(gt_seg, pred_seg)
            max_overlap = max(max_overlap, overlap)
        recall_scores.append(max_overlap)
    
    affiliation_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    # F1-score
    if affiliation_precision + affiliation_recall > 0:
        affiliation_f1 = 2 * affiliation_precision * affiliation_recall / (affiliation_precision + affiliation_recall)
    else:
        affiliation_f1 = 0.0
    
    # Existence affiliation: 检测到的异常段比例
    detected_gt_segments = sum(1 for gt_seg in gt_segments 
                               if any(_segment_overlap(gt_seg, pred_seg) > 0 for pred_seg in pred_segments))
    existence_affiliation = detected_gt_segments / len(gt_segments) if gt_segments else 0.0
    
    # Range affiliation: 平均重叠比例
    range_affiliation = affiliation_recall
    
    return {
        'affiliation_precision': affiliation_precision,
        'affiliation_recall': affiliation_recall,
        'affiliation_f1': affiliation_f1,
        'existence_affiliation': existence_affiliation,
        'range_affiliation': range_affiliation
    }


def _find_segments(labels):
    """找到所有连续的异常段"""
    segments = []
    start = None
    
    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    
    if start is not None:
        segments.append((start, len(labels) - 1))
    
    return segments


def _segment_overlap(seg1, seg2):
    """计算两个段的重叠比例"""
    start1, end1 = seg1
    start2, end2 = seg2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start > overlap_end:
        return 0.0
    
    overlap_len = overlap_end - overlap_start + 1
    seg1_len = end1 - start1 + 1
    
    return overlap_len / seg1_len


def print_affiliation_results(metrics, title="Affiliation Metrics"):
    """打印affiliation指标结果"""
    if metrics is None:
        print(f"\n{title}: Not available")
        return
    
    print(f"\n{'='*80}")
    print(f"{title}:")
    print(f"{'='*80}")
    print(f"Affiliation Precision: {metrics['affiliation_precision']:.4f}")
    print(f"Affiliation Recall:    {metrics['affiliation_recall']:.4f}")
    print(f"Affiliation F1:        {metrics['affiliation_f1']:.4f}")
    print(f"Existence Affiliation: {metrics['existence_affiliation']:.4f}")
    print(f"Range Affiliation:     {metrics['range_affiliation']:.4f}")
    print(f"{'='*80}")
