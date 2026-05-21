#!/bin/bash

# SpectraGuard: Run all datasets
# Usage: bash run_all.sh

echo "============================================================"
echo "SpectraGuard: Multi-Dataset Training"
echo "============================================================"
echo ""

EPOCHS=3
BATCH_SIZE=256
LR=1e-4
WIN_SIZE=100
SCALES="2 4 8"

DATASETS=("SMD" "MSL" "SMAP" "PSM" "SWAT")

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Window size: $WIN_SIZE"
echo "  Scales: $SCALES"
echo ""

SUCCESS=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running $dataset"
    echo "============================================================"
    echo ""
    
    python train_spectraguard.py \
        --dataset $dataset \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --win_size $WIN_SIZE \
        --scales $SCALES
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $dataset completed successfully"
        ((SUCCESS++))
    else
        echo ""
        echo "✗ $dataset failed"
        ((FAILED++))
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Success: $SUCCESS"
echo "Failed: $FAILED"
echo ""
