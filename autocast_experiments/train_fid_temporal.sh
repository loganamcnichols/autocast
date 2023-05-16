#!/bin/sh
python train_fid_temporal.py \
    --max_seq_len 128 \
    --finetune_encoder 0 \
    --adjust_targets 1 \
    --model_size small \
    --per_gpu_batch_size 8 \
    --epochs 1 \
    --answer_maxlength 12 \
    --text_maxlength 512 \
    --train_questions data/train_questions.json \
    --test_questions data/test_questions.json \
    --train_crowd data/train_crowd.json \
    --test_crowd data/test_crowd.json \
    --train_schedule data/train_reading.json \
    --test_schedule data/test_reading.json \
    --n_context 2 \
    --name temporal_t5_3b_top2_seqlen128_fixed_wdecay1e-2_lr5e-5_bs8_ep5_retrbm25ce_finetune0_adjusttarget1 \
    --optim adamw \
    --lr 5e-5 \
    --weight_decay 1e-2 \
    --scheduler fixed \
    --warmup_steps 100 \
    --train_data_size 4387

