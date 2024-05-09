#!/bin/bash

# Monolingual training data:
#     EN: /home/bodonlp/UnsupervisedMT/NMT/en-brx-data/mono/all.en.tok.6000.pth
#     FR: /home/bodonlp/UnsupervisedMT/NMT/en-brx-data/mono/all.brx.tok.6000.pth
# Parallel validation data:
#     EN: /home/bodonlp/UnsupervisedMT/NMT/en-brx-data/para/dev/dev.en.tok.6000.pth
#     FR: /home/bodonlp/UnsupervisedMT/NMT/en-brx-data/para/dev/dev.brx.tok.6000.pth
# Parallel test data:
#     EN: /home/bodonlp/UnsupervisedMT/NMT/en-brx-data/para/dev/test.en.tok.6000.pth
#     FR: /home/bodonlp/UnsupervisedMT/NMT/en-brx-data/para/dev/test.brx.tok.6000.pth
 
CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --exp_name en_brx_unmt_01 --transformer True \
    --n_enc_layers 4 --n_dec_layers 4 \
    --share_enc 3 --share_dec 3 \
    --share_lang_emb True --share_output_emb True \
    --langs 'en,gb' --n_mono -1 \
    --mono_dataset 'en:./en-gb-data/mono/all.en.tok.6000.pth,,;gb:./en-gb-data/mono/all.gb.tok.6000.pth,,' \
    --para_dataset 'en-gb:,./en-gb-data/para/dev/dev.XX.tok.6000.pth,./en-gb-data/para/dev/test.XX.tok.6000.pth' \
    --mono_directions 'en,gb' --word_shuffle 3 \
    --word_dropout 0.1 --word_blank 0.2 \
    --pivo_directions 'en-gb-en,gb-en-gb' \
    --pretrained_emb './tools/MUSE/dumped/debug/4ezm549qf4/best_mapping.pth' \
    --pretrained_out True \
    --lambda_xe_mono '0:1,87174:0.1,300000:0' \
    --lambda_xe_otfd 1 \
    --otf_num_processes 30 \
    --otf_sync_params_every 1000 \
    --enc_optimizer adam,lr=0.0001 \
    --group_by_size True \
    --batch_size 32 \
    --epoch_size 100 \
    --stopping_criterion bleu_en_gb_valid,10 
