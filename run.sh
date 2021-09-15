#!/bin/bash 
input_dim=257
output_dim=257
left_context=0
right_context=0
lr=0.001

win_len=320
win_inc=160
fft_len=320

win_type=hanning
batch_size=2
max_epoch=100
rnn_units=256
rnn_layers=2


tr_clean_list='./data/clean_cv.lst'
tr_noise_list='./data/noise_cv.lst'

cv_noise_list='./data/noise_cv.lst'
cv_clean_list='./data/clean_cv.lst'

tr_rir_list='./data/rir_cv.lst'
cv_rir_list='./data/rir_cv.lst'

tt_list='./data/blind_test_track1.lst'

dropout=0.0
kernel_size=6
kernel_num=9
nropout=0.2
retrain=1
sample_rate=16k
num_gpu=1
batch_size=$[num_gpu*batch_size]



stage=SRNet
exp_dir=exp/sddnet/${stage}

if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi

train_stage=2

if [ $train_stage -le 1 ] ; then
#/home/work_nfs/common/tools/pyqueue_asr.pl \
    #-q g.q --gpu 1 --num-threads ${num_gpu} \
    #${exp_dir}/${save_name}.log \
    CUDA_VISIBLE_DEVICES=0,1 nohup python -u ./steps/run_sddnet.py \
    --decode=0 \
    --stage=${stage} \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-noise-list=${tr_noise_list} \
    --tr-clean-list=${tr_clean_list} \
    --tr-rir-list=${tr_rir_list} \
    --cv-noise-list=${cv_noise_list} \
    --cv-clean-list=${cv_clean_list} \
    --cv-rir-list=${cv_rir_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --window-type=${win_type} > ${exp_dir}/${stage}.log &
    exit 0
fi
###decode
if [ $train_stage -le 2 ] ; then 
    CUDA_VISIBLE_DEVICES='0' python -u ./steps/run_sddnet.py\
    --decode=1 \
    --stage=${stage} \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-noise-list=${tr_noise_list} \
    --tr-clean-list=${tr_clean_list} \
    --tr-rir-list=${tr_rir_list} \
    --cv-noise-list=${cv_noise_list} \
    --cv-clean-list=${cv_clean_list} \
    --cv-rir-list=${cv_rir_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    exit 0
fi
