
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import sys
import os
import argparse
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import scipy
import scipy.io as sio
import torch.optim as optim
import time
import multiprocessing
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(
    os.path.dirname(sys.path[0]) + '/tools/speech_processing_toolbox')

#from model.dc_crn import CRN as Model 
#from model.dc_crn_crnn import CRN as Model #crnn
from model.sddnet import SDDNet as Model###ccat
from tools.misc import get_learning_rate, save_checkpoint, reload_for_eval, reload_model
from tools.time_dataset_dereverb import make_loader, Processer, DataReader

import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(model, args, device, writer):
    print('preparing data...')
    dataloader, _ = make_loader(
        clean_scp=args.tr_clean,
        noise_scp=args.tr_noise,
        rir_scp=args.tr_rir, 
        segement_length=args.segement_length,
        repeat=2,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_threads,
        processer=Processer()
    )
    print_freq = 200
    num_batch = len(dataloader)
    params = model.get_params(args.weight_decay)
    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=1, verbose=True)
    
    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0
    print('---------PRERUN-----------')
    lr = get_learning_rate(optimizer)
    print('(Initialization)')
    val_loss, val_sisnr = 50,50 #validation(model, args, lr, -1, device)
    writer.add_scalar('Loss/Train', val_loss, step)
    writer.add_scalar('Loss/Cross-Validation', val_loss, step)
    
    writer.add_scalar('SISNR/Train', -val_sisnr, step)
    writer.add_scalar('SISNR/Cross-Validation', -val_sisnr, step)

    for epoch in range(start_epoch, args.max_epoch):
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        loss_total = 0.0
        loss_print = 0.0 

        sisnr_total = 0.0 
        sisnr_print = 0.0 
     
        freloss_total = 0.0
        freloss_print = 0.0

        stime = time.time()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            torch.cuda.empty_cache()
            inputs, labels_full, labels_direct, labels_late = data
            inputs = inputs.to(device)
            labels_full = labels_full.to(device)
            labels_direct = labels_direct.to(device)
            labels_late = labels_late.to(device)

            model.zero_grad()
            [est_spec,gth_spec], est_wav = data_parallel(model, (inputs,labels_direct))
            #loss = model.loss(est_wav, labels, 'SiSNR')
            #loss.backward()
            spec_loss = model.loss(est_spec, gth_spec, loss_mode='MSE') 
            rgkl_loss = model.loss(est_spec, gth_spec, loss_mode='rGKL')
            sisnr = model.loss(est_wav, labels_direct, loss_mode='SI-SNR') 
            #mae_loss = model.loss(est_spec, gth_spec, loss_mode='MAE')
            loss = sisnr + spec_loss + rgkl_loss


            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            #print(loss)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            step += 1

            loss_total += loss.data.cpu()
            loss_print += loss.data.cpu()

            sisnr_total += sisnr.data.cpu()
            sisnr_print += sisnr.data.cpu()

            freloss_total += spec_loss.data.cpu()
            freloss_print += spec_loss.data.cpu()

            if (idx+1) % 3000 == 0:
                save_checkpoint(model, optimizer, -1, step, args.exp_dir)
            if (idx + 1) % print_freq == 0:
                eplashed = time.time() - stime
                speed_avg = eplashed / (idx+1)
                loss_print_avg = loss_print / print_freq
                sisnr_print_avg = sisnr_print / print_freq
                freloss_print_avg = freloss_print / print_freq
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} |'
                      '{:2.3f}s/batches | loss {:2.6f} | freloss {:2.6f} |'
                      'SI-SNR {:2.4f} '.format(
                          epoch, args.max_epoch, idx + 1, num_batch, lr,
                          speed_avg, 
                          loss_print_avg,
                          freloss_print_avg,
                          -sisnr_print_avg,
                          
                          ))
                sys.stdout.flush()
                writer.add_scalar('Loss/Train', loss_print_avg, step)
                writer.add_scalar('SISNR/Train', -sisnr_print_avg, step)
                loss_print = 0.0
                sisnr_print=0.0
                freloss_print = 0.0
        eplashed = time.time() - stime
        loss_total_avg = loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
        freloss_total_avg = freloss_total / num_batch
        print(
            'Training AVG.LOSS |'
            ' Epoch {:3d}/{:3d} | lr {:1.4e} |'
            ' {:2.3f}s/batch | time {:3.2f}mins |'
            ' loss {:2.6f} |'
            ' freloss {:2.6f} |'
            ' SISNR {:2.4f}|'
                    .format(
                                    epoch + 1,
                                    args.max_epoch,
                                    lr,
                                    eplashed/num_batch,
                                    eplashed/60.0,
                                    loss_total_avg.item(),
                                    freloss_total_avg.item(),
                                    -sisnr_total_avg.item()
                        ))
        val_loss, val_sisnr= validation(model, args, lr, epoch, device)
        writer.add_scalar('Loss/Cross-Validation', val_loss, step)
        writer.add_scalar('SISNR/Cross-Validation', -val_sisnr, step)
        writer.add_scalar('learn_rate', lr, step) 
        if val_loss > scheduler.best:
            print('Rejected !!! The best is {:2.6f}'.format(scheduler.best))
        else:
            save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir)
        scheduler.step(val_loss)
        sys.stdout.flush()
        stime = time.time()


def validation(model, args, lr, epoch, device):
    dataloader, _ = make_loader( 
        clean_scp=args.cv_clean,
        noise_scp=args.cv_noise,
        rir_scp=args.cv_rir, 
        segement_length=args.segement_length,
        repeat=2,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_threads,
        processer=Processer()
    )
    model.eval()
    loss_total = 0.0 
    sisnr_total = 0.0 
    freloss_total = 0.0
    num_batch = len(dataloader)
    stime = time.time()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels_full, labels_direct, labels_late = data
            inputs = inputs.to(device)
            labels_full = labels_full.to(device)
            labels_direct = labels_direct.to(device)
            labels_late = labels_late.to(device)

            [est_spec,gth_spec], est_wav = data_parallel(model, (inputs,labels_direct))
            #gth_spec = data_parallel(model.stft, (labels))
            spec_loss = model.loss(est_spec, gth_spec, loss_mode='MSE')
            sisnr = model.loss(est_wav, labels_direct, loss_mode='SI-SNR') 
            rgkl_loss = model.loss(est_spec, gth_spec, loss_mode='rGKL')
            loss = sisnr + spec_loss + rgkl_loss
            #print(loss)
            loss_total += loss.data.cpu()
            freloss_total += spec_loss.data.cpu()
            sisnr_total += sisnr.data.cpu()
        #loss_total=sisnr_total

        etime = time.time()
        eplashed = (etime - stime) / num_batch
        loss_total_avg = loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
        freloss_total_avg = freloss_total / num_batch

    print('CROSSVAL AVG.LOSS | Epoch {:3d}/{:3d} '
          '| lr {:.4e} | {:2.3f}s/batch| time {:2.1f}mins '
          '| loss {:2.6f} |'
          '| freloss {:2.6f} |'
          '| SISNR {:2.4f} '.format(
                        epoch + 1,
                        args.max_epoch,
                        lr,
                        eplashed,
                        (etime - stime)/60.0,
                        loss_total_avg,
                        freloss_total_avg,
                        -sisnr_total_avg,
              ))
    sys.stdout.flush()
    return loss_total_avg, sisnr_total_avg


def decode(model, args, snr, device):
    model.eval()
    with torch.no_grad():
        
        data_reader = DataReader(
            args.tt_list,
            win_len=args.win_len,
            win_inc=args.win_inc,
            left_context=args.left_context,
            right_context=args.right_context,
            fft_len=args.fft_len,
            window_type=args.win_type,
            target_mode=args.target_mode,
            sample_rate=args.sample_rate)
        PATH = os.path.join(args.exp_dir, 'DNS-challenge-2021/')
        if not os.path.isdir(PATH):
            os.mkdir(PATH)
        
        filename = args.tt_list.split('/')[-1]
        filename = filename.split('.')[0]
        '''
        if snr is not None: 
            filename = 'out_noisy_' + str(snr)###
        '''
        if not os.path.isdir(os.path.join(PATH, filename)):
            os.mkdir(os.path.join(PATH, filename))  
        
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            inputs, utt_id, nsamples = data_reader[idx]
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.to(device) 
            outputs = model(inputs, )
            outputs = outputs / torch.max(torch.abs(outputs)) * torch.max(torch.abs(inputs))
            if outputs.dim() != 1:
                outputs = torch.squeeze(outputs).cpu().data.numpy()
            else:
                outputs = outputs.cpu().data.numpy()
            outputs = outputs[:nsamples]
            
            sf.write(os.path.join(PATH, filename + '/' + utt_id), outputs, args.sample_rate) 
            #sf.write('fuck.wav', outputs, args.sample_rate)
            del outputs, inputs
        print('Decode Done!!!')


def main(args):
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    args.sample_rate = {
        '8k':8000,
        '16k':16000,
        '24k':24000,
        '48k':48000,
    }[args.sample_rate]
    model = Model(win_len=args.win_len,
                  win_inc=args.win_inc,
                  fft_len=args.fft_len,
                  stage=args.stage)
    if not args.log_dir:
        writer = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard'))
    else:
        writer = SummaryWriter(args.log_dir)
    model.to(device)
    #path = '/home/work_nfs3/lvshubo/workspace/Project/senior_project/DNS-challenge/se-dccrn/exp/MSA_DCCRN_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_400_100/'#FLAGS.exp_dir - "_retrain"
    #print(path)
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_dns/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_320_160/ori/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_320_160_multiband_mmdensenet_biquad/ori/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_320_160_multiband_dplstm/ori/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_320_160_multiband_complex_dplstm/ori/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_320_160_multiband_complex_dplstm_skipcnn/ori/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_320_160_multiband_complex_dplstm_skipcnn_cprelu/sec/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2022/acas_method/exp/SDDNet_320_160_320/DNNet/ori/'#'/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2022/acas_method/exp/SDDNet_320_160_320/DNNet/ori/'
    #path = '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2022/acas_method/exp/SDDNet_320_160_320/SRNet/ori/'
    #reload_for_eval(model, '/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2022/acas_method_new/exp/SDDNet_320_160_320/SRNet_new/ori/', FLAGS.use_cuda, True)###导入之前的模型
    if not args.decode:
        train(model, FLAGS, device, writer)
    reload_for_eval(model, FLAGS.exp_dir, FLAGS.use_cuda, True)
    '''
    path = '/home/work_nfs3/lvshubo/dasan/lstm/sruc_16k_auto_noise/test_data/lst/'#'/home/work_nfs3/lvshubo/workspace/Project/senior_project/DNS-challenge/se-dccrn/test_data/'
    for snr in [-5, 0, 5, 10, 15, 20]:
        args.tt_list = path + 'test_' + str(snr) + '.lst'
        #file = 'out_noisy_' + str(snr
        decode(model, args, snr, device)
    '''
    decode(model, args, None, device)
    
    #args.tt_list = '/home/work_nfs3/lvshubo/dasan/lstm/sruc_16k_auto_noise/data/hw_project_test.lst'
    #decode(model, args, None, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser.add_argument('--decode', type=int, default=0, help='if decode')
    parser.add_argument(
        '--exp-dir',
        dest='exp_dir',
        type=str,
        default='',
        help='the exp dir')
    parser.add_argument(
        '--stage',
        dest='stage',
        type=str,
        default='DNNet',
        help='the exp dir')
    parser.add_argument(
        '--segement-length', 
        dest='segement_length', 
        type=int,
        default=8,
        help='the segement length')   
    parser.add_argument(
        '--tr-clean', 
        dest='tr_clean', 
        type=str,
        default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/data_2020/clean_tr.lst',
        help='the train clean data list')
    parser.add_argument(
        '--tr-noise',
         dest='tr_noise', 
         type=str,
         default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/data_2020/noise_tr.lst', 
         help='the train noise data list')
    parser.add_argument(
        '--cv-clean',
         dest='cv_clean', 
         type=str,
         default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/data_2020/clean_cv.lst', 
         help='the validation clean data list')
    parser.add_argument(
        '--cv-noise',
         dest='cv_noise', 
         type=str,
         default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/data_2020/noise_cv.lst', 
         help='the validation noise data list')
    parser.add_argument(
        '--tr-rir',
         dest='tr_rir', 
         type=str,
         default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/data_2020/rir_tr.lst', 
         help='the train rir data list')
    parser.add_argument(
        '--cv-rir',
         dest='cv_rir', 
         type=str,
         default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2021/dccrn_experiment/data_2020/rir_cv.lst', 
         help='the validation rir data list')
    parser.add_argument(
        '--tt-list', dest='tt_list', type=str,default='/home/work_nfs3/lvshubo/workspace/Project/senior_project/DNS-challenge/se-dccrn/exp/MSA_DCCRN_ComplexCat_Time-SiSNR_DNS_RealSpec_0.001_2_256_16k_400_100/aishell/wav.lst', help='the test data list')    
    parser.add_argument(
        '--rnn-layers',
        dest='rnn_layers',
        type=int,
        default=2,
        help='the num hidden rnn layers')
    parser.add_argument(
        '--rnn-units',
        dest='rnn_units',
        type=int,
        default=256,
        help='the num hidden rnn units')
    parser.add_argument(
        '--learn-rate',
        dest='learn_rate',
        type=float,
        default=0.000035,
        help='the learning rate in training')
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=45,
        help='the max epochs')

    parser.add_argument(
        '--dropout',
        dest='dropout',
        type=float,
        default=0.2,
        help='the probility of dropout')
    parser.add_argument(
        '--left-context',
        dest='left_context',
        type=int,
        default=0,
        help='the left context to add')
    parser.add_argument(
        '--right-context',
        dest='right_context',
        type=int,
        default=0,
        help='the right context to add')
    parser.add_argument(
        '--input-dim',
        dest='input_dim',
        type=int,
        default=257,
        help='the input dim')
    parser.add_argument(
        '--output-dim',
        dest='output_dim',
        type=int,
        default=257,
        help='the output dim')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=36,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument(
        '--seed', dest='seed', type=int, default=20, help='the random seed')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default='/home/work_nfs/lvshubo/workspace/Project/DNS-Challenge_2022/acas_method_new/exp/SDDNet_320_160_320/SRNet_new/log/',
        help='the random seed')
    parser.add_argument(
        '--num-threads', dest='num_threads', type=int, default=10)
    parser.add_argument(
        '--window-len',
        dest='win_len',
        type=int,
        default=320,
        help='the window-len in enframe')
    parser.add_argument(
        '--window-inc',
        dest='win_inc',
        type=int,
        default=160,
        help='the window include in enframe')
    parser.add_argument(
        '--fft-len',
        dest='fft_len',
        type=int,
        default=320,
        help='the fft length when in extract feature')
    parser.add_argument(
        '--window-type',
        dest='win_type',
        type=str,
        default='hanning',
        help='the window type in enframe, include hamming and None')
    parser.add_argument(
        '--kernel-size',
        dest='kernel_size',
        type=int,
        default=6,
        help='the kernel_size')
    parser.add_argument(
        '--kernel-num',
        dest='kernel_num',
        type=int,
        default=9,
        help='the kernel_num')
    parser.add_argument(
        '--num-gpu',
        dest='num_gpu',
        type=int,
        default=1,
        help='the num gpus to use')
    parser.add_argument(
        '--target-mode',
        dest='target_mode',
        type=str,
        default='RealSpec',
        help='the type of target, MSA, PSA, PSM, IBM, IRM...')
    
    parser.add_argument(
        '--weight-decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument(
        '--clip-grad-norm', dest='clip_grad_norm', type=float, default=5.)
    parser.add_argument(
        '--sample-rate', dest='sample_rate', type=str, default='16k')
    parser.add_argument('--retrain', dest='retrain', type=int, default=0)
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    print(FLAGS.win_type)
    main(FLAGS)
