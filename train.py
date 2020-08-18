import torch
from config.options import parse_args
from src.data.video_lang_datasets import PhoenixVideo
from utils import init_logging, LossManager, ModelManager
import os
# from src.model.dilated_slr import DilatedSLRNet
from src.criterion.ctc_loss import CtcLoss
from src.model.full_conv import MainStream
from src.trainer import Trainer
import logging
import numpy as np
import uuid
from metrics.wer import get_phoenix_wer
from tqdm import tqdm
from src.data.vocabulary import Vocabulary
from src.iterative_generate import IterativeGenerate


        

def main():
    opts = parse_args()
    init_logging(os.path.join(opts.log_dir, '{:s}_log.txt'.format(opts.task)))

    if torch.cuda.is_available():
        torch.cuda.set_device(opts.gpu)
        logging.info("Using GPU!")
        device = "cuda"
    else:
        logging.info("Using CPU!")
        device = "cpu"
        
    logging.info(opts)

    train_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase="train", DEBUG=opts.DEBUG)
    valid_datasets = PhoenixVideo(opts.vocab_file, opts.corpus_dir, opts.video_path, phase="dev", DEBUG=opts.DEBUG)
    vocab_size=  valid_datasets.vocab.num_words
    blank_id = valid_datasets.vocab.word2index['<BLANK>']
    vocabulary = Vocabulary(opts.vocab_file)
    #model = DilatedSLRNet(opts, device, vocab_size, vocabulary,
    #                      dilated_channels=512, num_blocks=5, dilations=[1, 2, 4], dropout=0.0)
    model = MainStream(vocab_size)
    criterion = CtcLoss(opts, blank_id, device, reduction="none")
    
    # print(model)


    # Build trainer
    trainer = Trainer(opts, model, criterion, vocabulary, vocab_size, blank_id)

    if os.path.exists(opts.check_point):
        logging.info("Loading checkpoint file from {}".format(opts.check_point))
        epoch, num_updates, loss = trainer.load_checkpoint(opts.check_point)
    else:
        logging.info("No checkpoint file in found in {}".format(opts.check_point))
        epoch, num_updates, loss = 0, 0, 0.0

    trainer.set_num_updates(num_updates)
    model_manager = ModelManager(max_num_models=5)
    while epoch < opts.max_epoch and trainer.get_num_updates() < opts.max_updates:
        epoch += 1
        trainer.adjust_learning_rate(epoch)
        #trainer.dynamic_freeze_layers(epoch)
        loss = train(opts, train_datasets, valid_datasets, trainer, epoch, num_updates, loss)

        #if num_updates % opts.save_interval_updates == 0:
        if epoch <= opts.stage_epoch * 2:
            phoenix_eval_err = eval(opts, valid_datasets, trainer, epoch)
            phoenix_eval_err = eval_tf(opts, valid_datasets, trainer, epoch)
        else:
            phoenix_eval_err = eval(opts, valid_datasets, trainer, epoch)
            phoenix_eval_err = eval_dec(opts, valid_datasets, trainer, epoch)


        save_ckpt = os.path.join(opts.log_dir, 'ep{:d}_{:.4f}.pkl'.format(epoch, phoenix_eval_err[0]))
        trainer.save_checkpoint(save_ckpt, epoch, num_updates, loss)
        model_manager.update(save_ckpt, phoenix_eval_err, epoch)


def train(opts, train_datasets, valid_datasets, trainer, epoch, num_updates, last_loss):
    train_iter = trainer.get_batch_iterator(train_datasets, batch_size=opts.batch_size, shuffle=True)
    ctc_epoch_loss, dec_epoch_loss = [], []
    ctc_loss_manager = LossManager(print_step=opts.print_step, last_loss=last_loss)
    dec_loss_manager = LossManager(print_step=opts.print_step, last_loss=last_loss)
    for samples in train_iter:
        # print(samples.keys())
        # print(samples["data"].shape, samples["data"][0], samples["label"].shape)
        # print(samples["len_data"], torch.sum(samples["len_label"]))
        # exit()
        if epoch <= opts.stage_epoch:
            loss, num_updates = trainer.train_step(samples)
            ctc_loss = loss.item()
            ctc_loss_manager.update(ctc_loss, epoch, num_updates)
            ctc_epoch_loss.append(ctc_loss)
        else:
            loss, num_updates = trainer.train_decoder_step(samples)
            dec_loss = loss.item()
            dec_loss_manager.update(dec_loss, epoch, num_updates)
            dec_epoch_loss.append(dec_loss)

        # if num_updates % opts.save_interval_updates == 0:
        #     phoenix_eval_err = eval(opts, valid_datasets, trainer, epoch)
    if epoch <= opts.stage_epoch:
        logging.info("--------------------- ctc training ------------------------")
        logging.info('Epoch: {:d}, ctc loss: {:.3f} -> {:.3f}'.format(epoch, last_loss, np.mean(ctc_epoch_loss)))
        last_loss = np.mean(ctc_epoch_loss)
    else:
        logging.info("--------------------- Jointly training ------------------------")
        logging.info('Epoch: {:d}, dec loss: {:.3f} -> {:.3f}'.format(epoch, last_loss, np.mean(dec_epoch_loss)))
        last_loss = np.mean(dec_epoch_loss)
    return last_loss


def eval(opts, valid_datasets, trainer, epoch):
    eval_iter = trainer.get_batch_iterator(valid_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for samples in tqdm(eval_iter):
        err, correct, count = trainer.valid_step(samples, decoded_dict)
        val_err += err
        val_correct += correct
        val_count += count
    logging.info('-' * 50)
    logging.info('Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

    # ------ Evaluation with official script (merge synonyms) --------
    list_str_for_test = []
    for k, v in decoded_dict.items():
        start_time = 0
        for wi in v:
            tl = np.random.random() * 0.1
            list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                       valid_datasets.vocab.index2word[wi]))
            start_time += tl
    tmp_prefix = str(uuid.uuid1())
    txt_file = '{:s}.txt'.format(tmp_prefix)
    result_file = os.path.join('evaluation_relaxation', txt_file)
    with open(result_file, 'w') as fid:
        fid.writelines(list_str_for_test)
    phoenix_eval_err = get_phoenix_wer(txt_file, 'dev', tmp_prefix)
    logging.info('[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
    return phoenix_eval_err


def eval_tf(opts, valid_datasets, trainer, epoch):
    eval_iter = trainer.get_batch_iterator(valid_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for samples in tqdm(eval_iter):
        err, correct, count = trainer.valid_step_tf(samples, decoded_dict)
        val_err += err
        val_correct += correct
        val_count += count
    logging.info('-' * 50)
    logging.info('Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

    # ------ Evaluation with official script (merge synonyms) --------
    list_str_for_test = []
    for k, v in decoded_dict.items():
        start_time = 0
        for wi in v:
            tl = np.random.random() * 0.1
            list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                       valid_datasets.vocab.index2word[wi]))
            start_time += tl
    tmp_prefix = str(uuid.uuid1())
    txt_file = '{:s}.txt'.format(tmp_prefix)
    result_file = os.path.join('evaluation_relaxation', txt_file)
    with open(result_file, 'w') as fid:
        fid.writelines(list_str_for_test)
    phoenix_eval_err = get_phoenix_wer(txt_file, 'dev', tmp_prefix)
    logging.info('[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
    return phoenix_eval_err


def eval_dec(opts, valid_datasets, trainer, epoch):
    eval_iter = trainer.get_batch_iterator(valid_datasets, batch_size=opts.batch_size, shuffle=False)
    decoded_dict = {}
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for samples in tqdm(eval_iter):
        err, correct, count = trainer.valid_decoder_step(samples, decoded_dict)
        val_err += err
        val_correct += correct
        val_count += count
    logging.info('-' * 50)
    logging.info('Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

    # ------ Evaluation with official script (merge synonyms) --------
    list_str_for_test = []
    for k, v in decoded_dict.items():
        start_time = 0
        for wi in v:
            tl = np.random.random() * 0.1
            list_str_for_test.append('{} 1 {:.3f} {:.3f} {}\n'.format(k, start_time, start_time + tl,
                                                                       valid_datasets.vocab.index2word[wi]))
            start_time += tl
    tmp_prefix = str(uuid.uuid1())
    txt_file = '{:s}.txt'.format(tmp_prefix)
    result_file = os.path.join('evaluation_relaxation', txt_file)
    with open(result_file, 'w') as fid:
        fid.writelines(list_str_for_test)
    phoenix_eval_err = get_phoenix_wer(txt_file, 'dev', tmp_prefix)
    logging.info('[Relaxation Evaluation] Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        phoenix_eval_err[0], phoenix_eval_err[1], phoenix_eval_err[2], phoenix_eval_err[3]))
    return phoenix_eval_err



if __name__ == "__main__":
    main()