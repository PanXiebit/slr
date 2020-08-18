import torch.nn.functional as F
import torch.nn as nn
import torch
from src.modules.decoder import LevenshteinTransformerDecoder
import tensorflow as tf
from src.model.leven_utils import (
    _skip, _skip_encoder_out, _fill,
    _get_ins_targets, _get_del_targets,
    _apply_ins_masks, _apply_ins_words, _apply_del_words
)
import numpy as np
import ctcdecode


class DilatedSLRNet(nn.Module):
    def __init__(self, opts, device, vocab_size, vocabulary, dilated_channels=512,
                 num_blocks=1, dilations=[1, 2, 4], dropout=0.0):
        super(DilatedSLRNet, self).__init__()
        self.opts = opts
        self.device = device
        self.vocab_size = vocab_size
        self.in_channels = self.opts.feature_dim
        self.out_channels = dilated_channels
        self.vocab = vocabulary
        self.pad = self.vocab.pad()
        self.eos = self.vocab.eos()
        self.bos = self.vocab.bos()
        self.unk = self.vocab.unk()
        self.blank_id = self.vocab.blank()

        self.num_blocks = num_blocks
        self.dilations = dilations
        self.kernel_size = 3

        self.block_list = nn.ModuleList()
        for i in range(self.num_blocks):
            self.block_list.append(DilatedBlock(self.in_channels, self.out_channels,
                                                self.kernel_size, self.dilations))
        self.out_conv = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size,
                                  padding=(self.kernel_size - 1) // 2)
        self.act_tanh = nn.Tanh()
        self.fc = nn.Linear(self.out_channels, self.vocab_size)

        self.decoder = LevenshteinTransformerDecoder(opts, vocabulary)
        ctc_decoder_vocab = [chr(x) for x in range(20000, 20000 + vocab_size)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(ctc_decoder_vocab, beam_width=opts.beam_width,
                                               blank_id=self.blank_id, num_processes=10)


    def forward(self, video, len_video):
        out = 0
        for block in self.block_list:
            out += block(video)
        out = self.act_tanh(self.out_conv(out))
        logits = out.permute(0, 2, 1)
        logits = self.fc(logits)
        return logits


    def forward_decoder(self, video, len_video, tgt_tokens, len_tgt):

        bs = video.size(0)
        ctc_logits, encoder_out, prev_output_tokens = self.get_ctc_outputs_2(video, len_video)

#         # print("prev_output_tokens: ", prev_output_tokens.shape, prev_output_tokens[:2])
#         # print("tgt_tokens: ", tgt_tokens.shape, tgt_tokens[:2])

        # TODO, deletion.
        prev_output_tokens = prev_output_tokens.detach_()
        word_del_targets = _get_del_targets(prev_output_tokens, tgt_tokens, self.vocab.pad())
        # in word_del_targets, the index of element to delete is 1,
        #                      the index of element to preserve is 0.
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)
        word_del_masks = prev_output_tokens.ne(self.vocab.pad())

        # # print("==================== Delete ========================")
        # # print(word_del_targets.shape, word_del_out.shape, word_del_masks.shape)
        # exit()

        # TODO, prepare data by deleting for insert-predict
        word_del_targets_2 = word_del_targets.masked_fill(~word_del_masks, 1)
        target_score, target_rank = word_del_targets_2.sort(1)

        target_len = torch.LongTensor(bs * [prev_output_tokens.size(1)]).type_as(word_del_targets_2) \
                     - torch.sum(word_del_targets_2, dim=-1)  # 保留下来的tokens个数
        bs, new_len = word_del_targets_2.size()  # word_del_targets 的第二个维度等于prev_output_tokens和tgt_tokens中的较大者
        target_cutoff = torch.cat(bs * [torch.arange(new_len).unsqueeze_(0)], dim=0).type_as(target_len)
        target_cutoff = target_cutoff < target_len.unsqueeze(1)
        target_rank_2 = target_rank.masked_fill(~target_cutoff, 1000)
        target_rank_2, _ = target_rank_2.sort(1, descending=False)
        target_rank_2 = target_rank_2.masked_fill_(~target_cutoff, 0)
        prev_output_tokens = prev_output_tokens.gather(1, target_rank_2).masked_fill_(~target_cutoff, 0)
        
        # # print("=== After delete ======")
        # # print("prev_output_tokens: ", prev_output_tokens.shape, prev_output_tokens[:2])
        # # print("tgt_tokens: ", tgt_tokens.shape, tgt_tokens[:2])


        # TODO insert mask and predict after delete
        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.vocab.pad(), self.vocab.unk())
        # # print(masked_tgt_masks, masked_tgt_tokens, mask_ins_targets)

        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.vocab.pad())  # insert 是插入在两个值中间

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out
        )
        # # print("======================== Insert Mask =============================")
        # # print("mask_ins: ", mask_ins_out.shape, mask_ins_targets.shape, mask_ins_masks.shape)

        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out
        )
        # # print("======================== Insert Word =====================================")
        # # print("word_ins: ", word_ins_out.shape, tgt_tokens.shape, masked_tgt_masks.shape)


        return ctc_logits, {
            "mask_ins": {
                "out": mask_ins_out, "tgt": mask_ins_targets,
                "mask": mask_ins_masks, "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": masked_tgt_masks, "ls": self.opts.label_smoothing,
                "nll_loss": True
            },
            "word_del": {
                "out": word_del_out, "tgt": word_del_targets,
                "mask": word_del_masks
            }
        }

    def get_ctc_outputs(self, video, len_video):
        out = 0
        for block in self.block_list:
            out += block(video)
        out = self.act_tanh(self.out_conv(out))

        # ctc logits
        encoder_out = out.permute(0, 2, 1)  # [batch, video_len, hid_size]

        ctc_logits = self.fc(encoder_out)

        logits = tf.transpose(tf.constant(ctc_logits.detach().cpu().numpy()), [1, 0, 2])  # [len, batch, vocab_size]
        len_video = tf.constant(len_video.cpu().numpy(), dtype=tf.int32)
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, len_video, beam_width=6, top_paths=1)
        pred_seq = tf.sparse.to_dense(decoded[0]).numpy()  # # print(pred_seq.shape, decoded[0].dense_shape)
        if pred_seq.shape[1] == 0:
            pred_seq = np.zeros((pred_seq.shape[0], 1), dtype=np.int32)

        pred_seq = torch.LongTensor(pred_seq).detach_()  # [batch, out_len]
        pred_seq = pred_seq.to(self.device)
        prev_seq_len = pred_seq.ne(0).sum(-1)
        # # print("pred_seq: ", pred_seq.shape, pred_seq[:2])

        bs = pred_seq.size(0)
        max_prev_len = torch.max(prev_seq_len) + 2
        prev_output_tokens = torch.zeros((bs, max_prev_len)).long().to(pred_seq.device)

        # TODO prepare data for deletion.
        for i in range(bs):
            prev_output_tokens[i, 0] = self.vocab.bos()
            prev_output_tokens[i, 1:prev_seq_len[i] + 1] = pred_seq[i][:prev_seq_len[i]]
            prev_output_tokens[i, prev_seq_len[i] + 1] = self.vocab.eos()
        return ctc_logits, encoder_out, prev_output_tokens

    def get_ctc_outputs_2(self, video, len_video):
        out = 0
        for block in self.block_list:
            out += block(video)
        out = self.act_tanh(self.out_conv(out))

        # ctc logits
        encoder_out = out.permute(0, 2, 1)  # [batch, video_len, hid_size]

        ctc_logits = self.fc(encoder_out)

        # logits = tf.transpose(tf.constant(ctc_logits.detach().cpu().numpy()), [1, 0, 2])  # [len, batch, vocab_size]
        # len_video = tf.constant(len_video.cpu().numpy(), dtype=tf.int32)
        # decoded, _ = tf.nn.ctc_beam_search_decoder(logits, len_video, beam_width=6, top_paths=1)
        # pred_seq = tf.sparse.to_dense(decoded[0]).numpy()  # # print(pred_seq.shape, decoded[0].dense_shape)
        # if pred_seq.shape[1] == 0:
        #     pred_seq = np.zeros((pred_seq.shape[0], 1), dtype=np.int32)
        pred_seq, _, _, out_seq_len = self.ctc_decoder.decode(ctc_logits, len_video)
        # [bs, beam, out_len]. [bs, beam]

        pred_seq = pred_seq[:, 0, :].to(self.device)  # [bs, out_len]
        prev_seq_len = out_seq_len[:, 0]
        # print("pred_seq: ", pred_seq.shape, pred_seq[:2])

        bs = pred_seq.size(0)
        max_prev_len = torch.max(prev_seq_len) + 2
        prev_output_tokens = torch.zeros((bs, max_prev_len)).long().to(pred_seq.device)

        # TODO prepare data for deletion.
        for i in range(bs):
            prev_output_tokens[i, 0] = self.vocab.bos()
            prev_output_tokens[i, 1:prev_seq_len[i] + 1] = pred_seq[i][:prev_seq_len[i]]
            prev_output_tokens[i, prev_seq_len[i] + 1] = self.vocab.eos()
        prev_output_tokens = prev_output_tokens.detach_()
        # print("prev_output_tokens: ", prev_output_tokens)

        return ctc_logits, encoder_out, prev_output_tokens


    def infer_decoder(self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            # max_lens = torch.zeros_like(output_tokens).fill_(255)
            max_lens = torch.zeros(output_tokens.size(0)).fill_(255).type_as(output_tokens)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # print("max_lens: ", max_lens)

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        # print("can_del_word: ",can_del_word)

        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(self.reorder_encoder_out, encoder_out, can_del_word)
            )
            # print("word_del_score: ", word_del_score.shape)
            word_del_pred = word_del_score.max(-1)[1].to(torch.uint8)

#             print("word_del_pred: ", word_del_pred)

            word_del_attn = None
            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word] if output_scores is not None else None,
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # print("After delte: ", output_tokens, output_tokens.shape, max_lens.shape)
        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens

        # print("can_ins_mask: ", can_ins_mask)

        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.reorder_encoder_out, encoder_out, can_ins_mask)
            )
            # print("mask_ins_score: ", mask_ins_score.shape)

            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask] if output_scores is not None else None,
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # print("after insert mask: ", output_tokens, output_tokens.shape)

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0


        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.reorder_encoder_out, encoder_out, can_ins_word)
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word] if output_scores is not None else None,
                word_ins_pred,
                word_ins_score,
                self.unk,
            )
            word_ins_attn = None
            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # print("After inert word: ", output_tokens)

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off] if output_scores is not None else None
        attn = None if attn is None else attn[:, :cut_off, :]

        # print("output_tokens: ", output_tokens)

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )

    # @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out.index_select(1, new_order)


class DilatedCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCell, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        stride = 1
        # padding = ((kernel_size-1)*(stride-1) + dilation*(kernel_size-1)) // 2
        padding = (kernel_size - 1) * dilation // 2
        self.in_conv = nn.Conv1d(self.in_channels, self.out_channels,
                                 self.kernel_size, dilation=self.dilation, padding=padding)
        self.mid_conv = nn.Conv1d(self.out_channels, self.out_channels,
                                 self.kernel_size, padding=(self.kernel_size-1)//2)
        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        :param x: [B, C, L]
        :return:
        '''
        res = x
        x = self.in_conv(x)
        x = self.gate_tanh(x) * self.gate_sigmoid(x)
        o = self.gate_tanh(self.mid_conv(x))
        h = o + res
        return o, h

class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_list):
        super(DilatedBlock, self).__init__()
        self.num_cells = len(dilation_list)
        self.dilated_cells = nn.ModuleList()
        self.dilated_cells.append(DilatedCell(in_channels, out_channels, kernel_size, dilation_list[0]))
        for dilation in dilation_list[1:]:
            self.dilated_cells.append(DilatedCell(out_channels, out_channels, kernel_size, dilation))

    def forward(self, x):
        block_o = 0
        for cell in self.dilated_cells:
            o, x = cell(x)
            block_o += o
        return block_o




