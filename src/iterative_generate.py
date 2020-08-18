from collections import namedtuple
import torch
import numpy as np


DecoderOut = namedtuple('DecoderOut', [
    'output_tokens',
    'output_scores',
    'attn',
    'step',
    'max_step',
    'history'
])


class IterativeGenerate(object):
    def __init__(self,
                 vocabulary,
                 model,
                 eos_penalty=0.0,
                 max_iter=10,
                 max_ratio=None,
                 beam_size=1,
                 decoding_format=None,
                 retain_dropout=False,
                 adaptive=False,
                 retain_history=False,
                 reranking=False,
                 ):
        self.bos = vocabulary.bos()
        self.pad = vocabulary.pad()
        self.unk = vocabulary.unk()
        self.eos = vocabulary.eos()
        self.vocab_size = vocabulary.vocab_size()
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.beam_size = beam_size
        self.reranking = reranking
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.retain_history = retain_history
        self.adaptive = adaptive
        self.model = model

    @torch.no_grad()
    def generate(self, video, len_video, prefix_tokens=None):
        _, encoder_out, prev_output_tokens = self.model.get_ctc_outputs(video, len_video)

        # print("prev_output_tokens: ", prev_output_tokens)

        return self.itertive_decoder(encoder_out, prev_output_tokens)

    @torch.no_grad()
    def generate_ctcdecode(self, video, len_video, prefix_tokens=None):
        _, encoder_out, prev_output_tokens = self.model.get_ctc_outputs_2(video, len_video)

        # print("prev_output_tokens: ", prev_output_tokens)
        return self.itertive_decoder(encoder_out, prev_output_tokens)


    def itertive_decoder(self, encoder_out, prev_output_tokens):
        prev_decoder_out = DecoderOut(
            output_tokens=prev_output_tokens,
            output_scores=None,
            attn=None,
            step=0,
            max_step=0,
            history=None)

        bsz = prev_output_tokens.size(0)
        sent_idxs = torch.arange(bsz)

        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        finalized = [[] for _ in range(bsz)]

        # x, y, s, a
        # prev_output_tokens, decoder_out.output_tokens, decoder_out.output_scores, decoder_out.attn
        def is_a_loop(x, y, s, a):
            """ 判断x,y是否一致，这里要先去除到padding长度不一致带来的影响"""
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }

        for step in range(self.max_iter + 1):
            decoder_options = {
                "eos_penalty": self.eos_penalty,
                "max_ratio": self.max_ratio,
                "decoding_format": self.decoding_format,
            }
            prev_decoder_out = prev_decoder_out._replace(
                step=step,
                max_step=self.max_iter + 1,
            )

            decoder_out = self.model.infer_decoder(
                prev_decoder_out, encoder_out, **decoder_options
            )

            if self.adaptive:
                # terminate if there is a loop
                terminated, out_tokens, out_scores, out_attn = is_a_loop(
                    prev_output_tokens, decoder_out.output_tokens, decoder_out.output_scores, decoder_out.attn
                )
                decoder_out = decoder_out._replace(
                    output_tokens=out_tokens,
                    output_scores=out_scores,
                    attn=out_attn,
                )

            else:
                terminated = decoder_out.output_tokens.new_zeros(decoder_out.output_tokens.size(0)).to(torch.uint8)

            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences 收集已经结束了的sentence index
            finalized_idxs = sent_idxs[terminated]
            # print("finalized_idxs: ", finalized_idxs)
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated] if decoder_out.output_scores is not None else None
            finalized_attn = (
                None if (decoder_out.attn is None or decoder_out.attn.size(0) == 0) else decoder_out.attn[terminated]
            )

            if self.retain_history:
                finalized_history_tokens = [h[terminated] for h in decoder_out.history]

            # finalized 包含了每个sentence的最终结果
            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i] if finalized_scores is not None else None,
                        None if finalized_attn is None else finalized_attn[i],
                    )
                ]

                if self.retain_history:
                    finalized[finalized_idxs[i]][0]['history'] = []
                    for j in range(len(finalized_history_tokens)):
                        finalized[finalized_idxs[i]][0]['history'].append(
                            finalized_hypos(
                                step,
                                finalized_history_tokens[j][i],
                                None, None
                            )
                        )

            # check if all terminated
            if terminated.sum() == terminated.size(0):
                break

            # for next step
            not_terminated = ~terminated  # 只处理未结束的sentence
            prev_decoder_out = decoder_out._replace(
                output_tokens=decoder_out.output_tokens[not_terminated],
                output_scores=decoder_out.output_scores[
                    not_terminated] if decoder_out.output_scores is not None else None,
                attn=decoder_out.attn[not_terminated]
                if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
                else None,
                history=[h[not_terminated] for h in decoder_out.history]
                if decoder_out.history is not None
                else None,
            )
            encoder_out = self.model.reorder_encoder_out(encoder_out, not_terminated.nonzero().squeeze())
            sent_idxs = sent_idxs[not_terminated]
            prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.beam_size > 1:
            # aggregate information from length beam
            finalized = [
                finalized[np.argmax(
                    [finalized[self.beam_size * i + j][0]['score'] for j in range(self.beam_size)]
                ) + self.beam_size * i] for i in range(len(finalized) // self.beam_size)
            ]
        # print(finalized[4][0]["tokens"])
        # print(len(finalized), finalized[0][0].keys())
        # exit()
        return finalized



