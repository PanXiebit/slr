

from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch import Tensor


class CtcLoss(_Loss):
    def __init__(self, opts, blank_id, device, reduction="none"):
        super(CtcLoss, self).__init__()
        self.ctcloss = nn.CTCLoss(blank=blank_id, reduction=reduction)
        self.device = device
        self.ctc_weight = opts.ctc_weight
        self.dec_weight = opts.dec_weight

    def forward(self, model, samples):
        video = samples["data"]
        len_video = samples["len_data"]
        label = samples["label"]            # "(sum(target_lengths))"
        len_label = samples["len_label"]
        logits, len_video = model(video, len_video)
        logits = logits.permute(1, 0, 2)
        log_probs = logits.log_softmax(-1)   # T x N x C
        loss = self.ctcloss(log_probs.cpu(), label.cpu(), len_video.cpu(), len_label.cpu())
        loss = loss.mean()
        return loss.to(self.device)

    def forward_decoder(self, model, samples, reduce=True):
        video = samples["data"]
        len_video = samples["len_data"]
        label = samples["label"]            # "(sum(target_lengths))"
        len_label = samples["len_label"]
        decoder_label = samples["decoder_label"]
        len_decoder_label = samples["len_decoder_label"]
        ctc_logits, outputs = model.forward_decoder(video, len_video, decoder_label, len_decoder_label)

        ctc_logits = ctc_logits.permute(1, 0, 2)
        log_probs = ctc_logits.log_softmax(-1)  # T x N x C
        ctc_loss = self.ctcloss(log_probs.cpu(), label.cpu(), len_video.cpu(), len_label.cpu())
        ctc_loss = ctc_loss.mean().to(self.device)

        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)
        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                get_item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )
        loss = self.ctc_weight * ctc_loss + self.dec_weight * loss
        return loss, sample_size, logging_output

    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len
            policy_logprob: if there is some policy
                depends on the likelihood score as rewards.
        """


        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            # print(logits.shape, targets.shape)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                        1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

        loss = loss * factor
        # print("name: {}, loss: {}".format(name, loss))
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

def get_item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor