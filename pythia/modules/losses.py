# Copyright (c) Facebook, Inc. and its affiliates.
"""
Losses module contains implementations for various losses used generally
in vision and language space. One can register custom losses to be detected by
pythia using the following example.

.. code::

   from pythia.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_attributes:
       some_model:
           losses:
               - type: custom
               - params: {}
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils.rnn import pack_padded_sequence

from pythia.common.registry import registry
from pythia.utils.myUtils import create_batch_labels,create_batch_labels_diag

class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_attributes`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (List[ConfigNode]): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instanttions of each loss
                                   passed in config
    """

    def __init__(self, loss_list):
        super().__init__()
        self.losses = []
        tp = registry.get("config").training_parameters
        self._evalai_inference = tp.evalai_inference
        for loss in loss_list:
            self.losses.append(PythiaLoss(loss))

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if not hasattr(sample_list, "targets"):
            if not self._evalai_inference:
                warnings.warn(
                    "Sample list has not field 'targets', are you "
                    "sure that your ImDB has labels? you may have "
                    "wanted to run with --evalai_inference 1"
                )
            return output

        for loss in self.losses:
            output.update(loss(sample_list, model_output, *args, **kwargs))

        registry_loss_key = "{}.{}.{}".format(
            "losses", sample_list.dataset_name, sample_list.dataset_type
        )
        # Register the losses to registry
        registry.register(registry_loss_key, output)

        return output


class PythiaLoss(nn.Module):
    """Internal Pythia helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``PythiaLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params={}):
        super().__init__()
        self.writer = registry.get("writer")
        if "type" not in params:
            raise ValueError(
                "Parameters to loss must have 'type' field to"
                "specify type of loss to instantiate"
            )

        loss_name = params["type"]
        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        if loss_class is None:
            raise ValueError(
                "No loss named {} is registered to registry".format(loss_name)
            )
        # Special case of multi as it requires an array
        if loss_name == "multi":
            self.loss_criterion = loss_class(params)
        else:
            loss_params = params.get("params", {})
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = self.loss_criterion(sample_list, model_output, *args, **kwargs)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        key = "{}/{}/{}".format(
            sample_list.dataset_type, sample_list.dataset_name, self.name
        )

        return {key: loss}


@registry.register_loss("logit_bce")
class LogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("caption_cross_entropy")
class CaptionCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, "caption_len"):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= "1.1":
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = F.cross_entropy(scores, targets)

        return loss


@registry.register_loss("nll_loss")
class NLLLoss(nn.Module):
    """Negative log likelikehood loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the negative log likelihood.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        _, idx = targets.max(dim=1)
        loss = F.nll_loss(scores, idx, reduction="mean")

        return loss * targets.size(1)


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


@registry.register_loss("multi")
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        params (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: logit_bce
            weight: 0.3
            params: {}
          - type: attention_supervision
            weight: 0.7
            params: {}

    """

    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []
        self.writer = registry.get("writer")

        self.loss_names = []

        for loss_params in params["params"]:
            self.loss_names.append(loss_params["type"])
            loss_fn = PythiaLoss(loss_params)
            loss_weight = loss_params.get("weight", {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            loss += self.losses_weights[idx] * value
        return loss


@registry.register_loss("attention_supervision")
class AttentionSupervisionLoss(nn.Module):
    """Loss for attention supervision. Used in case you want to make attentions
    similar to some particular values.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = lambda *args, **kwargs: nn.functional.binary_cross_entropy(
            *args, **kwargs
        )

    def forward(self, sample_list, model_output):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        context_attentions = model_output["attentions"]
        attention_supervision = sample_list["info"]["attention_supervision"]

        loss = self.loss_fn(
            context_attentions[0],
            attention_supervision.float(),
            weight=attention_supervision.float(),
        )

        # Multiply average loss back with target size to get actual loss
        return loss * attention_supervision.size(1)


@registry.register_loss("weighted_softmax")
class WeightedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("softmax_kldiv")
class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("wrong")
class WrongLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, reduction="mean")
        loss *= target_score.size(1)
        return loss


@registry.register_loss("bce_kl_combined")
class CombinedLoss(nn.Module):
    def __init__(self, weight_softmax):
        super().__init__()
        self.weight_softmax = weight_softmax

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(
            pred_score, target_score, reduction="mean"
        )
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        return loss


@registry.register_loss("m4c_decoding_bce_with_mask")
class M4CDecodingBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss_mask = sample_list["train_loss_mask"]
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(
            scores, targets, reduction="none"
        )
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss

@registry.register_loss("m4c_decoding_bce_with_mask_contrastive")
class M4CDecodingBCEWithMaskContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss_mask = sample_list["train_loss_mask"]
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(
            scores, targets, reduction="none"
        )
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        if "contrastive_scores" in model_output:
            logits_per_image = model_output['contrastive_scores']
            logits_per_text = logits_per_image.t()
            if "cons_labels" in sample_list:
                # print(str(sample_list["cons_labels"]))
                labels = create_batch_labels(sample_list["cons_labels"])
                # print(str(labels))
                labels = labels.float()
                crition1 = torch.nn.BCEWithLogitsLoss()
                loss_i = crition1(logits_per_image, labels)
                loss_t = crition1(logits_per_text, labels)
            else:
                labels = torch.arange(0, logits_per_image.size(0)).to(device=logits_per_image.device)
                loss_i = F.cross_entropy(logits_per_image, labels)
                loss_t = F.cross_entropy(logits_per_text, labels)
            contrastive_loss = (loss_i + loss_t)/2
            #print(contrastive_loss)
            loss = loss + contrastive_loss
        return loss

@registry.register_loss("pretrainonly_m4c_decoding_bce_with_mask")
class pretrainonlyM4CDecodingBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])
        self.pretrainloss = PretrainLoss()
        #self.multiloss =Mutil2CrossEntropyLoss()
        self.softloss = SoftCrossEntropy()
        self.fuse_obj_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_obj_weight.data.fill_(0.5)
        #self.fuse_ocr_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.fuse_ocr_weight.data.fill_(0.5)
        self.fuse_ocr_weight = 0.5
        self.fuse_vis_weight  = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_vis_weight.data.fill_(0.5)

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        #print(scores.size())
        #print(targets.size())
        loss_mask = sample_list["train_loss_mask"]
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(
            scores, targets, reduction="none"
        )
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        if 'textcls_scores' in model_output:
            ## temporal solution: pretrain loss only
            if "term_scores" in model_output:
                #print(sample_list['adv_input_labels'])
                char_loss_mask = (sample_list['adv_input_labels'] != -1).float()
                term_vocab_size=30001 #30001
                adv_term_losses = F.cross_entropy(model_output["term_scores"].view(-1, term_vocab_size), sample_list['adv_input_labels'].view(-1),reduction="none", ignore_index=-1)
                ori_loss = self.pretrainloss(sample_list, model_output)
                adv_term_losses *= char_loss_mask.view(-1)
                count = torch.max(torch.sum(char_loss_mask), self.one.to(adv_term_losses.device))
                adv_term_loss = torch.sum(adv_term_losses) / count
                print(adv_term_loss)
                loss = loss*0.0 + ori_loss + adv_term_loss
            elif "contrastive_scores" in model_output:
                logits_per_image = model_output['contrastive_scores']
                logits_per_text = logits_per_image.t()
                if "o2r_labels" in sample_list:
                    #print(str(sample_list["cons_labels"]))
                    o2r_labels = create_batch_labels(sample_list["o2r_labels"])
                    r2o_labels = create_batch_labels(sample_list["r2o_labels"])
                    #print(str(labels))
                    mask = o2r_labels != -1
                    o2r_labels = o2r_labels.float()
                    r2o_labels = r2o_labels.float()
                    loss_i = F.binary_cross_entropy_with_logits(
                            logits_per_image[mask], o2r_labels[mask], reduction="mean"
                    )
                    loss_t = F.binary_cross_entropy_with_logits(
                        logits_per_text[mask], r2o_labels[mask], reduction="mean"
                    )
                    #loss_i = self.softloss(
                    #        logits_per_image[mask], o2r_labels[mask],o2r_labels.size(0)
                    #)
                    #loss_t =self.softloss(
                    #    logits_per_text[mask], r2o_labels[mask],o2r_labels.size(0)
                    #)
                    if "text_contrastive_scores" in model_output:
                        logits_per_image_2 = model_output['text_contrastive_scores']
                        logits_per_text_2 = logits_per_image_2.t()
                        loss_i_2 = F.binary_cross_entropy_with_logits(
                            logits_per_image_2[mask], o2r_labels[mask], reduction="mean"
                        )
                        loss_t_2 = F.binary_cross_entropy_with_logits(
                            logits_per_text_2[mask], r2o_labels[mask], reduction="mean"
                        )
                        loss_i = (loss_i + loss_i_2) / 2
                        loss_t = (loss_t + loss_t_2) / 2
                        #loss_i = loss_i_2
                        #loss_t = loss_t_2
                    if "ocr_contrastive_scores" in model_output:
                        logits_per_image_2 = model_output['ocr_contrastive_scores']
                        logits_per_text_2 = logits_per_image_2.t()
                        loss_i_2 = F.binary_cross_entropy_with_logits(
                            logits_per_image_2[mask], o2r_labels[mask], reduction="mean"
                        )
                        loss_t_2 = F.binary_cross_entropy_with_logits(
                            logits_per_text_2[mask], r2o_labels[mask], reduction="mean"
                        )
                        print(" with OCR feat except bbox")
                        #print(self.fuse_ocr_weight)
                        loss_i = loss_i + loss_i_2*self.fuse_ocr_weight #.to(loss_i_2.device)
                        loss_t = loss_t + loss_t_2*self.fuse_ocr_weight #.to(loss_i_2.device)
                    if "obj_contrastive_scores" in model_output:
                        #print(sample_list.keys())
                        #obj_labels = create_batch_labels_diag(sample_list["obj_labels"])
                        obj_labels = sample_list["obj_labels"]
                        logits_obj = model_output['obj_contrastive_scores']
                        mask = obj_labels != -1
                        loss_obj = F.binary_cross_entropy_with_logits(
                            logits_obj[mask], obj_labels[mask], reduction="mean"
                        )
                        #if "vis_contrastive_scores" in model_output:
                        #    vis_labels = create_batch_labels_diag(sample_list["vis_labels"])
                        #    logits_per_image_2 = model_output['vis_contrastive_scores']
                        #    logits_per_text_2 = logits_per_image_2.t()
                        #    mask = vis_labels != -1
                        #    loss_i_2 = F.binary_cross_entropy_with_logits(
                        #        logits_per_image_2[mask], vis_labels[mask], reduction="mean"
                        #    )
                        #    loss_t_2 = F.binary_cross_entropy_with_logits(
                        #        logits_per_text_2[mask], vis_labels[mask], reduction="mean"
                        #    )
                        #    print(" with obj and OCR vis")
                        #    print(self.fuse_weight)
                        #    loss_i = loss_i + (loss_i_2+loss_obj)*self.fuse_weight.to(loss_obj.device)
                        #    loss_t = loss_t + (loss_t_2+loss_obj)*self.fuse_weight.to(loss_obj.device)
                        #else:
                        print(" with obj vis loss")
                        print(self.fuse_obj_weight)
                        loss_i = loss_i + loss_obj*self.fuse_obj_weight.to(loss_obj.device)
                        loss_t = loss_t + loss_obj*self.fuse_obj_weight.to(loss_obj.device)
                        # loss_i = loss_obj
                        # loss_t = loss_obj
                    if "vis_contrastive_scores" in model_output: #and "obj_contrastive_scores" not in model_output:
                        # print(sample_list.keys())
                        vis_labels = create_batch_labels_diag(sample_list["vis_labels"])
                        logits_per_image_2 = model_output['vis_contrastive_scores']
                        logits_per_text_2 = logits_per_image_2.t()
                        mask = vis_labels != -1
                        loss_i_vis = F.binary_cross_entropy_with_logits(
                            logits_per_image_2[mask], vis_labels[mask], reduction="mean"
                        )
                        loss_t_vis = F.binary_cross_entropy_with_logits(
                            logits_per_text_2[mask], vis_labels[mask], reduction="mean"
                        )
                        loss_i = loss_i + loss_i_vis * self.fuse_vis_weight.to(loss_i_vis.device)
                        loss_t = loss_t + loss_t_vis * self.fuse_vis_weight.to(loss_i_vis.device)
                        print("with ocr vis loss")
                else:
                    print("use old diagonal labels")
                    labels = torch.arange(0, logits_per_image.size(0)).to(device=logits_per_image.device)
                    loss_i = F.cross_entropy(logits_per_image, labels)
                    loss_t = F.cross_entropy(logits_per_text, labels)
                contrastive_loss = (loss_i + loss_t)/2
                #print(contrastive_loss)
                ori_loss = self.pretrainloss(sample_list, model_output)
                if "deep_predict_scores" in model_output:
                    PoN_labels = sample_list["PoN_labels"].view(-1)
                    deep_predict_scores = model_output['deep_predict_scores'].view(-1)
                    mask_dp = PoN_labels != -1
                    loss_dp = F.binary_cross_entropy_with_logits(
                        deep_predict_scores[mask_dp], PoN_labels[mask_dp], reduction="mean"
                    )
                    loss = loss * 0.0 + ori_loss + contrastive_loss + loss_dp
                else:
                    loss = loss*0.0 + ori_loss + contrastive_loss
            else:
                print("no term scores")
                loss = loss * 0.0 + self.pretrainloss(sample_list, model_output)

            if "recover_scores" in model_output:
                char_loss_mask = (sample_list['recover_labels'] != -1).float()
                term_vocab_size = 30001  # 30001
                recover_losses = F.cross_entropy(model_output["recover_scores"].view(-1, term_vocab_size),
                                                  sample_list['recover_labels'].view(-1), reduction="none",
                                                  ignore_index=-1)
                recover_losses *= char_loss_mask.view(-1)
                count = torch.max(torch.sum(char_loss_mask), self.one.to(recover_losses.device))
                recover_losses = torch.sum(recover_losses) / count
                loss = loss + recover_losses
        return loss

@registry.register_loss("softCE")
class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target, bn):
        probs = F.softmax(logits, -1)
        loss = (- target * torch.log(probs)).sum()
        loss = loss/bn

        return loss

@registry.register_loss("mce")
class MutilCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prob, target, soft_label=None):
        if not soft_label:
            soft_label = torch.tensor([0.0, 0.7, 0.9, 1.0])
        prob = torch.sigmoid(prob)
        loss = (target - soft_label[1]) * (target - soft_label[2]) * (target - soft_label[3]) * torch.log(
            1 + soft_label[0] - prob) \
               + (target - soft_label[0]) * (target - soft_label[2]) * (target - soft_label[3]) * torch.log(
            1 + soft_label[1] - prob) \
               + (target - soft_label[0]) * (target - soft_label[1]) * (target - soft_label[3]) * torch.log(
            1 + soft_label[2] - prob) \
               + (target - soft_label[0]) * (target - soft_label[1]) * (target - soft_label[2]) * torch.log(
            1 + soft_label[3] - prob)
        loss = abs(loss)
        loss = torch.sum(loss) / torch.numel(target)
        return loss

@registry.register_loss("mce2")
class Mutil2CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prob, target, soft_label=None):
        if not soft_label:
            #soft_label = torch.tensor([0.0, 0.7, 0.9, 1.0])
            soft_label = torch.tensor([1.0, 0.9, 0.7, 0.0])
        prob = torch.sigmoid(prob)
        if not True:
            loss = ((target - soft_label[1]) * (target - soft_label[2]) * (target - soft_label[3]) * torch.log(
                1 + soft_label[0] - prob)) / ((soft_label[0] - soft_label[1]) * (soft_label[0] - soft_label[2]) * (
                        soft_label[0] - soft_label[3])) \
                   + ((target - soft_label[0]) * (target - soft_label[2]) * (target - soft_label[3]) * torch.log(
                1 + soft_label[1] - prob)) / ((soft_label[1] - soft_label[0]) * (soft_label[1] - soft_label[2]) * (
                        soft_label[1] - soft_label[3])) \
                   + ((target - soft_label[0]) * (target - soft_label[1]) * (target - soft_label[3]) * torch.log(
                1 + soft_label[2] - prob)) / ((soft_label[2] - soft_label[0]) * (soft_label[2] - soft_label[1]) * (
                        soft_label[2] - soft_label[3])) \
                   + ((target - soft_label[0]) * (target - soft_label[1]) * (target - soft_label[2]) * torch.log(
                1 + soft_label[3] - prob)) / ((soft_label[3] - soft_label[0]) * (soft_label[3] - soft_label[1]) * (
                        soft_label[3] - soft_label[2]))
        else:
            loss = ((target - soft_label[1]) * (target - soft_label[2]) * (target - soft_label[3]) * torch.log(
                1 + soft_label[0] - prob)) \
                   + ((target - soft_label[0]) * (target - soft_label[2]) * (target - soft_label[3]) * torch.log(
                1 + soft_label[1] - prob))  \
                   + ((target - soft_label[0]) * (target - soft_label[1]) * (target - soft_label[3]) * torch.log(
                1 + soft_label[2] - prob))  \
                   + ((target - soft_label[0]) * (target - soft_label[1]) * (target - soft_label[2]) * torch.log(
                1 + soft_label[3] - prob))
        loss = abs(loss)
        print(loss)
        print(target)
        print(torch.isnan(loss).int().sum())
        print(torch.sum(loss))
        loss = torch.sum(loss) / torch.numel(target)
        print(loss)
        return loss

@registry.register_loss("pretrainloss_sum")
class PretrainLoss(nn.Module):
    """
    Collect all used pretrain losses
    """
    def __init__(self):
        super().__init__()
        self.mlm_cls = M4CDecodingPretrainLoss()
        self.pollute_cls = M4CDecodingPollutePretrainLoss()
        self.overlap_cls = M4CDecodingOverkapPretrainLoss()

    def forward(self, sample_list, model_output):
        return self.mlm_cls(sample_list, model_output) * 1. + \
            self.pollute_cls(sample_list, model_output) * 1. + \
            self.overlap_cls(sample_list, model_output) * 1.

@registry.register_loss("m4c_decoding_pretrain")
class M4CDecodingPretrainLoss(nn.Module):
    """
    Text mask and Seq matching for now
    """
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        """
        MLM compuated on the un-polluted samples, otherwise can't predict mask
        """
        scores = model_output["textcls_scores"].permute(0,2,1)
        targets = sample_list["cmb_text_mask_label"]
        loss_mask = (targets!=-1).float()
        pollute_mask = sample_list["tag_pollute"].float()
        pollute_mask = (1-pollute_mask).repeat(1,loss_mask.shape[-1])   ## 1 is polluted, not compute MLM
        loss_mask *= pollute_mask
        assert scores.dim() == 3 and loss_mask.dim() == 2
        losses = F.cross_entropy(
            scores, targets, reduction="none", ignore_index=-1
        )
        losses *= loss_mask
        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss

@registry.register_loss("m4c_decoding_pollute_pretrain")
class M4CDecodingPollutePretrainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        scores = model_output["pollutecls_scores"]
        targets = sample_list["tag_pollute"].float()

        loss = F.binary_cross_entropy_with_logits(
            scores, targets, reduction="mean"
        )
        return loss

@registry.register_loss("m4c_decoding_overlap_pretrain")
class M4CDecodingOverkapPretrainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        scores = model_output["overlapcls_scores"].squeeze(-1)
        targets = sample_list["overlap"].float()
        ## if binary
        loss = torch.zeros(1).cuda()
        mask = targets!=-1
        if mask.float().sum()!=0:
            loss = F.binary_cross_entropy_with_logits(
                scores[mask], targets[mask], reduction="mean"
            )
        # ## if cls
        # losses = F.cross_entropy(
        #     scores, targets.long(), reduction="none", ignore_index=-1
        # )   #ignore_index=-1
        # loss_mask =(targets!=-1).float()
        # losses *= loss_mask
        # count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        # loss = torch.sum(losses) / count
        return loss
