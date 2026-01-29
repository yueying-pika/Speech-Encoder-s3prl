# Copyright (c) Facebook, Inc. All Rights Reserved

import os
import editdistance
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized, get_rank
from torch.nn.utils.rnn import pad_sequence

from .model import *
from ..model import *
from .dataset import SequenceDataset
from .dataset import load_aishell_manifests, aishell_text_to_char_tokens
from .dictionary import Dictionary
from s3prl.metric.common import cer_zh as metric_cer_zn


def token_to_word(text):
    # Hard coding but it is only used here for now.
    # Assumption that units are characters. Doesn't handle BPE.
    # Inter-character separator is " " and inter-word separator is "|".
    return text.replace(" ", "").replace("|", " ").strip()


def get_decoder(decoder_args_dict, dictionary):
    decoder_args = Namespace(**decoder_args_dict)

    if decoder_args.decoder_type == "kenlm":
        from .w2l_decoder import W2lKenLMDecoder

        decoder_args.beam_size_token = len(dictionary)
        if isinstance(decoder_args.unk_weight, str):
            decoder_args.unk_weight = eval(decoder_args.unk_weight)
        return W2lKenLMDecoder(decoder_args, dictionary)

    return None


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs
    ):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame

            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.

                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir

        dict_path = self.datarc.get("dict_path", str(Path(__file__).parent / "char.dict"))
        dict_path = str(dict_path)
        if not os.path.isabs(dict_path):
            # Resolve relative paths w.r.t current working dir
            dict_path = os.path.normpath(os.path.join(os.getcwd(), dict_path))

        # Auto-build dictionary if not present (same idea as asr_seame).
        # IMPORTANT: All ranks must call barrier the same number of times to avoid deadlocks.
        def _build_dict():
            manifest_dir = self.datarc.get("manifest_dir")
            assert (
                manifest_dir is not None
            ), "config.downstream_expert.datarc.manifest_dir is required for asr_aishell_1"
            train_manifests = self.datarc.get("train")
            assert isinstance(train_manifests, (list, tuple)) and len(train_manifests) > 0, (
                "config.downstream_expert.datarc.train must be a non-empty list of manifest filenames"
            )
            manifest_paths = [
                m if os.path.isabs(m) else os.path.join(manifest_dir, m) for m in train_manifests
            ]
            segments = load_aishell_manifests(manifest_paths)
            transcripts = [aishell_text_to_char_tokens(s.text) for s in segments]

            d = Dictionary()
            Dictionary.add_transcripts_to_dictionary(transcripts, d, num_workers=1)
            d.finalize(threshold=-1, nwords=-1, padding_factor=8)

            os.makedirs(os.path.dirname(dict_path), exist_ok=True)
            d.save(dict_path)

        if is_initialized():
            if get_rank() == 0:
                if (not os.path.isfile(dict_path)) or os.path.getsize(dict_path) == 0:
                    _build_dict()
            torch.distributed.barrier()
        else:
            if (not os.path.isfile(dict_path)) or os.path.getsize(dict_path) == 0:
                _build_dict()

        self.dictionary = Dictionary.load(dict_path)

        self.projector = nn.Linear(upstream_dim, self.modelrc["project_dim"])
        model_cls = eval(self.modelrc["select"])
        model_conf = self.modelrc[self.modelrc["select"]]
        self.model = model_cls(
            self.modelrc["project_dim"],
            len(self.dictionary.symbols),
            upstream_rate,
            **model_conf,
        )
        self.blank = self.dictionary.bos()
        self.objective = nn.CTCLoss(
            blank=self.blank, zero_infinity=self.datarc["zero_infinity"]
        )
        decoder_args = self.datarc.get("decoder_args")
        self.decoder = get_decoder(decoder_args, self.dictionary)
        self.register_buffer("best_score", torch.ones(1) * 100)

    # Interface
    def get_dataloader(self, split):
        """
        Args:
            split: string
                The name of the dataloader, can be train/dev/test-clean/test-other for asr

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """
        if not hasattr(self, f"{split}_dataset"):
            batch_size = (
                self.datarc["batch_size"]
                if split == "train"
                else self.datarc["eval_batch_size"]
            )
            setattr(
                self,
                f"{split}_dataset",
                SequenceDataset(split, batch_size, self.dictionary, **self.datarc),
            )

        if split == "train":
            return self._get_train_dataloader(self.train_dataset)
        else:
            return self._get_eval_dataloader(getattr(self, f"{split}_dataset"))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _compute_uer(self, pred_tokens_all, target_tokens_all) -> float:
        """
        Unit Error Rate (UER) on token sequences (percentage, can be > 100).

        NOTE:
        - This follows the original ASR downstream behavior: compute edit distance on
          space-separated unit tokens returned by `Dictionary.string(...)`.
        - For AISHELL-1 (pure Chinese), we ignore the boundary token '|' during scoring,
          so UER reflects character units only.
        """
        unit_error_sum = 0.0
        unit_length_sum = 0
        for pred_tokens, target_tokens in zip(pred_tokens_all, target_tokens_all):
            pred_seq = [t for t in (pred_tokens or "").split() if t != "|"]
            target_seq = [t for t in (target_tokens or "").split() if t != "|"]
            unit_error_sum += editdistance.eval(pred_seq, target_seq)
            unit_length_sum += len(target_seq)
        if unit_length_sum == 0:
            return 0.0 if unit_error_sum == 0 else 100.0
        return 100.0 * unit_error_sum / unit_length_sum

    def _compute_cer_zn(self, pred_tokens_all, target_tokens_all):
        """
        Compute Chinese CER (CER_ZN) using `s3prl.metric.common.cer_zh`.

        Note:
        - Our internal token representation uses " " between characters and "|" as word boundary.
        - `cer_zh` will keep only CJK characters and compute edit distance safely.
        """
        hyps_text = [token_to_word(t) for t in pred_tokens_all]
        refs_text = [token_to_word(t) for t in target_tokens_all]
        return metric_cer_zn(hyps_text, refs_text) * 100.0

    def _decode(self, log_probs, input_lens):
        """log概率 - > 带空格的自然语言序列和不带空格的自然语言序列"""
        pred_tokens_batch = []
        pred_words_batch = []

        for log_prob, in_len in zip(log_probs, input_lens):
            log_prob = log_prob[:in_len].unsqueeze(0)
            decoded = None
            if self.decoder is not None and not self.training:
                decoded = self.decoder.decode(log_prob)
                if len(decoded) >= 1:
                    decoded = decoded[0]
                    decoded = None if len(decoded) < 1 else decoded[0]

            pred_token_ids = log_prob.argmax(dim=-1).unique_consecutive()
            pred_token_ids = pred_token_ids[pred_token_ids != self.blank].tolist()
            pred_tokens = self.dictionary.string(pred_token_ids)

            if decoded is not None and "words" in decoded:
                pred_words = decoded["words"]
            else:
                pred_words = token_to_word(pred_tokens).split()

            pred_tokens_batch.append(pred_tokens)
            pred_words_batch.append(pred_words)

        return pred_tokens_batch, pred_words_batch

    def _get_log_probs(self, features):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features])
        features = pad_sequence(features, batch_first=True).to(device=device)
        features = self.projector(features)
        logits, log_probs_len = self.model(features, features_len)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs, log_probs_len

    def inference(self, features, filenames):
        log_probs, log_probs_len = self._get_log_probs(features)
        _, pred_words_batch = self._decode(
            log_probs.float().contiguous().cpu(), log_probs_len
        )
        hyps = [" ".join(hyp) for hyp in pred_words_batch]

        if filenames != []:
            with open(Path(self.expdir) / "inference.ark", "w") as file:
                for hyp, filename in zip(hyps, filenames):
                    file.write(f"{filename} {hyp}\n")

        return hyps

    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
        """
        Args:
            split: string
                The name of the dataloader, can be train/dev/test-clean/test-other for asr

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        log_probs, log_probs_len = self._get_log_probs(features)
        # print("log_probs: ", log_probs)
        # print("log_probs.shape: ", log_probs.shape)
        device = features[0].device
        labels = [torch.IntTensor(l) for l in labels]
        
        labels_len = torch.IntTensor([len(label) for label in labels]).to(device=device)
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.dictionary.pad(),
        ).to(device=device)
        loss = self.objective(
            log_probs.transpose(0, 1),  # (N, T, C) -> (T, N, C)
            labels,
            log_probs_len,
            labels_len,
        )
        records["loss"].append(loss.item())

        target_tokens_batch = []
        target_words_batch = []
        for label in labels:
            label_idx = (label != self.dictionary.pad()) & (
                label != self.dictionary.eos()
            )
            target_token_ids = label[label_idx].tolist()
            # print("target_token_ids: ", target_token_ids) [687, 695, 248, 118, 19, 815, 99, 23, 586, 1597, 1163, 282, 200, 7, 73, 30, 740, 3075, 277, 318, 4]
            target_tokens = self.dictionary.string(target_token_ids)
            # print("target_tokens: ", target_tokens) 德 胜 门 内 大 街 九 三 号 塌 陷 处 正 在 进 行 防 汛 改 造 |
            target_words = token_to_word(target_tokens).split()
            # print("target_words: ", target_words) '德胜门内大街九三号塌陷处正在进行防汛改造']

            target_tokens_batch.append(target_tokens)
            target_words_batch.append(target_words)

        with torch.no_grad():
            pred_tokens_batch, pred_words_batch = self._decode(
                log_probs.float().contiguous().cpu(), log_probs_len
            )

        records["target_tokens"] += target_tokens_batch
        records["target_words"] += target_words_batch
        records["pred_tokens"] += pred_tokens_batch
        records["pred_words"] += pred_words_batch
        records["filenames"] += filenames

        return loss

    # interface
    def log_records(
        self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        """
        Args:
            split: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev' or 'test-clean' or 'test-other' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        loss = torch.FloatTensor(records["loss"]).mean().item()
        print(f"{split} loss: {loss}")

        uer = self._compute_uer(records["pred_tokens"], records["target_tokens"])
        # print("records['pred_tokens']: ", records["pred_tokens"]) '房 企 和 电 商 平 均 投 资 体 育 产 业 |'
        # print("records['target_tokens']: ", records["target_tokens"])  '并 且 仅 仅 有 果 粉 传 统 意 义 上 的 支 持 还 不 够 |', '国 内 机 器 人 行 业 投 资 分 为 三 个 阶 段 第 一 个 阶 段 |'
        cer_zn = self._compute_cer_zn(records["pred_tokens"], records["target_tokens"])
        logger.add_scalar(f"asr/{split}-loss", loss, global_step=global_step)
        logger.add_scalar(f"asr/{split}-uer", uer, global_step=global_step)
        logger.add_scalar(f"asr/{split}-cer_zn", cer_zn, global_step=global_step)
        print(f"{split} uer: {uer}")
        print(f"{split} cer_zn: {cer_zn}")

        save_names = []
        # Lower is better
        if split == "dev" and cer_zn < self.best_score:
            self.best_score = torch.ones(1) * uer
            save_names.append(f"{split}-best.ckpt")

        if "test" in split or "dev" in split:
            lm = "noLM" if self.decoder is None else "LM"
            hyp_ark = open(os.path.join(self.expdir, f"{split}-{lm}-hyp.ark"), "w")
            ref_ark = open(os.path.join(self.expdir, f"{split}-{lm}-ref.ark"), "w")
            for filename, hyp, ref in zip(
                records["filenames"], records["pred_words"], records["target_words"]
            ):
                hyp = " ".join(hyp)
                ref = " ".join(ref)
                hyp_ark.write(f"{filename} {hyp}\n")
                ref_ark.write(f"{filename} {ref}\n")
            hyp_ark.close()
            ref_ark.close()

        return save_names
