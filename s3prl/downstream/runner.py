import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

from huggingface_hub import HfApi, HfFolder, Repository

SAMPLE_RATE = 16000

MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu', weights_only=False) if self.args.init_ckpt else {}

        self.upstream = self._get_upstream() # 调用函数_get_upstream, 加载预训练的上游模型（如 wav2vec2、HuBERT）（调用函数_init_model、_load_weight）
        self.featurizer = self._get_featurizer() # 调用函数_get_featurizer, 特征提取器（调用函数_init_model）
        self.downstream = self._get_downstream() # 调用函数_get_downstream, 下游任务模型（如 ASR、分类）（调用函数_init_model）
        self.all_entries = [self.upstream, self.featurizer, self.downstream]
        
        # 用于记录是否已经打印过 batch 格式信息
        self._batch_format_logged = False

    #  -e参数，加载预训练权重，从之前保存的 checkpoint 中恢复模型权重（用于断点续训）
    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)

    # 初始化模型
    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)

    # 判断模型来源(hugging or 本地), 分布式训练同步,  创建模型实例并移到 GPU, _init_model() 返回
    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model( # 调用函数 _init_model()
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable, # -f 参数，如果为 True，上游模型参数会参与训练（微调）；如果为 False，只提取特征，参数冻结
            interfaces = ["get_downsample_rates"]
        )

    # 调用 interface.py 函数, 创建特征提取器，从上游模型的多层输出中选择和处理特征。
    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,  # 上游模型引用
            feature_selection = self.args.upstream_feature_selection, # 选哪种特征
            layer_selection = self.args.upstream_layer_selection, # 选哪一层
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize, # 是否归一化
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )

    # 动态加载下游任务模型（如 ASR、语音分类、说话人识别等）
    def _get_downstream(self):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")  # -d参数，加载"asr"任务（也可以是IC、SID、ER）
        Downstream = getattr(expert, "DownstreamExpert") # 不同的下游任务有不同的 DownstreamExpert 类
        # 下游任务的参数
        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim, # 输入特征维度
            upstream_rate = self.featurizer.model.downsample_rate, # 时间下采样率
            **self.config, # 任务特定配置
            **vars(self.args) # 运行时参数
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
        )


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        self._load_weight(optimizer, 'Optimizer')
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    # ============================================================
    # 新增：健壮的 filepaths 提取方法
    # ============================================================
    def _parse_batch_others(self, others):
        """
        解析 dataloader 返回的 others，提取 filepaths
        
        ASR dataloader 可能返回:
        - 旧格式: (wavs, labels, filenames)           -> others = [labels, filenames]
        - 新格式: (wavs, labels, filenames, filepaths) -> others = [labels, filenames, filepaths]
        
        Args:
            others: dataloader 返回的除 wavs 外的其他数据
        
        Returns:
            tuple: (filepaths, others_for_downstream)
                - filepaths: 文件路径列表，如果不存在则为 None
                - others_for_downstream: 传递给 downstream 的数据（不包含 filepaths）
        """
        filepaths = None
        others_for_downstream = others
        
        # 至少需要 3 个元素才可能包含 filepaths
        # others = [labels, filenames, filepaths]
        if len(others) >= 3:
            candidate = others[-1]
            
            # 验证条件：
            # 1. 是列表或元组
            # 2. 非空
            # 3. 元素是字符串
            # 4. 字符串看起来像文件路径（包含路径分隔符或音频扩展名）
            is_filepath_list = False
            
            if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
                if isinstance(candidate[0], str):
                    sample = candidate[0]
                    # 检查路径特征：包含分隔符或常见音频扩展名
                    looks_like_path = (
                        '/' in sample or 
                        '\\' in sample or
                        sample.endswith(('.flac', '.wav', '.mp3', '.ogg', '.m4a'))
                    )
                    if looks_like_path:
                        is_filepath_list = True
            
            if is_filepath_list:
                filepaths = candidate
                others_for_downstream = others[:-1]
        
        # 首次调用时打印格式信息，方便调试
        if not self._batch_format_logged:
            self._batch_format_logged = True
            print(f"[Runner] Batch format detected:")
            print(f"[Runner]   - others length: {len(others)}")
            print(f"[Runner]   - filepaths detected: {filepaths is not None}")
            if filepaths and len(filepaths) > 0:
                print(f"[Runner]   - sample filepath: {filepaths[0][:100]}...")
        
        return filepaths, others_for_downstream


    # -m 参数，调用train()，训练函数
    # 准备阶段：设置 train/eval 模式；创建优化器、调度器；初始化进度条、日志  
    # 训练循环：前向传播；计算损失；反向传播；梯度更新；日志/评估/保存
    # 结束阶段：关闭日志
    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_entries:
            if entry.trainable:
                entry.model.train().to(self.args.device)
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval()

        # set amp
        amp = self.config['runner'].get('fp16', False)
        if amp:
            print('[Runner] - Enabled fp16 training')
            scaler = torch.cuda.amp.GradScaler()

        # optimizer
        optimizer = self._get_optimizer(trainable_models) # 创建优化器（如 Adam、AdamW）

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer) # 创建学习率调度器（如 warmup、cosine decay）

        # specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)
        train_split = self.config['runner'].get("train_dataloader", "train")
        while pbar.n < pbar.total:
            try:
                dataloader = self.downstream.model.get_dataloader(train_split, epoch=epoch)
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstream.model.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise

            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    if len(others) >= 4 and isinstance(others[0], dict):
                        # 新格式：有预处理数据
                        preprocessed_inputs = others[0]
                        others_for_downstream = others[1:-1]  # labels, filenames
                        filepaths = others[-1]
                    else:
                        # 旧格式：兼容
                        preprocessed_inputs = None
                        filepaths, others_for_downstream = self._parse_batch_others(others)

                    # ============================================================
                    # 修改：使用健壮的 filepaths 提取方法
                    # ============================================================
                    # filepaths, others_for_downstream = self._parse_batch_others(others)

                    # wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                    
                    wavs = [
                        wav if isinstance(wav, torch.Tensor) 
                        else torch.as_tensor(wav, dtype=torch.float32)
                        for wav in wavs
                    ]
                                        
                    with torch.cuda.amp.autocast(enabled=amp):
                        if self.upstream.trainable:
                            # features = self.upstream.model(wavs, wav_paths=filepaths)
                            features = self.upstream.model(
                                wavs, 
                                preprocessed_inputs=preprocessed_inputs,  
                                wav_paths=filepaths
                            )
                        else:
                            with torch.no_grad():
                                features = self.upstream.model(
                                    wavs, 
                                    preprocessed_inputs=preprocessed_inputs,  
                                    wav_paths=filepaths
                                )
                        features = self.featurizer.model(wavs, features)

                        if specaug:
                            features, _ = specaug(features)

                        loss = self.downstream.model(
                            train_split,
                            features, *others_for_downstream,
                            records = records,
                        )
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                    loss = (loss / gradient_accumulate_steps)
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    del loss

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # unscale
                if amp:
                    scaler.unscale_(optimizer)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config['runner']['gradient_clipping'])

                # optimize
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif math.isnan(grad_norm):
                    print(f'[Runner] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    self.downstream.model.log_records(
                        train_split,
                        records = records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloader),
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if len(save_names) > 0:
                    all_states = {
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Args': self.args,
                        'Config': self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if is_initialized():
                        all_states['WorldSize'] = get_world_size()

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)

                pbar.update(1)
            epoch += 1

        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()


    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            # ============================================================
            # 与 train() 保持一致的格式检测
            # ============================================================
            if len(others) >= 4 and isinstance(others[0], dict):
                # 新格式：有预处理数据
                preprocessed_inputs = others[0]
                others_for_downstream = others[1:-1]  # labels, filenames
                filepaths = others[-1]
            else:
                # 旧格式：兼容
                preprocessed_inputs = None
                filepaths, others_for_downstream = self._parse_batch_others(others)

            wavs = [
                wav.to(self.args.device) if isinstance(wav, torch.Tensor) 
                else torch.as_tensor(wav, dtype=torch.float32).to(self.args.device)
                for wav in wavs
            ]
            
            with torch.no_grad():
                features = self.upstream.model(   # 先算上游特征
                    wavs, 
                    preprocessed_inputs=preprocessed_inputs,
                    wav_paths=filepaths
                )
                features = self.featurizer.model(wavs, features)   # 再把上游特征交给 featurizer
                self.downstream.model(
                    split,
                    features, *others_for_downstream,
                    records = records,
                    batch_id = batch_id,
                )
                batch_ids.append(batch_id)

        save_names = self.downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train().to(self.args.device)

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            # 对于 inference，传递文件路径
            features = self.upstream.model(wavs, wav_paths=[str(filepath)])
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        print(f"[Runner] - Organisation to push fine-tuned model to: {organization}")
        
        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        print(f"[Runner] - Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(REPO_ROOT_DIR, self.args.downstream, self.args.expname)
        print(f"[Runner] - Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR, clone_from=repo_url, use_auth_token=huggingface_token
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(self.args.expdir, REPO_TASK_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns(HF_HUB_DIR))

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            print("[Runner] - Did not find a best checkpoint! Using the final checkpoint instead ...")
            CKPT_PATH = (
                os.path.join(REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt")
                )
        elif len(checkpoints) > 1:
            print(f"[Runner] - More than one best checkpoint found! Using {checkpoints[0]} as default ...")
            CKPT_PATH = checkpoints[0]
        else:
            print(f"[Runner] - Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        print("[Runner] - Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        print("[Runner] - Training run complete!")