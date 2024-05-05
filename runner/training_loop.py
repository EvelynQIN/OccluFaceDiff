# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/openai/guided-diffusion
# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import functools

import os

import torch
import gc
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import create_named_schedule_sampler
from torch.optim import AdamW
from utils.scheduler import WarmupCosineSchedule
from tqdm import tqdm
from utils import dist_util
import wandb
# from model.deca import EMOCA
# from utils import utils_transform
from memory_profiler import profile

# training loop for the diffusion given "model" as the denoising model
class TrainLoop:
    def __init__(self, args, model_cfg, denoise_model, diffusion, train_loader, val_loader):
        """

        Args:
            model: the denoising model (diffuseMLP)
            diffusion: spaced diffusion
            data: the data loader
        """
        self.args = args
        self.model_cfg = model_cfg
        self.dataset = args.dataset
        self.model = denoise_model 
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.load_optimizer = args.load_optimizer
        self.use_fp16 = False
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.epoch = 0
        self.step = 0
        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        self.resume_epoch = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * self.gradient_accumulation_steps
        self.num_epochs = args.num_epoch
        self.num_steps = self.num_epochs * self.steps_per_epoch
        
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.args.cosine_scheduler:
            self.scheduler = WarmupCosineSchedule(self.opt, warmup_steps=args.warmup_steps, t_total=self.num_steps)
        
        if self.resume_epoch and self.load_optimizer:
            self._load_optimizer_state()

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.use_ddp = False
        self.ddp_model = self.model 

        self.loss_keys = None

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_epoch = parse_resume_epoch_from_filename(resume_checkpoint)
            self.resume_step = self.resume_epoch * self.steps_per_epoch
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint,
                    map_location=dist_util.dev(),
                ),
                strict=False
            )

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt_{self.resume_epoch}.pt"
        )

        print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        assert os.path.exists(opt_checkpoint), "optimiser states does not exist."
        state_dict = dist_util.load_state_dict(
            opt_checkpoint, map_location=dist_util.dev()
        )
        self.opt.load_state_dict(state_dict)
    
    def run_loop(self):
        local_step = 0
        for epoch in range(self.resume_epoch+1, self.num_epochs+1):
            self.model.train()
            print(f"Starting training epoch {epoch}")
            self.epoch = epoch

            if epoch > 0 and epoch % 3 == 0:
                self.model.unfreeze_wav2vec()
            else:
                self.model.freeze_wav2vec()

            for batch in tqdm(self.train_loader):
                local_step += 1
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                target = batch['target']
                model_kwargs = {}
                for k in batch:
                    if k != 'target':
                        model_kwargs[k] = batch[k]

                grad_update = True if local_step % self.gradient_accumulation_steps == 0 else False 
                self.run_step(target, grad_update, **model_kwargs)
            if epoch == self.num_epochs or epoch % self.save_interval == 0:
                self.save()
            if epoch % self.log_interval == 0:
                self.validation()

        # Save the last checkpoint if it wasn't already saved.
        if (self.epoch) % self.save_interval != 0:
            self.save()
    
    # @profile
    def run_step(self, batch, grad_update, **model_kwargs):
        if grad_update:
            self.step += 1
            self.forward_backward(batch, log_loss=True, **model_kwargs)
            self.mp_trainer.optimize(self.opt)
            if self.args.cosine_scheduler:
                self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()[0]
            else:
                self._step_lr()
            self.mp_trainer.zero_grad()
        else:
            self.forward_backward(batch, log_loss=False, **model_kwargs)
        # # free gpu memory
        del batch
        torch.cuda.empty_cache()
        # gc.collect()

    # @profile
    def forward_backward(self, batch, log_loss=True, **model_kwargs):

        t, _ = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,
            t,
            model_kwargs
        )

        loss_dict = compute_losses()
        if self.loss_keys is None:
            self.loss_keys = loss_dict.keys()

        # if isinstance(self.schedule_sampler, LossAwareSampler):
        #     self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        # normalize loss to account for batch accumulation
        loss = loss_dict['loss'] / self.gradient_accumulation_steps
        self.mp_trainer.backward(loss)
        if log_loss:
            loss_dict['loss'] = loss_dict['loss'].detach().item()
            self.log_loss_dict(loss_dict, "train")
        
        del batch, model_kwargs, loss_dict
        
    
    def validation(self):
        self.model.eval()
        print("start eval ...")
        val_loss = dict()
        for key in self.loss_keys:
            val_loss[key] = 0.0
        eval_steps = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                eval_steps += 1
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                target = batch['target']
                model_kwargs = {}
                for k in batch:
                    if k != 'target':
                        model_kwargs[k] = batch[k]
                t, weights = self.schedule_sampler.sample(target.shape[0], dist_util.dev())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    target,
                    t,
                    model_kwargs
                )

                loss_dict = compute_losses()
                loss_dict['loss'] = loss_dict['loss'].detach().item()
                for k in val_loss:
                    val_loss[k] +=  loss_dict[k]
                del batch, model_kwargs, loss_dict
                # torch.cuda.empty_cache()
                
            for k in val_loss:
                val_loss[k] /= eval_steps
            
            self.log_loss_dict(val_loss, phase="validation")
            del val_loss

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _step_lr(self):
        # One-step learning rate decay if needed.
        if not self.lr_anneal_steps:
            return
        if (self.step + self.resume_step) > self.lr_anneal_steps:
            self.lr = self.lr / 30.0
            self.lr_anneal_steps = False
        else:
            self.lr = self.lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr
    
    def log_loss_dict(self, losses, phase="train"):
        loss_dict = {}
        for key, values in losses.items():
            kid = f"{phase}/{key}"
            loss_dict[kid] = values
        loss_dict["epoch"] = self.epoch
        if phase == "train":
            loss_dict['lr'] = self.lr
        if self.args.wandb_log:
            wandb.log(loss_dict)
            # # Log the quantiles (four quartiles, in particular).
            # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            #     quartile = int(4 * sub_t / diffusion.num_timesteps)
            #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss

    def ckpt_file_name(self):
        return f"model_{(self.epoch)}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            print("saving model...")
            filename = self.ckpt_file_name()

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            with open(
                os.path.join(self.save_dir, filename),
                "wb",
            ) as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with open(
            os.path.join(self.save_dir, f"opt_{self.epoch}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
    
    def save_wav2vec(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        ckpt_path = os.path.join(self.save_dir, f'wav2vec_{self.epoch}.pt')
        self.model.save_audio_ckpt(ckpt_path)


def parse_resume_epoch_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0



