import os
import torch
import torch.nn as nn

from .utils import save_image, EMA
from .metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class DummyScheduler:
    @staticmethod
    def step():
        pass

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def state_dict():
        return None


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            diffusion,
            epochs,
            trainloader,
            sampler=None,
            sample_c=None,
            scheduler=None,
            use_ema=False,
            grad_norm=1.0,
            shape=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            chkpt_intv=5,  # save a checkpoint every {chkpt_intv} epochs
            num_save_images=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0  # process id for distributed training
    ):
        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        self.sample_c = sample_c
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = nullcontext()
        self.grad_norm = grad_norm
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.num_save_images = num_save_images

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.is_main = rank == 0

        self.stats = RunningStatistics(loss=None)

        self.stats_log = [] # saved in chkpt as 'stats_log':[{'loss': 1.003756324450175,...},...]

    def loss(self, x, emb):
        B = x.shape[0]
        T = self.diffusion.timesteps
        t = torch.randint(T, size=(B, ), dtype=torch.int64, device=self.device)
        loss = self.diffusion.train_losses(self.model, x_0=x, t=t, c=emb)
        assert loss.shape == (B, )
        return loss

    def step(self, x, emb):
        B = x.shape[0]
        loss = self.loss(x, emb).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient clipping by global norm
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
        self.optimizer.step()
        # adjust learning rate every step (warming up)
        self.scheduler.step()
        if self.use_ema:
            self.ema.update()
        self.stats.update(B, loss=loss.item() * B)

    def sample_fn(self, noise, c_emb, diffusion=None):
        if diffusion is None:
            diffusion = self.diffusion
        shape = noise.shape
        with self.ema:
            B, *_ = noise.shape
            guide_w = 1.0
            print(f'guide_w: {guide_w}, c.shape:{c_emb.shape}, c:{c_emb}')
            sample = diffusion.p_sample(
                denoise_fn=self.model, c=c_emb, guide_w=guide_w, shape=shape, device=self.device, noise=noise)
        assert sample.grad is None
        return sample

    def train(self, evaluator=None, chkpt_path=None, image_dir=None, sample_0=False, c_in_dim=512):

        num_samples = self.num_save_images
        if num_samples:
            noise = torch.randn((num_samples,) + self.shape)  # fixed x_T for image generation

        c_tensor = self.sample_c[None, :] # extend sample dim
        c_tensor = c_tensor.repeat(num_samples,1).to(self.device) # [1, emb_dim] -> [num_samples, emb_dim]

        assert c_tensor.shape[1] == c_in_dim, f"ASSERT ERROR: c_tensor ({c_tensor.shape[1]}) and c_in_dim ({c_in_dim}) are not equal"

        if sample_0:
            x = self.sample_fn(noise, c_tensor).cpu()
            save_image(x, os.path.join(image_dir, f"0.jpg"))

        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(self.trainloader, desc=f"{e+1}/{self.epochs} epochs", disable=not self.is_main) as t:
                for i, (x, emb) in enumerate(t):

                    if emb.ndim == 1:
                        # emb for mnist and cifar10 is [B,] and should be [B,c_in_dim]
                        emb = emb[:, None] # extend feature dim
                        emb = emb.repeat(1,c_in_dim).type(torch.float)

                    self.step(x.to(self.device), emb.to(self.device))
                    t.set_postfix(self.current_stats)

                    if i == len(self.trainloader) - 1:
                        self.model.eval()
                        if evaluator is not None:
                            eval_results = evaluator.eval(self.sample_fn)
                        else:
                            eval_results = dict()

                        results = dict()
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)

            self.stats_log.append(self.stats.extract())
            results.update({'stats_log':self.stats_log})

            if self.is_main:
                if not (e + 1) % self.chkpt_intv and chkpt_path:
                    self.save_checkpoint(chkpt_path, epoch=e+1, **results)
                if num_samples and image_dir:
                    x = self.sample_fn(noise, c_tensor).cpu()
                    save_image(x, os.path.join(image_dir, f"{e+1}.jpg"))
            if self.distributed:
                dist.barrier()  # synchronize all processes here

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def resume_from_chkpt(self, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            getattr(self, trainee).load_state_dict(chkpt[trainee])
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class Evaluator:
    def __init__(
            self,
            dataset,
            diffusion=None,
            eval_batch_size=256,
            max_eval_count=10000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.diffusion = diffusion
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn):
        self.istats.reset()
        for _ in range(0, self.max_eval_count + self.eval_batch_size, self.eval_batch_size):
            x = sample_fn(self.eval_batch_size, diffusion=self.diffusion)
            self.istats(x.to(self.device))
        gen_mean, gen_var = self.istats.get_statistics()
        return {"fid": calc_fd(gen_mean, gen_var, self.target_mean, self.target_var)}
