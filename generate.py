if __name__ == "__main__":
    import os
    import json
    import math
    import uuid
    import torch
    from tqdm import trange
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from concurrent.futures import ThreadPoolExecutor
    from ddpm_torch import *
    from ddim import DDIM, get_selection_schedule
    from argparse import ArgumentParser

    # python generate.py --dataset celeba --c 1.0 --guide-w 1.0 --batch-size 6 --total-size 6 --chkpt-path chkpts/ddpm_celeba_26.pt --device cuda:0

    parser = ArgumentParser()
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--total-size", default=50000, type=int)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-path", default="", type=str)
    parser.add_argument("--save-dir", default="./images/eval", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--eta", default=0., type=float)
    parser.add_argument("--skip-schedule", default="linear", type=str)
    parser.add_argument("--subseq-size", default=10, type=int)
    parser.add_argument("--suffix", default="", type=str)
    
    parser.add_argument("--guide-w", default=1.0, type=float)
    parser.add_argument("--c", default=1.0, type=float)

    args = parser.parse_args()

    dataset = args.dataset
    root = os.path.expanduser("~/datasets")

    in_channels = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    config_dir = args.config_dir
    with open(os.path.join(config_dir, dataset + ".json")) as f:
        configs = json.load(f)

    diffusion_kwargs = configs["diffusion"]
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
    betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)

    use_ddim = args.use_ddim
    if use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"
        skip_schedule = args.skip_schedule
        eta = args.eta
        subseq_size = args.subseq_size
        subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)
        diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    device = torch.device(args.device)
    model = UNet(out_channels=in_channels, **configs["denoise"])
    model.to(device)
    chkpt_dir = args.chkpt_dir
    chkpt_path = args.chkpt_path or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
    folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
    use_ema = args.use_ema
    if use_ema:
        state_dict = torch.load(chkpt_path, map_location=device)["ema"]["shadow"]
    else:
        state_dict = torch.load(chkpt_path, map_location=device)["model"]
    for k in list(state_dict.keys()):
        if k.split(".")[0] == "module":  # state_dict of DDP
            state_dict[".".join(k.split(".")[1:])] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    folder_name = folder_name + args.suffix
    save_dir = os.path.join(args.save_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = args.batch_size
    total_size = args.total_size
    num_eval_batches = math.ceil(total_size / batch_size)
    shape = (batch_size, in_channels, image_res, image_res)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i in trange(num_eval_batches):
            if i == num_eval_batches - 1:
                shape = (total_size - i * batch_size, in_channels, image_res, image_res)
                c = torch.ones((shape[0], configs["denoise"]["c_in_dim"])) * args.c
                x = diffusion.p_sample(model, c=c, guide_w=args.guide_w, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
            else:
                c = torch.ones((shape[0], configs["denoise"]["c_in_dim"])) * args.c
                x = diffusion.p_sample(model, c=c, guide_w=args.guide_w, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
            x = (x * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)

            if in_channels == 1:
                x = x.repeat(1, 3, 1, 1)
            print(x.shape)
            img = make_grid(x, nrow=8, normalize=False, value_range=(0., 255.))
            img = img.permute(1, 2, 0)
            print(img.shape)
            _ = plt.imsave(f"{save_dir}/c-{args.c}_gw-{args.guide_w}_{uuid.uuid4()}.png", img.numpy())
