import math
import torch
import torch.nn as nn
from ..modules import Linear, Conv2d, SamePad2d, Sequential
from ..functions import get_timestep_embedding


DEFAULT_NONLINEARITY = nn.SiLU()  # f(x)=x*sigmoid(x)
DEFAULT_NORMALIZER = nn.GroupNorm


# python train.py --dataset celeba --batch-size 4 --train-device cuda:0 --epochs 50 --chkpt-intv 1 --summary


class AttentionBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER

    def __init__(
            self,
            in_channels,
            mid_channels=None,
            out_channels=None
    ):
        super(AttentionBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.project_in = Conv2d(in_channels, 3 * mid_channels, 1)
        self.project_out = Conv2d(mid_channels, out_channels, 1, init_scale=0.)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.skip = nn.Identity() if in_channels == out_channels else Conv2d(in_channels, out_channels, 1)

    @staticmethod
    def qkv(q, k, v):
        B, C, H, W = q.shape
        w = torch.einsum("bchw, bcHW -> bhwHW", q, k)
        w = torch.softmax(
            w.reshape(B, H, W, H * W) / math.sqrt(C), dim=-1).reshape(B, H, W, H, W)
        out = torch.einsum("bhwHW, bcHW -> bchw", w, v)
        return out

    def forward(self, x, t_emb=None, c_emb=None):
        skip = self.skip(x)
        C = x.shape[1]
        assert C == self.in_channels
        q, k, v = self.project_in(x).chunk(3, dim=1)
        #print(f'q:{q.shape} k:{k.shape} v:{v.shape}')
        x = self.qkv(q, k, v)
        x = self.project_out(x)
        return x + skip


class ResidualBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(
            self,
            in_channels,
            out_channels,
            embed_dim,
            num_groups=32,
            drop_rate=0.5
    ):
        super(ResidualBlock, self).__init__()
        self.norm1 = self.normalize(num_groups, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.fc = Linear(embed_dim, out_channels)
        self.norm2 = self.normalize(num_groups, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, init_scale=0.)
        self.skip = nn.Identity() if in_channels == out_channels else Conv2d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)

    def forward(self, x, t_emb):
        skip = self.skip(x)
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        x += self.fc(self.nonlinearity(t_emb))[:, :, None, None]
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        x = self.conv2(x)
        return x + skip

class ConditionalResidualBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(
            self,
            in_channels,
            out_channels,
            t_embed_dim,
            c_embed_dim,
            num_groups=32,
            drop_rate=0.5
    ):
        super(ConditionalResidualBlock, self).__init__()
        self.norm1 = self.normalize(num_groups, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.t_fc = Linear(t_embed_dim, out_channels)
        self.c_fc = Linear(c_embed_dim, out_channels)
        self.norm2 = self.normalize(num_groups, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, init_scale=0.)
        self.skip = nn.Identity() if in_channels == out_channels else Conv2d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)

    def forward(self, x, t_emb, c_emb):
        skip = self.skip(x)
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        x *= self.c_fc(self.nonlinearity(c_emb))[:, :, None, None]
        x += self.t_fc(self.nonlinearity(t_emb))[:, :, None, None]
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        x = self.conv2(x)
        return x + skip


class UNet(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(
            self,
            in_channels,
            hid_channels,
            out_channels,
            ch_multipliers,
            num_res_blocks,
            apply_attn,
            t_embed_dim=None,
            c_embed_dim=None,
            c_in_dim=None,
            num_groups=32,
            drop_rate=0.,
            resample_with_conv=True,
            c_value_range=[0.0,1.0] # [0, n_classes]
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels

        self.t_embed_dim = t_embed_dim or 4 * self.hid_channels
        self.c_embed_dim = c_embed_dim or 4 * self.hid_channels

        assert c_in_dim is not None and c_in_dim > 0, "ASSERT FAILED: c_in_dim cannot be None or <= 0"
        self.c_in_dim = c_in_dim

        self.c_value_range = c_value_range

        levels = len(ch_multipliers)
        self.ch_multipliers = ch_multipliers
        if isinstance(apply_attn, bool):
            apply_attn = [apply_attn for _ in range(levels)]
        self.apply_attn = apply_attn
        self.num_res_blocks = num_res_blocks
        self.drop_rate = drop_rate
        self.resample_with_conv = resample_with_conv

        self.t_embed = Sequential(
            Linear(self.hid_channels, self.t_embed_dim),
            self.nonlinearity,
            Linear(self.t_embed_dim, self.t_embed_dim)
        )

        self.c_embed = Sequential(
            Linear(self.c_in_dim, self.c_embed_dim),
            self.nonlinearity,
            Linear(self.c_embed_dim, self.c_embed_dim)
        )

        self.in_conv = Conv2d(in_channels, hid_channels, 3, 1, 1)
        
        self.levels = levels
        self.downsamples = nn.ModuleDict({f"level_{i}": self.downsample_level(i) for i in range(levels)})
        
        mid_channels = ch_multipliers[-1] * hid_channels
        self.middle = Sequential(
            ConditionalResidualBlock(mid_channels, mid_channels, t_embed_dim=self.t_embed_dim, c_embed_dim=self.c_embed_dim, drop_rate=drop_rate),
            AttentionBlock(mid_channels),
            ConditionalResidualBlock(mid_channels, mid_channels, t_embed_dim=self.t_embed_dim, c_embed_dim=self.c_embed_dim, drop_rate=drop_rate)
        )

        self.upsamples = nn.ModuleDict({f"level_{i}": self.upsample_level(i) for i in range(levels)})
        
        self.out_conv = Sequential(
            self.normalize(num_groups, hid_channels),
            self.nonlinearity,
            Conv2d(hid_channels, out_channels, 3, 1, 1, init_scale=0.)
        )

    def get_level_block(self, level):
        drop_rate = self.drop_rate
        if self.apply_attn[level]:
            def block(in_chans, out_chans):
                return Sequential(
                    ConditionalResidualBlock(in_chans, out_chans, t_embed_dim=self.t_embed_dim, c_embed_dim=self.c_embed_dim, drop_rate=drop_rate),
                    AttentionBlock(out_chans)
                )
        else:
            def block(in_chans, out_chans):
                return ConditionalResidualBlock(in_chans, out_chans, t_embed_dim=self.t_embed_dim, c_embed_dim=self.c_embed_dim, drop_rate=drop_rate)
        return block

    def downsample_level(self, level):
        block = self.get_level_block(level)
        prev_chans = self.hid_channels if level == 0 else self.ch_multipliers[level-1] * self.hid_channels
        curr_chans = self.ch_multipliers[level] * self.hid_channels
        modules = nn.ModuleList([block(prev_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1): # -1 pq ya se ha añadido un bloque en la instrucción anterior
            modules.append(block(curr_chans, curr_chans))
        if level != self.levels - 1: # si no es el último
            if self.resample_with_conv:
                downsample = Sequential(
                    SamePad2d(3, 2),
                    Conv2d(curr_chans, curr_chans, 3, 2)
                )
            else:
                downsample = nn.AvgPool2d(2)
            modules.append(downsample)
        return modules

    def upsample_level(self, level):
        block = self.get_level_block(level)
        ch = self.hid_channels # 128
        chs = list(map(lambda x: ch*x, self.ch_multipliers)) # [ 128, 256, 256, 256 ]
        next_chans = ch if level == 0 else chs[level - 1]
        prev_chans = chs[-1] if level == self.levels - 1 else chs[level + 1]
        curr_chans = chs[level]
        modules = nn.ModuleList([block(prev_chans + curr_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(2 * curr_chans, curr_chans))
        modules.append(block(next_chans + curr_chans, curr_chans))
        if level != 0:
            # Note: the official TensorFlow implementation specifies `align_corners=True`
            # However, PyTorch does not support align_corners for nearest interpolation
            upsample = [nn.Upsample(scale_factor=2, mode="nearest")]
            if self.resample_with_conv:
                upsample.append(Conv2d(curr_chans, curr_chans, 3, 1, 1))
            modules.append(Sequential(*upsample))
        return modules

    def forward(self, x, t, c, c_shape, debug=False):

        t_emb = get_timestep_embedding(t, self.hid_channels)
        t_emb = self.t_embed(t_emb)

        if c is None:
            c = torch.rand(c_shape) * (self.c_value_range[1] - self.c_value_range[0]) + self.c_value_range[0] # if unconditional: make c irrelevant
            c = c.to(x.device)

        assert c.shape[1] == self.c_in_dim, f"ASSERT FAILED: c.shape[-1]={c.shape[-1]} but self.c_in_dim={self.c_in_dim}"
        c_emb = self.c_embed(c)

        # downsample
        
        debug_names = []
        debug_shapes = []
        hs = [self.in_conv(x)]
        debug_names.append(self.in_conv.__class__.__name__+"(x)")
        debug_shapes.append(x.shape)
        debug_shapes.append(hs[-1].shape)
        for i in range(self.levels):
            downsample = self.downsamples[f"level_{i}"]
            for j, layer in enumerate(downsample):
                h = hs[-1]                                                      ### ???
                s = layer.__class__.__name__
                if s == "Sequential":
                    s += '(' + ','.join([l.__class__.__name__ for l in layer]) + ')'
                if j != self.num_res_blocks:                                    ### ???
                    hs.append(layer(h, t_emb=t_emb, c_emb=c_emb))
                    debug_names.append(s+"(h,t_emb,c_emb)")
                else:
                    hs.append(layer(h))                                         ### ???
                    debug_names.append(s+"(h)")
                debug_shapes.append(hs[-1].shape)

        if debug:
            print("## DOWN")
            print(f'[{",".join(debug_names)}]')
            print(f'[{",".join([str(s) for s in debug_shapes])}]')

        # middle
        
        debug_names = []
        debug_shapes = []
        h = self.middle(hs[-1], t_emb=t_emb, c_emb=c_emb)
        s = self.middle.__class__.__name__
        if s == "Sequential":
            s += '(' + ','.join([l.__class__.__name__ for l in self.middle]) + ')'
        debug_names.append(s+"(h,t_emb,c_emb)")
        debug_shapes.append(h.shape)

        if debug:
            print("## MIDDLE")
            print(f'[{",".join(debug_names)}]')
            print(f'[{",".join([str(s) for s in debug_shapes])}]')

        # upsample
        
        debug_names = []
        debug_shapes = []
        for i in range(self.levels-1, -1, -1):
            upsample = self.upsamples[f"level_{i}"]
            for j, layer in enumerate(upsample):
                s_name = layer.__class__.__name__
                if s_name == "Sequential":
                    s_name += '(' + ','.join([l.__class__.__name__ for l in layer]) + ')'
                if j != self.num_res_blocks + 1:                                ### ???
                    s_shape = f"cat[{h.shape},{hs[-1].shape}]"
                    h = layer(torch.cat([h, hs.pop()], dim=1), t_emb=t_emb, c_emb=c_emb)     ### ???
                    debug_names.append(s_name+"(h+hs[-1],t_emb,c_emb)")
                    debug_shapes.append(s_shape + " -> " + str(h.shape))
                else:
                    h = layer(h)                                                ### ???
                    debug_names.append(s_name+"(h)")
                    debug_shapes.append(h.shape)

        h = self.out_conv(h)

        s_name = self.out_conv.__class__.__name__
        if s_name == "Sequential":
            s_name += '(' + ','.join([l.__class__.__name__ for l in self.out_conv]) + ')'
        debug_names.append(s_name+"(h)")

        debug_shapes.append(h.shape)

        if debug:
            print("## UP")
            print(f'[{",".join(debug_names)}]')
            print(f'[{",".join([str(s) for s in debug_shapes])}]')
        return h


if __name__ == "__main__":
    model = UNet(3, 64, 3, (1, 2, 4), 3, (True, True, True))
    print(model)
    out = model(torch.randn(16, 3, 32, 32), t=torch.randint(1000, size=(16, )))
    print(out.shape)
