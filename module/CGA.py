""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

# 损失率
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

"""
将图像划分为一个个的windows
"""
def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape # 获取图像信息
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]，划分多少个小window，每个window大小是7x7
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

"""
将一个个window还原为原来的图像
"""
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module): # 降采样，在第1个stage之前用
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W # 得到下采样后的高度和宽度，此时x:[B,HW,C]

class Up_dim(nn.Module): # 降采样，在第1个stage之前用
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=1, stride=1)

    def forward(self, x, H, W):
        H, W = H, W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B,H,W,C).permute(0,3,1,2)
        x = self.proj(x)
        return x # 得到下采样后的高度和宽度，此时x:[B,HW,C]


class PatchMerging(nn.Module): # 每个stage后经过patch merging层进行下采样。通道数翻倍，宽度、高度减半，降采样，在第二、三、四个stage前用
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) # 输入4倍的dim，输出2倍的dim
        self.norm = norm_layer(4 * dim) # 输入的dim是原始输入dim的4倍

    def forward(self, x, H, W): # 输入的数据，传入之前记录的H, W
        """
        x: B, H*W, C
        """
        B, L, C = x.shape # 只知道H * W 的乘积，不知道具体H, W，来自PatchEmbed的x，所以x为[B, HW, C]
        assert L == H * W, "input feature has wrong size" # 如果L 不等于 H * W ，则报错

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding（下采样2倍）
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            # 下面是从x结构的后面开始计算的，0,0表示C的两个参数，0,W%2表示W的两个参数，0,,H%2表示H的两个参数
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 对于所有窗口，在相同位置处进行拼接
        """
            ##############################
            #  蓝  #   黄  #  蓝  #   黄  #
            ##############################
            #  绿  #   红  #  绿  #   红  #
            ##############################
            #  蓝  #   黄  #  蓝  #   黄  #
            ##############################
            #  绿  #   红  #  绿  #   红  #
            ##############################
        """
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C] #0行，0列开始，步长为2，拼接到一起，Batch维度取所有值，高度，宽度方向以2为间隔进行采样。表示 蓝色[0,0],[0,2],[2,0],[2,2]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C] #1行，0列开始，步长为2，拼接到一起 表示绿色
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] #表示黄色
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C] # 表示红色
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C] ， 拼接后通道数翻4倍
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C] # 将H/2 * W/2相乘

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]  # 将channel从4C调整为2C

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw] # 计算以每个当前点为参考点，其他点的相对位置
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]  # 变换位置，方便计算[[[0,0],[0,-1],[-1,0]，[-1,-1]],[[0,1]，[0,0]，[-1,1]，[-1,0]],[...],[...]]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0 行标加 M-1
        relative_coords[:, :, 1] += self.window_size[1] - 1 # 列标加M-1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 行标乘2M-1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw] 行标和列标相加
        self.register_buffer("relative_position_index", relative_position_index) # 将位置序列放入缓存中，需要用的时候取出来。

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 为什么会×3，这是因为在forward，分离q、k、v，需要3.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        x_type = x.type()
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # mask在这里起作用，对与mask-100，在使用softmax计算的时候，得到值很小。
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        # x.contiguous()
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size # SW-MSA移动的步数
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size" # 如果shift_size 等于 0 则是W-MSA，否则是SW-MSA

        # self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # shortcut = x
        # x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape # 补齐padding后的x的实际高，宽

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # 根据移动步数，重新分割图像
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # shifted_x 加入 padding 后新的特征图
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift 恢复原状
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        # print("asfaafa",x.type())

        # # FFN
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2 # 移动大小等于 窗口大小除以2向下取整，因为取7，所以整个特征图像右下角移动3步

        # build blocks，堆叠swin transforms的block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size, # W-MSA 和 SW-MAS 成对出现
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # # patch merging layer，这是下个stage的patch merging
        # if downsample is not None:
        #     self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        # else:
        #     self.downsample = None

    """
    创建一个自注意机制的掩码，这个函数的输入参数包括 x（输入特征图）以及 H 和 W （输入特征图的高度和宽度）。该函数的作用是创建一个用于自注意力计算的掩码。
    掩码的目的是在计算自注意力时控制哪些位置信息应该被考虑，哪些位置的信息应该被忽略。
    """
    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        """
        在函数中，首先通过对输入的高度和宽度进行取整操作，确保 Hp 和 Wp 分别是 self.window_size 的整数倍。
        这是因为自注意力通常被分割成小的窗口（windows）来计算，确保窗口的大小是 self.window_size 的整数倍有助于计算的规整性。
        ***为了支持多尺度设定的***
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        """
        这部分计算将输入特征图的高度除以窗口大小并向上取整，以确保窗口能够完整覆盖输入特征图的高度。然后，乘以 self.window_size，以获取输出特征图的高度 Hp，确保它是窗口大小的整数倍。
        """
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # 创建一个名为 img_mask 的零矩阵，其形状为 (1, Hp, Wp, 1)，这个矩阵将用于存储掩码信息.
        # 高度方向切片，window=7，shift=7/2向下取整为 3
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices: # 遍历每个切片，对第一个区域赋予数值0，第二个区域赋值为1，依次类推。
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # window_partition 函数将img_mask 分割成多个窗口，每个窗口的大小为self.window_size，从而得到一个形状为[nW, Mh, Mw, 1]其中 nW 是窗口的数量，Mh 和 Mw 是窗口在高度和宽度上的尺寸。
        mask_windows = window_partition(img_mask, self.window_size)  # [B*nW, Mh, Mw, C]，因为B为1，所以[nW, Mh, Mw, C]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        # 创建一个 attn_mask 张量，它表示窗口之间的相对位置关系。这个张量的形状为 [nW, 1, Mh * Mw] 减去 [nW, Mh * Mw, 1]，最终形成一个形状为 [nW, Mh * Mw, Mh * Mw] 的张量。
        # 这个张量将用于计算自注意力时对窗口之间的相互作用进行加权。最后，将 attn_mask 中不等于零的元素替换为 -100.0，将等于零的元素替换为 0.0，从而得到最终的注意力掩码。
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask # [nW, Mh * Mw, Mh* Mw]

    def forward(self, x, H, W):
        # x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw] # 通过调用 create_mask 函数创建注意力掩码
        for blk in self.blocks: # 将输入数据传递给模型的块（blocks）
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        # if self.downsample is not None: # 如果定义了下采样（downsample）操作，将对输出进行下采样，并更新输出的高度和宽度。
        #     x = self.downsample(x, H, W)
        #     H, W = (H + 1) // 2, (W + 1) // 2

        return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96，经过Linear Embedding 得到的特征的通道数，此后每个阶段输出特征的通道数是当前的倍数。
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, dim, patch_size=4, in_chans=3,
                 embed_dim=96, depths=2, num_heads=3,
                 window_size=7, drop_rate=0.,out_chans=64):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim # 经过Linear Embedding得到的特征图的通道数
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # self.layers = nn.ModuleList()
        self.layers = BasicLayer(
                    dim=dim,
                   # patch_size=patch_size,
                   # in_chans=in_chans,
                    depth=depths,
                   window_size=window_size,
                   num_heads=num_heads)
        self.Up_dim = Up_dim(in_c=embed_dim,embed_dim=out_chans)

    def forward(self, x):
        # x: x的表示是[B, L, C]
        x, H, W = self.patch_embed(x) # 对x下采样4倍，H/4,W/4，输出为96或128或196
        x = self.pos_drop(x) # x在dropout层按照一定比例丢失

        x = self.layers(x, H, W)
        x = self.Up_dim(x, H, W)
        return x

def CGA(dim=96,patch_size = 4, in_chans=3,embed_dim=96,window_size=7,depth=2,num_heads=3,out_chans=64):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    # split image into non-overlapping patches
    model = SwinTransformer(
                            dim=dim,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            embed_dim=embed_dim,
                            window_size=window_size,
                            depths=depth,
                            num_heads=num_heads,
                            out_chans=out_chans
                            )
    return model

# input = torch.rand(1,6,224,224)
# model = CGA(dim=192,embed_dim=192,in_chans=6)
# print(model)
# print(model(input))