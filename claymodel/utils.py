import torch


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# h、w：Patch 网格的行数、列数
# dim：每个 Patch 的位置编码维度
# gsd：地面采样距离，反映影像的空间分辨率
# temperature：控制编码频率的参数
# dtype：返回张量的数据类型
def posemb_sincos_2d_with_gsd(
    h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32
):
    # 生成每个 Patch 的坐标；meshgrid 将它们展开为二维坐标网格
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    gsd = gsd.to(x.device)
    # 生成归一化的编号
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    # 计算各维度的频率
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)
    # 行坐标乘上各个频率
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    # 位置编码
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(waves, dim, temperature: int = 10000, dtype=torch.float32):
    assert (
        dim % 2 == 0
    ), "Feature dimension must be a multiple of 2 for sincos embedding"
    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)
