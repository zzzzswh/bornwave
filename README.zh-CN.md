# bornwave

[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

> English documentation: [README.md](README.md)

无需时间步进的 GPU 声波模拟。

`bornwave` 求解二维声波 Helmholtz 方程，支持**任意非均匀速度、密度与吸收**。
它是 Stanziola, Arridge, Treeby & Cox（JASA, 2026）收敛 Born 级数（CBS）
求解器的矩阵自由 PyTorch 实现，并在其基础上扩展了频率 × 炮联合批量、
CUDA Graph 执行、带限波场精确合成，以及基于伴随状态的自动微分接口。
一次调用即可得到炮集记录、完整波场动画与 FWI 梯度。

名字里虽然有 Born，但这是**全波**求解器：这里的 "Born" 指迭代级数的形式，
而非一阶 Born 近似。收敛后的解在实测真残差意义下满足非均匀 Helmholtz 方程组，
包含全部层间多次波与绕射。

<p align="center">
  <img src="docs/demo_wavefield.gif" width="70%" alt="两层模型含低速透镜体的压力波场动画"/>
</p>
<p align="center">
  <img src="docs/demo_shot_p.png" width="45%" alt="道归一化压力炮集记录"/>
  <br/>
  <em>两层模型 + 低速透镜体：压力波场动画与炮集记录，
  均由 <code>examples/demo_engine.py</code> 一次调用产生。</em>
</p>

---

## 安装

需要 Python ≥ 3.10、PyTorch ≥ 2.4（CUDA 可选，但强烈建议）、以及 NumPy、
SciPy、Matplotlib。导出 MP4 需要 `ffmpeg`；没有 `ffmpeg` 时自动回退为 GIF。

```bash
git clone https://github.com/zzzzswh/bornwave.git
cd bornwave
uv sync                 # 或: pip install -e .
```

拿到代码请先跑这个——它把新引擎与参考求解器做逐场交叉验证，并做端到端一致性
检查（CPU 上约 1–2 分钟）：

```bash
uv run tests/test_engine_api.py
```

然后复现上面的图，以及完整的物理验证套件：

```bash
uv run examples/demo_engine.py        # 记录 + 波场动画
uv run examples/two_layer_ricker.py   # 解析验证（Hankel、Zoeppritz、时距曲线）
uv run examples/fwi_gradient.py       # 单频 FWI 梯度
```

## 快速上手

```python
import numpy as np
from bornwave import acoustic2d, trace_norm, plot_shot, plot_wavefield_video

nz, nx = 300, 400                        # 数组是 (nz, nx)，即 vp[z, x]
dh, dt, nt, f0 = 10.0, 1e-3, 2000, 15.0

vp  = np.full((nz, nx), 2500.0); vp[180:]  = 3200.0
rho = np.full((nz, nx), 2000.0); rho[180:] = 2300.0

res = acoustic2d(
    vp, rho, dh, dt, nt, f0,
    sx=[nx // 2], sz=[10],               # 多炮传等长列表，
    rx=np.arange(0, nx, 2), rz=10,       # 一批同时求解
    nbc=60, snap_interval=25,
)

res.seis_p, res.seis_vx, res.seis_vz     # (nt, nrec) 时间域记录
res.snaps                                # (nsnap, nz, nx) 波场快照
res.H_p                                  # 单位源传递函数
res.resynthesize(other_wavelet)          # 零成本更换子波
res.stats.kernel_time_s                  # 计时与迭代诊断

plot_shot(trace_norm(res.seis_p), "shot.png", dt=dt)
plot_wavefield_video(res.snaps, "wavefield.mp4", fps=12, dh=dh,
                     snap_times=res.snap_times, adaptive_clims=True, model=vp)
```

## 方法

### 控制方程

求解对象是频域一阶声学方程组，离散在交错 Fourier 网格上，场分量顺序为
$[u_x, u_z, p]$（$u_x$ 沿 $x$ 偏移 $+\Delta/2$，$u_z$ 沿 $z$ 偏移
$+\Delta/2$，$p$ 位于整格点）：

$$
\begin{pmatrix}
\rho_0^{+}\,(i\omega + \gamma^{+}) & \nabla^{+} \\
\nabla^{-}\cdot & \dfrac{i\omega + \gamma}{\rho_0 c^2}
\end{pmatrix}
\begin{pmatrix}\mathbf{u}\\ p\end{pmatrix}
=
\begin{pmatrix}\hat{s}_u\\ \hat{s}_p\end{pmatrix}
$$

上标 $\pm$ 表示前向/后向交错，$\rho_0^{+}$ 是线性插值到半格点的密度。
自由空间辐射条件由吸收项 $\gamma$ 施加——它是一条多项式斜坡，可在整格点与
半格点上解析求值。吸收通过复数声速平方进入方程：

$$
c^2 = \frac{c_0^2}{1 - 2i\alpha c_0/\omega}
\qquad\text{或对常 } Q \text{ 模型}\qquad
c^2 = \frac{c_0^2}{1 - i/Q},
$$

后者与频变的 $\alpha(\omega) = \omega/(2c_0Q)$ 完全等价，但装配时与频率无关。

### 分裂预处理与不动点迭代

把方程组写成 $D w = \hat{s}$，其中 $D = \mathrm{Diag} + \mathcal{L}$，
微分算子块 $\mathcal{L}$ 与介质无关，介质**只**通过三个对角场进入。
每个对角块先做复移位 $a$（用逐分量中位数近似最小包围圆圆心，
即论文 Algorithm 1 的廉价做法），再按 $\lambda = \max|d - a| / \beta$
（$\beta = 0.95$）缩放，得到散射势 $V = (\mathrm{Diag} - a)/\lambda$，
满足 $\|V\| \le \beta < 1$。在缩放变量 $x = C^{1/2} w$、
$y = C^{-1/2}\hat{s}$、
$C^{1/2} = \mathrm{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_1}, \sqrt{\lambda_2})$
下，方程组化为 $Ax = y$（$A = L + V$），CBS 迭代为

$$
x \leftarrow x + \nu\,B\left[(L+I)^{-1}(Bx + y) - x\right],
\qquad B = I - V,\quad \nu = 0.9 .
$$

保证收敛的是压缩性 $\|V\| < 1$：介质对比度可以任意大，只要有界即可。

### 单步迭代的代价

$(L+I)^{-1}$ 在 Fourier 域是对角的——每个波数一个 $3\times3$ 矩阵。
记 $\mu = |k|^2 + (a_1+\lambda_1)(a_2+\lambda_2)$，交错相位
$S_e = e^{+i k_e \Delta/2}$，则一次迭代只有：

> 逐点乘加 → 一次批量 FFT → 展开的逐 $k$ $3\times3$ 乘法 → IFFT → 逐点乘加

没有矩阵装配，没有分解，没有内层求解器，也没有任何随炮数增长的开销：
所有算子张量在炮批量间共享，这正是"频域里加炮几乎免费"的来源——
增加一炮的代价基本只有额外的场内存。

那个 $3\times3$ 乘法是刻意手工展开成逐元素运算的：等价的 `einsum` 在 CUDA 上
会降级为 `permute + bmm`，每次迭代都要复制整个符号张量，反而成为主要开销。

### 从频域到时间域

子波用 `rfft` 分解，保留幅度高于阈值的频点，按相邻频率分块求解。
记录由 $d(t) = \mathrm{irfft}(W \cdot H)$ 得到；波场快照则是对保存的全场
传递函数做**精确的带限逆变换采样**，因此从不物化 $(n_t, n_z, n_x)$ 数据立方。
由于传递函数 $H$ 已经保存，换一个子波重新合成只需一次 FFT，无需重新求解。

## 特性

- **任意非均匀模型**——速度、密度（线性插值到交错半格点）、以及 Np/m 或
  常 $Q$ 形式的吸收。
- **空间谱精度**——Fourier 拟谱离散，不存在会累积的数值频散；验证套件跑在
  每波长 10 点，理论下限是 2 点。
- **频率 × 炮联合批量**——引擎迭代单个 `(F, B, 3, Nz, Nx)` 张量，
  已收敛的频点即时定稿并从工作张量中压缩移除。
- **CUDA Graph**——在本方法的网格规模与迭代次数下，墙钟时间由 kernel 启动
  延迟主导，而不是浮点运算。不动点主循环被捕获为单次 replay；每次压缩后
  自动重新捕获；捕获失败则带警告回退 eager 执行，结果完全一致。
- **波场动画不需要额外求解**——快照与记录来自同一组传递函数，
  与 `np.fft.irfft` 达到机器精度一致。
- **可微**——`solve_helmholtz` 用隐函数定理实现伴随状态梯度：结果精确，
  且显存开销与迭代次数无关（$O(1)$）。

## API

### `acoustic2d`——一行正演

| 参数 | 含义 |
|---|---|
| `vp`, `rho` | `(nz, nx)` 速度 [m/s] 与密度 [kg/m³]，必须严格为正；`rho` 可传标量。 |
| `dh`, `dt`, `nt` | 网格间距 [m]、采样间隔 [s]、采样点数。 |
| `f0` | Ricker 主频 [Hz]；传了 `wavelet` 时忽略。 |
| `sx`, `sz` | 炮点 x/z 网格索引，标量或等长序列，作为一批求解。 |
| `rx`, `rz` | 检波点 x/z 网格索引；`rz` 可为标量并自动广播。 |
| `alpha` / `Q` | 吸收系数 [Np/m]，或常 $Q$ 品质因子，二者互斥。 |
| `nbc` | 海绵层厚度（格点数），40–60 足够。它是多项式 $\gamma$ 斜坡，不是有限差分边界，不需要 FD 那种上百格的宽度。 |
| `tol` | 相对增量停止判据。经验值 `2e-4` 对应约 0.1–1 % 的振幅精度（见[验证](#验证)）。 |
| `freq_batch` | 每块联合求解的频点数，是内存/吞吐的主要旋钮。 |
| `snap_interval` | 每隔多少个时间采样保存一次完整压力波场。 |
| `cuda_graph` | `True` / `False` / `"auto"`。 |

返回 `AcousticResult`，包含 `seis_p`、`seis_vx`、`seis_vz`、`snaps`、
`snap_times`、传递函数 `H_p`、逐频 `iterations` 与 `residuals`、
`stats` 诊断命名空间，以及 `resynthesize(wavelet)`。

注意 `vx`/`vz` 记录取的是检波点格子上的交错场 $u_x$、$u_z$；
在地震尺度的网格间距下，半格偏移远小于一个波长。

### 底层接口

```python
from bornwave import CBSSolver2D, CBSFreqShotBatch2D, synthesize_shot, solve_helmholtz
```

| 对象 | 用途 |
|---|---|
| `CBSSolver2D` | 单频、炮批量。参考实现。 |
| `CBSFreqBatch2D` | 频率批量 + 收敛压缩。 |
| `CBSFreqShotBatch2D` | 频率 × 炮联合批量，CUDA Graph 加速，`acoustic2d` 的底座。 |
| `synthesize_shot` | 子波 → 频带 → 时间域道集，不经过引擎层。 |
| `solve_helmholtz` | 可微单频求解，对 $c_0$、$\rho_0$、$\alpha$ 与源均可求梯度。 |

### 可微求解

`solve_helmholtz` 是基于隐函数定理的 `torch.autograd.Function`。
反向传播只需在同一套 CBS 机制上做一次伴随求解（$V \to \bar V$，
符号换成逐 $k$ 共轭转置），梯度是逐点零延迟互相关，因此只需保存正演解。
预处理器内部（移位、缩放、符号）由 **detach** 后的对角场构造：
收敛解并不依赖它们，所以这个梯度是精确的，不是近似。
`examples/fwi_gradient.py` 给出一个三炮单频 FWI 梯度，
能把初始模型里并不存在的界面成像出来。

## 验证

每一层都用它自己生成不了的东西来检验——算子恒等式、解析格林函数、
平面波反射理论、互易性，以及不同实现之间的交叉验证。

| 检验项 | 参照 | 结果 | 脚本 |
|---|---|---|---|
| $(L+I)(L+I)^{-1} = I$ 逐波数 | 精确恒等式 | 9.0e-16 | `test_operator_identity.py` |
| 伴随符号；微分块斜厄米性 | 精确恒等式 | 1.4e-15 / 0 | `test_operator_identity.py` |
| 均匀介质，每波长 10 点 | 解析二维 Hankel 格林函数 | 相对 L2 **8.9e-5**，振幅比 1.0000，相位 0.00° | `test_homogeneous_hankel.py` |
| 强对比圆盘（2× 速度、2.5× 密度） | 声学互易性 | **3.2e-6** | `test_heterogeneous_reciprocity.py` |
| 两层模型 + 15 Hz Ricker，直达波窗口 | 带限解析 Hankel | 平均 0.075 %，最大 **0.18 %** | `examples/two_layer_ricker.py` |
| 零偏移距反射振幅 | 流体 Zoeppritz $R_0 = 0.5714$ | 0.5710（**0.08 %**） | `examples/two_layer_ricker.py` |
| AVO 曲线，偏移距 ≤ 600 m | 流体 Zoeppritz $R(\theta)$ | 平均 0.13 %，最大 0.54 % | `examples/two_layer_ricker.py` |
| 反射时距曲线 / 反演界面深度 | 射线理论，$h = 596.25$ m | ≤ 1.3 ms（< 1 个采样）；拟合 596.3 m | `examples/two_layer_ricker.py` |
| 引擎（频率 × 炮批量） | `CBSFreqBatch2D` 参考求解器 | 2.6e-7 | `test_engine_api.py` |
| 炮批量 | 同样的炮逐个求解 | 0.0 | `test_engine_api.py` |
| 波场快照在共同采样点上 | 检波点记录 | 3e-16 | `test_engine_api.py` |
| 检波点间直达波时差 | 偏移距 / $c$ | 精确（50 / 50 采样） | `test_engine_api.py` |
| 带限时间切片合成 | `np.fft.irfft` | 机器精度 | `test_timesynth.py` |
| 频率批量 | 串行 `CBSSolver2D` | 移位/缩放精确一致；$H$ 达迭代精度 | `test_freq_batch.py` |
| 对 $(c_0, \rho_0, \alpha, s)$ 的自动微分 | `torch.autograd.gradcheck` + 方向有限差分 | 通过，相对偏差 < 3e-5 | `test_autograd.py` |

上表中求解器的残差都是**真残差** $\|y - Ax\|/\|y\|$，
由独立装配的前向符号计算，不是迭代自身增量的自证。

完整实测日志见 `tests/test-log-260726.txt`。

## 性能

基准问题——`examples/demo_engine.py`：

| | |
|---|---|
| 网格 | 300 × 400，填充后 420 × 525 |
| 时间采样 | 2000 |
| 频点 | 95 个，覆盖 0.5–47.5 Hz |
| CBS 迭代 | 合计 76,456 次 |
| Kernel 时间 | **27 s**，CUDA Graph 开启 |
| 硬件 | 单卡 CUDA GPU（`Tesla V100-PCIE-32GB`） |

加炮共享全部算子张量，除额外的场内存外几乎没有代价。
每块的工作集大小在启动时打印，由 `freq_batch` 控制；
迭代次数随频率单调增长，所以把相邻频点分在同一块里，压缩浪费很小。

## 实现约定（改内部之前请先读）

以下四点是论文没有写明、或在交错网格上与论文不同的地方。
每一条都由实测确定，并由测试锁定。

1. **时间约定是 $+i\omega t$**（$H_0^{(2)}$ 分支）。按字面实现后，
   它恰好与 NumPy/PyTorch 的 FFT 传递函数约定一致，
   因此合成就是 $d(t) = \mathrm{irfft}(W \cdot H)$，**任何地方都不需要共轭**。

2. **二维点源归一化就是 幅值/$\Delta^2$。** 谱方法下单个格点代表一个单位积分的
   带限 sinc；论文里的 $2c_0/\Delta$ 修正是一维专用的。
   与解析 Hankel 解对比，实测振幅比为 1.0000。

3. **交错网格下逐 $k$ 的 $(L+I)^{-1}$ 矩阵不对称**——交错相位
   $e^{\pm ik\Delta/2}$ 破坏了对称性。伴随符号是逐 $k$ 的**共轭转置**，
   而不是逐元素共轭（后者只对非交错版本成立）。两者计算代价相同。

4. **掠射海绵虚像。** 海绵吸收对近水平传播的能量效率很低。
   炮检点要离吸收层大约一个主波长以上；更近的话，沿海绵掠射的残余反射
   与直达波无法分窗，大偏移距误差可达百分之几。
   这与时间域有限差分吸收边界的掠射问题是同一回事，
   设计地表观测系统时需要留意。

## 局限

- **时间域绕卷（wraparound）。** 频率采样 $\Delta f = 1/(n_t \Delta t)$ 使合成
  响应以 $T = n_t \Delta t$ 为周期：任何在 $t = T$ 仍在振荡的尾波会混叠回
  $t = 0$，表现为**在震源激发之前就出现能量**。关心晚至能量时请增大 `nt`，
  或做加窗/衰减处理。迭代求解的残余噪声本底同样是非因果的
  （沿时间均匀分布），其量级由 `tol` 决定。
- **暂不支持自由表面。** 真空单元（`vp = 0`）在构造上就落在 CBS 收敛域之外，
  因为压缩性要求介质对比有界。四边都是吸收海绵，所以没有表面多次波；
  层间多次波则完整保留。
- **仅二维**，且只支持单一均匀网格间距。

## 仓库结构

```
bornwave/
  solver.py      CBSSolver2D —— 单频、炮批量
  multifreq.py   CBSFreqBatch2D —— 频率批量 + 收敛压缩
  engine.py      CBSFreqShotBatch2D —— 频率 × 炮批量 + CUDA Graph
  api.py         acoustic2d —— 一行正演入口
  autograd.py    可微求解（隐函数定理 / 伴随）
  operators.py   (L+I)^-1 与 (L+I) 的逐 k Fourier 符号（交错版）
  grid.py        FFT 友好尺寸、海绵剖面、交错平均
  synthesis.py   Ricker 子波、频带选择、逐频合成
  timesynth.py   频带谱 → 时间切片（不依赖 torch，机器精度）
  viz.py         炮集绘图、波场动画（不依赖 torch）
  analytic.py    Hankel 格林函数、流体 Zoeppritz（仅用于验证）
examples/        demo_engine.py, two_layer_ricker.py, fwi_gradient.py
tests/           验证套件 + 实测日志
```

## 路线图

- 镜像法自由表面
- 复频率阻尼，抑制时间域绕卷
- Anderson 流体圆柱散射解析解，用于变密度路径的解析级基准
- 基于真残差的逐频自适应停止判据
- 频段分桶调度
- Osnabrugge (2021) 超薄吸收边界层
- 三维（4 个场，每步 8 次 FFT）

## 参考文献

本仓库是独立实现。算法归功于以下论文；实现方式、引擎层以及其中的任何 bug
由本仓库负责。

1. A. Stanziola, S. R. Arridge, B. E. Treeby, B. T. Cox,
   *Iterative Born solver for the acoustic Helmholtz equation with
   heterogeneous sound speed and density*,
   J. Acoust. Soc. Am. **159** (2026) 1457–1470.
   [doi:10.1121/10.0042259](https://doi.org/10.1121/10.0042259) ·
   [arXiv:2507.16087](https://arxiv.org/abs/2507.16087)
2. T. Vettenburg, I. M. Vellekoop,
   *A universal matrix-free split preconditioner for the fixed-point iterative
   solution of non-symmetric linear systems*,
   [arXiv:2207.14222](https://arxiv.org/abs/2207.14222) (2022).
3. G. Osnabrugge, S. Leedumrongwatthanakun, I. M. Vellekoop,
   *A convergent Born series for solving the inhomogeneous Helmholtz equation
   in arbitrarily large media*,
   J. Comput. Phys. **322** (2016) 113–124.
   [doi:10.1016/j.jcp.2016.06.034](https://doi.org/10.1016/j.jcp.2016.06.034)

BibTeX 见 [README.md](README.md#references)。

## 许可证

尚未确定。如果你的使用场景需要明确的许可证，请提 issue。