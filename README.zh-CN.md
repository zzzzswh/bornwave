> English documentation: [README.md](README.md)

# bornwave

PyTorch 实现的改进收敛 Born 级数（CBS）频域声波求解器，支持**任意非均匀速度、密度与吸收**，面向地震勘探 / 声波方程模拟。算法来自 Stanziola, Arridge, Treeby & Cox, *Iterative Born solver for the acoustic Helmholtz equation with heterogeneous sound speed and density* (arXiv:2507.16087)，其核心是 Vettenburg & Vellekoop 的 universal split-preconditioner (arXiv:2207.14222) 作用在一阶声学方程组上。

矩阵自由、无预处理开销，每步迭代只有「逐点乘加 → 3 场批量 FFT → 每 k 点 3×3 einsum → 批量 IFFT → 逐点乘加」，天然适合 GPU 与炮批量。

## 当前状态（v0.2，阶段 0–4）

- 2D 单频求解器：交错网格一阶系统 `[u_x, u_z, p]`，多项式斜坡 sponge 吸收层，FFT 友好网格尺寸（2/3/5/7 小素数积）
- 变密度：`ρ0` 线性插值到半格点（论文式 39），交错相位 `e^{±ik∆/2}` 进入 `(L+I)^{-1}` 符号（附录 B 式 44 的 2D 版）
- 炮批量：源可带任意前导 batch 维，形状 `(B, 3, Nz, Nx)` 一次迭代同推所有炮（算子张量全部共享）
- **多频合成（阶段 4）**：`synthesize_shot` = Ricker 子波 `rfft` 分解 → 有效频带（|W|≥阈值）逐频 Helmholtz 求解 → `irfft` 合成时间域道集
- 复移位 `a1, a2` 用逐分量中位数近似最小包围圆圆心；`λ = max|d−a|/β`，`β=0.95`
- 真残差 `‖y−Ax‖/‖y‖` 用独立装配的前向符号核验（非迭代自证）

### 验证（tests/，全部通过）

| 测试 | 结果 |
|---|---|
| `(L+I) @ (L+I)^{-1} = I` 逐 k 点 | 9e-16 |
| 伴随符号、微分块斜厄米性 | 1e-15 / 0 |
| 均匀介质 vs 解析 Hankel 格林函数（160²内域，10 ppw） | 相对 L2 误差 **8.9e-5**，振幅比 1.0000，相位偏差 0.00° |
| 强对比圆盘（2× 速度，2.5× 密度）声学互易性 | **3.2e-6**（complex64 下机器精度级） |
| 两层模型 + 15 Hz Ricker 道集：直达波窗口 vs 解析 Hankel | 平均 0.075%，最大 **0.18%** |
| 零偏移距反射振幅 vs 法向 Zoeppritz R₀=0.5714 | **0.08%** |
| AVO 曲线（0–600 m 偏移距）vs 流体 Zoeppritz R(θ) | 最大 0.54% |
| 反射时距曲线 / 反演界面深度 | ≤1.3 ms（<1 采样）/ 596.3 vs 596.25 m |

### 实测确定的四个约定（写代码前容易踩的坑）

1. **时间约定是 `+iωt`（H0⁽²⁾ 分支）**，即论文式 (5) 按字面 `+iω` 实现后，恰好等于 numpy/torch FFT 的传递函数约定：多频合成 `d(t) = irfft(W(f)·H(f))` **不需要任何共轭翻转**。
2. **2D 点源归一化就是 `幅值/dx²`**（谱方法下单格点=单位积分 sinc），实测振幅比 1.0000——论文里的 `2c0/dx` 修正是 1D 专用，2D 不需要。
3. **交错网格下 `(L+I)^{-1}` 逐 k 点矩阵不对称**（S 与 S̄ 相位），论文式 (34) 的"伴随=逐元素共轭"只对非交错版成立；交错版伴随符号是逐 k 点**共轭转置**（计算同样免费，见 `test_operator_identity.py`）。
4. **掠射海绵虚象**：炮检线离吸收层太近（如 60 m）时，直达波沿顶部海绵掠射的残余反射与直达波无法分窗，大偏移距误差可达 5%；把测线放到 ~λ_dom 深（150 m）并用 60 点海绵后降到 0.18%。与时间域 FDM 的吸收边界掠射问题同源，做地表观测系统时要留意。

## 快速上手

```python
import torch, numpy as np
from bornwave import CBSSolver2D, point_source_2d

n, dx, f = 256, 3.0, 50.0
omega = 2 * np.pi * f
c0   = torch.full((n, n), 1500.)   # 任意 2D 速度模型
rho0 = torch.full((n, n), 1000.)   # 任意 2D 密度模型

solver = CBSSolver2D(c0, rho0, omega, dx, alpha=0.0,
                     abs_points=40, dtype=torch.complex64, device="cpu")

# 单炮或炮批量：sp 形状 (nz,nx) 或 (B,nz,nx)
sp = torch.stack([point_source_2d(n, n, 20, ix, dx) for ix in (64, 128, 192)])
res = solver.solve(sp=sp, tol=1e-6)

res.p, res.ux, res.uz        # (B, nz, nx)，u 在半格点
res.iterations, res.rel_residual
```

地震 Q 模型：`alpha = alpha_from_Q(omega, c0, Q)`。

### 时间域道集（多频合成）

```python
from bornwave import synthesize_shot
import numpy as np

out = synthesize_shot(c0, rho0, dx,
                      src=(20, 120),            # (iz, ix)
                      rec_z=20, rec_x=np.arange(12, 229, 3),
                      nt=768, dt=2e-3, f0=15.0, # 15 Hz Ricker（可传自定义 wavelet）
                      tol=2e-4, abs_points=60)
out["gather"]        # (nt, nrec) 时间域道集
out["H"]             # 单位源传递函数 (nfreq, nrec)，可复用换子波零成本
```

时间约定已实测对齐 torch/numpy FFT（`+iωt`），合成端不需要共轭。完整两层模型验证（直达波、Zoeppritz AVO、时距曲线）见 `examples/two_layer_ricker.py`。

## 路线图

- [ ] Anderson 2D 流体圆柱散射解析解对比（变密度路径的解析级验证）
- [ ] 频率批量 `(B_freq, B_shot, 3, Nz, Nx)` + 收敛掩码 / 频段分桶调度
- [ ] 隐函数定理 `torch.autograd.Function`：forward 收敛迭代 + backward 伴随解（共轭转置符号已就位），显存 O(1) → FWI
- [x] 多频合成层（v0.2）；剩余：与时间域 FDM（Julia 交错网格+HABC）trace 级交叉验证
- [ ] `torch.compile` 融合逐点链 + CUDA graph 捕获主循环；`(L+I)^{-1}` 存/现算按显存自动切换
- [ ] Osnabrugge 2021 超薄吸收边界层
- [ ] 3D（4 场，每步 8 个 FFT）

## 依赖

`torch`（CPU 或 CUDA 均可）、`numpy`；验证与示例另
## 引擎 API(v0.3):一行正演 → 道集 / 波场 / 动画

```python
import numpy as np
from bornwave import acoustic2d, trace_norm, plot_shot, plot_wavefield_video

nz, nx = 300, 400                       # 注意:数组是 (nz, nx),即 vp[z, x]
dh, dt, nt, f0 = 10.0, 1e-3, 2000, 15.0
vp  = np.full((nz, nx), 2500.); vp[180:]  = 3200.
rho = np.full((nz, nx), 2000.); rho[180:] = 2300.

res = acoustic2d(vp, rho, dh, dt, nt, f0,
                 sx=[nx // 2], sz=[10],           # 多炮传等长列表,一批同解
                 rx=np.arange(0, nx, 2), rz=10,
                 nbc=60, tol=2e-4, snap_interval=25)

res.seis_p / res.seis_vx / res.seis_vz    # (nt, nrec) 时间域记录
res.snaps                                  # (nsnap, nz, nx) 压力波场快照
res.H_p                                    # 单位源传递函数,换子波零成本重合成
res.stats.kernel_time_s                    # 计时 / 迭代诊断

plot_shot(trace_norm(res.seis_p), "shot_p.png", dt=dt)
plot_wavefield_video(res.snaps, "wavefield.mp4", fps=12, dh=dh,
                     snap_times=res.snap_times, adaptive_clims=True, model=vp)
```

实现与性能:

- **频率 × 炮联合批量**(`bornwave/engine.py`):工作张量 `(F, B, 3, Nz, Nx)`,
  算子符号跨炮共享;收敛的频点即时定稿并压缩移除。收敛判据与
  `CBSFreqBatch2D` 完全一致(最后一步相对增量,每 `check_every` 步检查)。
- **CUDA Graph**:GPU 上把 `check_every-1` 次迭代捕获为单次 replay,消除
  每迭代 ~15 个 kernel 的 launch 延迟(本方法小网格 × 上千次迭代的场景下
  这是主要开销)。压缩后自动重捕获;捕获失败自动回退 eager,结果不变。
- **波场快照零额外求解**:快照 = 保留频带的全场传递函数在快照时刻的
  精确逆变换采样(`timesynth.time_slices_from_band`,与 `np.fft.irfft`
  机器精度一致,见 `tests/test_timesynth.py`),从不物化 `(nt, nz, nx)`。
- **约束**:真空单元(vp=0)不可表示——CBS 收缩性要求介质对比有界;
  平坦压力自由表面拟用镜像法实现(路线图)。`nbc` 是海绵厚度,40–60 足够。

**拿到代码先跑** `python tests/test_engine_api.py`:新引擎 vs 已验证
`CBSFreqBatch2D` 的场级一致性、炮批量 vs 逐炮、快照 vs 道集共采样点
一致性、直达波时差运动学,全部 complex128 紧公差。之后跑
`python examples/demo_engine.py` 出图出动画。
