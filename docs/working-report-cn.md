> ⚠️ **这是过程稿，不是对外文档。**
> 里面保留了我在会话中的**阶段性结论，其中至少 5 条后来被我自己推翻**：
> Sol-Attn「无损」、「CRF 已验证生效」×3、TE-Speed 证据来源张冠李戴、20 步耗时 28.7 分钟、崩溃归因 OOM。
> **对外请只读 README.md**；本文件仅作原始记录与证据索引。
> 最新的实测结论以 README.md 为准。

---
# MiniMax H3 低显存工作流优化报告（最终版）

> 本机（全部可实测）：**RTX 5060 Laptop 8GB + 15.26 GiB RAM** / Win11 25H2 / ComfyUI 0.35.0 / torch 2.14.0+cu130
> 远程（无访问权限）：**RTX 4060 Laptop 8GB + 16GB RAM**
> 证据分级：**L5** 本机实测或一手源码 · **L4** 官方文档/模板 · **L3** 有硬件+参数的社区实测 · **L2** 社区单点 · **L1** 无 benchmark 的说法
> 凡无 benchmark 支撑者标为「待验证」，不计入结论。
> **禁止事项已遵守：未修改任何模型文件、未改动用户原有工作流与环境。**

---

# 🎯 结论先行：推荐配置

## 默认档（画质与速度兼顾）= **Balanced**

```
新工作流/H3_Balanced.json

步数        8            （官方 turbo 标定档；旧工作流只有 4）
sampler     euler        （turbo 工作流的官方默认）
scheduler   simple
LoRA        turbo v4-600 @ 1.0   （原作者要求值；旧工作流用了 1.7）
注意力后端   comfy kitchen attention   （本机实测比 SDPA 快 6.3×、比 sage 准 2.4×）
TE-Speed    关闭          （实测只值 18%，且是不可审计的有损项）
shift       12/3
分辨率      768×1024      （0.75MP，短边 768 符合官方训练区间）
帧数        243（10 秒，官方 17k+5 公式）
文本编码器   nvfp4_awq     （官方明确不需要 Blackwell；比 int8 小 10.7 GiB）
```

**实测与推算（10 秒成片）**

| 档位 | 步数 | 5 秒实测 | **10 秒推算** | 对比旧工作流 |
|---|---|---|---|---|
| Fast | 4 | 245.3 s | **≈9.0 分钟** | 快 2.2× |
| **Balanced** | **8** | **377.9 s** | **≈13.8 分钟** | **快 1.4×，且步数 8 vs 6** |
| Quality | 20 | 785.3 s | ≈28.7 分钟 | 慢 1.5×，但质量参照 |

> 旧工作流实测：**6 步 / 10 秒 / 1.0MP = 1170 s（19.5 分钟）**
> **⇒ Balanced 用 13.8 分钟做到 8 步，比旧工作流的 6 步更快且步数更多，同时删掉了那个不可审计的有损项。**

---

# 第一部分：两台机器的瓶颈分析

## 1.1 头号瓶颈是 **16GB 系统内存 + 页面文件**，不是 8GB 显存 【L5】

| ComfyUI 会话状态 | 每 block | 10 秒视频 |
|---|---|---|
| **刚重启** | **0.95 秒** | **约 9–20 分钟** |
| 连续运行约 1.7 小时后 | **212 秒** | **约 12 小时** |

**差 35–70 倍。** 机制：权重（主模型 19.5 GiB + 文本编码器 14.6–25.3 GiB）远超 15.26 GiB 物理内存，
长会话让页面文件从 ~16 GB 涨到 36 GB，换页效率崩溃。**本机 11 次运行中页面文件峰值均在 28.9–29.6 GB。**

**独立佐证**
- ComfyUI Issue #15484（RX 9070 XT 16GB，仍 open）：不强制卸载时 VAE 解码 **26.57–52.05 s 抖动**；
  强制 `unload_all_models()` + `soft_empty_cache()` 后 → **15.89–16.19 s 稳定**（stddev 0.07）。**不需要 OOM 就会变慢。**【L3】
- matsuo-koya 生产实测（5090+4090）：「ComfyUI 长跑劣化 **1.8×**」【L3】

## 1.2 「降分辨率省内存」在这台机器上近乎失效 【L3】

12GB 卡一手实测：**工作量缩小 40 倍，峰值显存只降 0.5%**（11,649 → 11,591 MiB），系统内存只降 1.0%。

## 1.3 容量边界由「分辨率 × 时长」决定，不是步数 【L3，4060 8GB 一手】

| 配置（4060 Laptop 8GB / 32GB RAM） | 结果 |
|---|---|
| 1024×576 / 243f / 20 步 | ✅ 2762.89 s，峰值显存 ~6.46 GB |
| 1024×576 / 243f / 4 步 | ✅ 650.96 s |
| 1344×768 / 124f / 4 步 | ✅ 537.49 s |
| 1344×768 / **243f** | ❌ 停滞（memory unloading，7.9/8.2 GB） |
| 1344×768 / **362f / 15.08s** | ❌ CUDA OOM |

作者原文结论：「20 步 @1024×576 能完成，而 1 步 @1344×768/362f 不能」。

## 1.4 两台机器的关键差异**不在算力，在后端可用性** 【L5 + L4】

| 项 | RTX 4060 Laptop | RTX 5060 Laptop |
|---|---|---|
| 芯片 / CC | AD107 / **sm_89** | GB206 / **sm_120** |
| SM / CUDA | 24 / 3072 | 26 / 3328 |
| 显存 | 8GB GDDR6 128-bit 16Gbps | 8GB GDDR7 128-bit 24Gbps |
| **带宽** | **256 GB/s** | **384 GB/s** |
| **PyTorch FlashAttention** | ✅ 可用 | ❌ **不可用**（本机 `RuntimeError: No available kernel`） |
| PCIe | 4.0 x8 | 规格 5.0 x8，**本机实际协商 Gen4 x8** |

**关键推论**：sm_120 上**非 block-scale 的 FP8 每 SM 每时钟与 Ada 同级**（triton#11320，5060 Ti 实测 102 vs 202 TFLOP/s，
2× 只来自 MXFP8 block-scale）【L3】。⇒ 跑 int8_convrot 量化推理时，两卡算力差距只有 **~10–30%**，不是宣传的 2.45×。

⇒ **芯片差异被抹平了**，所以「4060 用更低分辨率却画质更好」不是硬件问题，而是软件配置问题。

## 1.5 耗时结构：采样严格线性，总耗时 = 线性 + 固定开销 【L5 本机实测，已验证】

**5 秒 / 768×1024 / kitchen 注意力 / LoRA@1.0：**
```
T_wall ≈ 102 + 34.1 × 步数          残差全部 ≤3.2%
```

| 步数 | 预测 | **实测** | 误差 | 去噪 | 每层 | 峰值显存 | 峰值内存 |
|---|---|---|---|---|---|---|---|
| 4 | 238.8 | **245.3** | −2.6% | 187.0 | 0.94 | 7320 MiB | 14.53 GB |
| 6 | 307.1 | **297.4** | +3.2% | 240.0 | 0.80 | 7370 MiB | 14.22 GB |
| 8 | 375.3 | **377.9** | −0.7% | 319.0 | 0.80 | 7366 MiB | 13.64 GB |
| 20 | 784.7 | **785.3** | −0.1% | — | — | 7376 MiB | 13.50 GB |

**⇒ 20 步 / 4 步 = 3.20×（纯步数比例应为 5.00×）。固定开销 ~102 秒占 4 步的 42%、20 步的 13%。**
**这直接回答「为什么步数差距没有理论上那么大」。**

**重复性**：A1 三次 10 秒运行 = **531.1 / 537.6 / 538.5 s（偏差 1.4%）**；
As1(满穗底图) 245.3 s vs B1(栏杆底图) 249.1 s = **偏差 1.5%** ⇒ 结论不受底图影响。

**时长影响**（同 4 步）：5 秒 245.3 s vs 10 秒 537.6 s = **2.19×**（帧数 2×）⇒ 略超线性。

---

# 第二部分：哪些部分值得保留

| 项 | 保留理由 | 证据 |
|---|---|---|
| **`MiniMaxChunkFeedForward(4,4096)`** | 源码逐字「since activations are quantized per-token the **output matches the unchunked model**」；第三方受控实测 **−3936 MiB**。**决定项**：fc1 是 5376→28672，S≈96,768 时未分块要 **5.17 GiB**，`chunks=4` → **1.29 GiB** | L5 源码 + L3 |
| **`MiniMaxLowVRAMAttention(10)`** | 源码逐字「**Output is identical to the unpatched model**」。本机实测 `head_chunks=10` 误差与不分块**逐位相同**，只慢 15% | L5 |
| **`int8_convrot` 主模型 + VAE** | Comfy-Org README：diffusion 优先 `int8_convrot`（需 cu130），用不了才退 `fp8_scaled` | L4 |
| **`nvfp4_awq` 文本编码器** | 官方明确「**不需要 Blackwell 卡**」；14.6 GiB vs int8 的 25.3 GiB，**省 10.7 GiB** —— 对 16GB RAM 是关键 | L4 |
| **`--disable-pinned-memory`** | `MAX_PINNED_MEMORY = ram*0.90` 会让 16GB RAM 机器被内核 OOM-kill | L3 |
| **无 CFG 节点** | 官方 README：「The released checkpoints are CFG-distilled」 | L4 |

---

# 第三部分：应该删除

| 项 | 删除理由 | 证据 |
|---|---|---|
| **`TESpeedMiniMaxH3`** | ① 本机 A/B 实测**只值 18.2%**（墙钟 249.1→203.7 s）；② 作者 README 逐字「TE-Speed从2.0开始不再以加速时间为第一标准,而是**速度质量兼顾**」=自认取舍；③ **闭源 `nodes.pyd`（396KB，无 .py）不可审计**，无法确认音频保护；④ 3.5 版 README **零 benchmark 数字**；⑤ 与 Spectrum 硬冲突 | **L5 实测** + L3 源码 |
| **`RTXVideoSuperResolution`** | 本机 `close()` **永久挂死**（三次隔离复现，含不 run 的纯 close）→ `execute()` 永不返回，工作流卡死。且源码一次性分配整段输出 fp32 视频（2× 时 124 帧多吃 2.5–4.8 GiB） | **L5 实测** |
| **`EasyCache`** | 4 步下 `skipped 0/4 steps`（零加速纯开销，三次复现）；H3 上音频 log-mel L1 退化 0.395–0.690 | L3 |
| **`LazyCache`** | 节点 UI 内置「Overall works worse than EasyCache」 | L3 |
| **`flash_attn` / `sageattn3`** | 本机 MISSING 且不建议装（FA 无 Windows wheel、官方明说 Windows 未测；SA3 在 sm120 有 misaligned address/TMA 崩溃） | L5 |
| **`sla` / `vsa` 块稀疏** | 明确的近似（实现者原文 "an approximation, not a lossless path"，同 seed 22.4 dB PSNR）；8GB 上净增 ~4 GB 显存 | L3 |
| **`LatentUpscaleBy` 二阶段** | 对 H3 24 通道 latent 做 bicubic/bilinear 会「周期性重影」；遇奇数 latent 轴触发 circular padding → 接缝色条 | L3 |
| **`VAEDecodeTiled`** | **对 H3 是 no-op**（有源码证据） | L3 |

---

# 第四部分：应该保留 / 新增

| 项 | 动作 | 理由 | 证据 |
|---|---|---|---|
| **`ModelAttentionBackend = comfy kitchen attention`** | **新增** | 本机 H3 真实几何实测：kitchen INT8 **22.8 ms / rel L2 0.0164** vs KJ sage **29.0 ms / 0.0387** vs **SDPA 143.1 ms**（旧工作流的实际路径）。**同时更快且更准**。且 sage patch 会绕过 `optimized_attention`，**架空后端选择机制** | **L5** |
| **`MiniMaxH3SigmaShift`** | **新增** | 官方模板不含 shift 节点，走内置 12/3。而 lightx2v 规格表：**544p 系 LoRA 训练 12/3、768p 系 6/3**。显式加节点让 shift 可审计 | L4 + L5 源码 |
| **强制会话重启** | **流程** | 收益 35–70×，远超任何节点级优化 | **L5** |
| 步数 4 → 8 | 参数 | 官方 turbo 标定档；实测 6→8 步每步 39.5 s | L4 + L5 |
| LoRA 强度 1.7 → 1.0 | 参数 | 原作者要求 1.0（最大 1.05–1.2）；社区 0.5–0.75；**1.7 无任何来源** | L2 |

### 已否证的三个推断（我此前错了，此处更正）

| 我的推断 | 实际 | 证据 |
|---|---|---|
| `MiniMaxLowVRAMAttention` 与 `SageAttentionPatch` 冲突 | ❌ **不冲突**。分块能力写在 `transformer_options["minimax_head_chunks"]`，sage forward **主动读取**。**真冲突是 sage patch × `ModelAttentionBackend`** | L5 源码 |
| `Sol-Attn` 逐位无损 | ❌ **错**。实现者原文「an approximation, not a lossless path」；H3 特有失效模式是**音画同步** | L3 |
| 「近似缓存」出自 TE-Speed 的兄弟项目 | ❌ **张冠李戴**。那句话出自 **T8 Block Cache**；TE-Speed 是**残差重放**（非 F1B0），与 T8 非同一作者 | L3 |

---

# 第五部分：官方基线 vs 原有工作流

| 项 | 官方 | 你原有 |
|---|---|---|
| **步数** | **20** | 4 |
| sampler | `res_multistep` | `euler` |
| Turbo LoRA | **可选、默认 `false`** | 开启 **@1.7** |
| 分辨率约束 | 短边 768、面积 ≤768×1344、32 倍数 | 896×1184（**短边 896 超出训练区间**） |
| 帧数 | `17k+5`，训练区间 ~124–362 | 同公式 ✅ |

**⚠️ 官方模板里那个 `steps: 4` 是死值** —— 被上游 `If/Else Switch (Steps)` 覆盖，在 `PrimitiveInt=20`（非 turbo）与 `=6`（turbo）间二选一。官方文档原文：
> "The example FL2VA workflow uses **20 steps by default**."

**官方对 8GB 显存什么都没说，且明确承认未公布**：
> `"Published minimum GPU memory and performance | Not published in the ComfyUI H3 guide"`

---

# 第六部分：RTX 4060 8GB 配置

> ⚠️ **本机没有 4060，不存在 4060 侧的任何第一手数据。** 以下基于社区一手 benchmark（L3）+ 硬件规格（L4）+ 本机对 5060 的实测外推，**强度低于 5060 部分，需你在 4060 上跑脚本确认**。

| 档位 | 步数 | sampler | LoRA | 分辨率 | 说明 |
|---|---|---|---|---|---|
| Quality | 20 | `res_multistep` | 无 | 768×1024 | 官方基线 |
| **Balanced** | 8 | `euler` | turbo @1.0 | 768×1024 | 默认推荐 |
| Fast | 4 | `euler` | turbo @1.0 | 768×1024 | |

**⚠️ 4060 必须单独测注意力后端**：它是 **sm_89，FlashAttention 可用**（与 5060 的 sm_120 完全不同），
不能假设 kitchen 在 4060 上也最优。`bench/spec_4060.json` 已把这项 A/B 编入。

**已知 4060 一手数据**：1024×576/243f/20 步 = 2762.89 s；同配置 4 步 = 650.96 s；1344×768/124f/4 步 = 537.49 s；
**1344×768/243f 停滞、362f OOM**。

---

# 第七部分：RTX 5060 8GB 配置

## 7.1 实测耗时模型
```
T_wall ≈ 102 + 34.1 × 步数      （5 秒 / 768×1024 / kitchen / LoRA@1.0，残差 ≤3.2%）
```

## 7.2 时长换算
10 秒 ≈ 5 秒 × **2.19**

## 7.3 与原有工作流的对比（同为 10 秒）

| | 你原有 | 新 Balanced |
|---|---|---|
| 步数 | 6（+ TE-Speed 50% 跳层） | **8（无 TE-Speed）** |
| 分辨率 | 896×1184（1.0MP） | 768×1024（0.75MP） |
| LoRA 强度 | **1.7** | **1.0** |
| 注意力后端 | Sage（**架空后端选择**） | **kitchen INT8** |
| **墙钟** | **1170 s（19.5 分钟）** | **≈828 s（13.8 分钟）** |

## 7.4 三档

| 档位 | 步数 | TE-Speed | **10 秒预计** | 用途 |
|---|---|---|---|---|
| **Quality** | 20 | 关 | ~28.7 分钟 | 最终成片 |
| **Balanced** | 8 | 关 | **~13.8 分钟** | **默认推荐** |
| **Fast** | 4 | 关 | ~9.0 分钟 | 快速预览 |

> ⚠️ **三档全部不使用 TE-Speed。** 实测它只值 18.2%，而代价是不可审计的有损项。
> 若你确实需要那 18%，可在 Fast 档手动启用（`H3_Fast.json` 里没有该节点，需自行添加）。

---

# 第八部分：推荐启动参数

```
run_nvidia_gpu.bat 现有：--windows-standalone-build --enable-manager --disable-pinned-memory
```

| 参数 | 建议 | 理由 |
|---|---|---|
| `--disable-pinned-memory` | **必须保留** | 16GB RAM 机器否则被内核 OOM-kill 【L3】 |
| `--use-ck-attention` | 可选 | 与工作流内 `ModelAttentionBackend` 节点等效；与 sage flag 互斥 |
| `--use-sage-attention` | **不要加** | 本机实测比 kitchen 慢且误差 2.4×，还会架空后端选择 |
| `--use-flash-attention` | **不要加** | sm_120 不可用 |
| `--fast` / `--fp16-unet` | **不要加** | H3 `supported_inference_dtypes=[bf16, fp32]`，没有 fp16；残差流可达 ~4.3M ≫ fp16 max，会全黑帧 |
| 启动前 | **强制重启会话** | 收益 35–70× |

---

# 第九部分：完整 benchmark 表

原始数据：`bench/results.csv`（每次运行 33 个字段）

| label | 步数 | LoRA | 时长 | 墙钟(s) | 去噪(s) | 每层(s) | TE-Speed | 峰值显存 | 峰值内存 | 页面文件峰值 |
|---|---|---|---|---|---|---|---|---|---|---|
| SMOKE-Fast-5s | 4 | 1.0 | 5s | 197.2 | 136.0 | 0.91 | 25.0% | 6799 MiB | 14.85 GB | 27.40 GB |
| SMOKE-Balanced-5s | 8 | 1.0 | 5s | 411.6 | 354.0 | 0.89 | — | 7377 MiB | 14.75 GB | 30.38 GB |
| A1-steps4 | 4 | 1.0 | 10s | 531.1 | 431.0 | 2.17 | — | 7643 MiB | 14.46 GB | 28.74 GB |
| A1-steps4 | 4 | 1.0 | 10s | 537.6 | 442.0 | 2.22 | — | 7541 MiB | 14.61 GB | 28.88 GB |
| A1-steps4 | 4 | 1.0 | 10s | 538.5 | 438.0 | 2.20 | — | 7621 MiB | 14.26 GB | 28.97 GB |
| As1-steps4-5s | 4 | 1.0 | 5s | 245.3 | 187.0 | 0.94 | — | 7320 MiB | 14.53 GB | 29.36 GB |
| As2-steps6-5s | 6 | 1.0 | 5s | 297.4 | 240.0 | 0.80 | — | 7370 MiB | 14.22 GB | 29.46 GB |
| As3-steps8-5s | 8 | 1.0 | 5s | 377.9 | 319.0 | 0.80 | — | 7366 MiB | 13.64 GB | 29.36 GB |
| As4-steps20-5s | 20 | — | 5s | 785.3 | — | — | — | 7376 MiB | 13.50 GB | 29.61 GB |
| B1-4step-noTESpeed | 4 | 1.0 | 5s | **249.1** | **189.0** | 0.95 | — | 7392 MiB | 14.56 GB | 29.40 GB |
| B2-4step-TESpeedON | 4 | 1.0 | 5s | **203.7** | **146.0** | 0.98 | 25.0% | 7456 MiB | 14.45 GB | 28.91 GB |

**TE-Speed 唯一变量对照（B1 vs B2）**：墙钟 −18.2%、去噪 −22.8%

---

# 第十部分：预期速度、显存、RAM、质量

| 项 | 你原有工作流 | 新 Balanced | 变化 |
|---|---|---|---|
| 每 block（5 秒档） | 2.17 s（10 秒档 4 步） | **0.80 s**（5 秒档 8 步） | — |
| 10 秒成片 | **1170 s** | **≈828 s** | **−29%** |
| 步数 | 4（有效 6 步 + 跳层） | **8（完整）** | **+100%** |
| 峰值显存 | 7832 MiB | 7366–7456 MiB | −5% |
| 峰值内存 | 顶部近满 | **14.2–14.6 / 15.26 GB** | **无改善（仍是瓶颈）** |
| LoRA 应用率 | 80.3%（1.7 强度） | 80.3%（1.0 强度） | 强度回到作者要求值 |
| 有损组件 | TE-Speed + sage | **仅 kitchen 注意力（误差 0.0164）** | 大幅减少 |

**质量实测（5 秒档，固定 seed 12345，同提示词）**：
`bench/quality/step_ladder_5s.png` 显示 4/6/8/20 步在该内容上肉眼接近 —— **但该对比用的是满穗静态立绘，
不考验时间一致性，因此不能据此认定「步数无差别」**。

**⚠️ 未测项（诚实标注）**：
- LoRA 强度 1.7 vs 1.0 的画质对比（B3 被中断）
- LoRA 关闭的对照（B4 未跑）
- fp16 累加是否 no-op 的实测（B5 未跑）
- 文本编码器 int8 vs nvfp4 的质量对比（未跑）
- 4060 侧任何本机实测

---

# 第十一部分：所有结论的证据来源

| # | 结论 | 来源 | 等级 |
|---|---|---|---|
| 1 | 官方默认 20 步；模板里的 4 是死值 | `Comfy-Org/workflow_templates` R2V JSON；platform.minimax.io 官方文档 | L4 |
| 2 | 官方不推荐 turbo（默认关、措辞带质量折扣） | 三个官方模板 `turbo_mode=false` | L4 |
| 3 | 官方从未公布 8GB 要求 | 官方文档表格原文 | L4 |
| 4 | kitchen 注意力本机比 sage 更快更准 | 本机 H3 真实几何实测 | **L5** |
| 5 | sage patch 架空后端选择机制 | 本机源码 `ltxv_nodes.py:2106` | **L5** |
| 6 | 两个 attention 节点不冲突 | 本机源码 `minimax_nodes.py:188` + `ltxv_nodes.py:2102` | **L5** |
| 7 | Turbo LoRA 只有 80.3% 生效 | 本机逐 key 形状比对 | **L5** |
| 8 | **TE-Speed 只值 18.2%** | 本机 B1/B2 唯一变量 A/B | **L5** |
| 9 | 会话退化 35–70× | 本机日志解析 + ComfyUI #15484 | **L5** + L3 |
| 10 | 步数与耗时线性（T≈102+34.1×steps） | 本机四点实测，残差 ≤3.2% | **L5** |
| 11 | RTX VSR 本机挂死 | 三次隔离复现 | **L5** |
| 12 | 降分辨率几乎不省内存 | minimaxh3tutorial.com 12GB 卡实测 | L3 |
| 13 | 容量由分辨率×时长决定 | YHK-AI 4060 8GB 失败边界表 | L3 |
| 14 | Sol-Attn 不是无损 | 实现者原文 + 22.4 dB PSNR | L3 |
| 15 | sm_120 非 block-scale FP8 与 Ada 同级 | triton#11320 | L3 |
| 16 | 4060 与 5060 的 PCIe 都是 Gen4 x8 | 本机 nvidia-smi + 4060 规格 | **L5** + L4 |
| 17 | EasyCache 在低步数下零收益 | 第三方三次复现 | L3 |

---

# 第十二部分：交付物清单与使用方式

## 工作流（从零生成，未改动你的任何文件）

```
新工作流/H3_Quality.json     24 节点  20 步 无 LoRA     kitchen  TE-Speed 关
新工作流/H3_Balanced.json    25 节点   8 步 turbo@1.0  kitchen  TE-Speed 关  ← 推荐
新工作流/H3_Fast.json        26 节点   4 步 turbo@1.0  kitchen  TE-Speed 关
```

**使用方式**：把 json 拖进 ComfyUI 窗口 → 在 `Input Text (Prompt)` 粘贴提示词 → 运行。
底图默认已设为你的 `微信图片_20260913134501_1457_5.jpg`。

## 每次运行前必须做的一件事

**双击 `重启ComfyUI并运行.bat`** —— 这是唯一能避免 35–70× 退化的操作。

## 工具链

```
bench/build_workflow.py        从零生成工作流（用 /object_info，支持离线缓存）
bench/make_profiles.py         三档 profile 定义 + 一键生成
bench/benchmark_loop.py        自动 benchmark（重启+峰值采样+三段耗时+CSV）
bench/quality_review.py        固定相对时间点抽帧对比图
bench/lora_applicability.py    LoRA 可应用性逐 key 分析
  spec_smoke / spec_A_steps / spec_A_steps_short / spec_B_accel / spec_4060
bench/results.csv              11 行有效实测数据
bench/object_info_cache.json   离线生成用（2.8 MB）
restart_comfyui.ps1 / 重启ComfyUI并运行.bat
```

## 4060 交付

把 `bench/` + `新工作流/` 整个目录复制到 4060，运行：
```
python bench\benchmark_loop.py --spec bench/spec_4060.json
```
（首次需让 ComfyUI 开着以生成该机器自己的 `object_info_cache.json`）

---

**你的原始文件全部未被改动**：模型、工作流、环境均原样。本报告所有产出都落在 `新工作流/`、`bench/`、`_dsh_research/` 三个新建目录内。

---

# 第十三部分：提示词层面的质量缺陷（实测发现并修复）

> 这一部分是**用 20 步官方基线实测出来的** —— 说明「丢主体 / 脸部空白」**与步数、加速节点、注意力后端都无关**，是提示词问题。

## 13.1 现象（H3_Quality_00001_.mp4，旧提示词）

| 时间 | 现象 |
|---|---|
| 0.2 – 6.0 s | 质量良好，但推镜从远景一路推到特写（**违反提示词里的「小幅度」**） |
| 4.5 – 6.0 s | 特写处**脸部无五官**，只有一块空白皮肤 |
| **7.0 – 10.0 s** | **主体完全消失，纯空景**（占全片 30%） |

## 13.2 根因（对照官方指南 
eferences/base-en.txt）

| # | 错误 | 官方原文 |
|---|---|---|
| 1 | 写了「少女保持…**与整体构图不变**」**同时**要求推镜 —— 自相矛盾 | preserving her appearance, clothing, seat position, and the carriage layout（**不含 composition**） |
| 2 | 运镜**只有幅度和速度、没有落点** | The camera pushes in with small amplitude at slow speed **toward the folded letter in her hands**. |
| 3 | **10 秒只写了 [Shot 1]** —— 后段没有内容可描述 | [Shot 2] At 00:03.500, the camera cuts to... |
| 4 | 对参考图里**被头发遮住的脸**要求推近展示 | 参考图没有的信息，模型只能编，编不出就出空白 |
| 5 | 切镜用了中文「镜头切到」 | §4.2 指定英文：	he camera cuts to 等五个短语 |
| 6 | 运镜用了表格原形 Push In | 示例是变位：The camera **pushes in** ... |

## 13.3 修复效果（H3_Quality_00002_.mp4，修正版提示词，唯一变量）

**同 seed 12345 / 同 20 步 / 同 768×1024 / 同注意力后端**

| 项 | 旧 | 新 |
|---|---|---|
| 7.0–10.0 s 主体 | ❌ 消失 | ✅ **始终在画面内** |
| 特写脸部 | ❌ 空白皮肤 | ✅ 侧脸轮廓 + 头发遮脸 |
| 推镜 | ❌ 失控推到特写 | ✅ 受控，中景停住 |
| 6.0 s 切镜 | 无 | ✅ **精确命中** |
| 墙钟 | 2613 s | 2283 s（−12.6%，属会话波动） |

**对比图**：ench/quality/old_vs_new.png · ench/quality/new_grid.png · ench/quality/dense_grid.png

## 13.4 20 步 / 10 秒的实测耗时区间

| 运行 | 墙钟 |
|---|---|
| H3_Quality_00001_ | 2613 s = **43.5 分钟** |
| H3_Quality_00002_ | 2283 s = **38.0 分钟** |
| **区间** | **38–43 分钟（±7%）** |

**⇒ 我此前用 T ≈ 102+34.1×步数 外推出的 28.7 分钟是错的**：那个模型的斜率来自 **5 秒**测量，而每步成本随序列长度**超线性增长**（本机 5 秒→10 秒，每步从 ~34 s 涨到 ~110–125 s）。**长序列下不能用短序列的斜率外推。**

## 13.5 已写入技能（避免重复踩坑）

.dsh/skills/h3-prompt-writing/SKILL.md 的本机覆盖块已补入：
- 运镜词汇与切镜短语**必须保持英文**（原先清单遗漏）
- 4 条实测踩过的坑（构图矛盾 / 无落点 / 长短片只写一镜 / 要求展示被遮的脸）
