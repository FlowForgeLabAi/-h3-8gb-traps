# 发布文案（复制粘贴用）

---

## 1. 仓库名建议

```
h3-8gb-traps
```
备选：`minimax-h3-8gb-notes` · `comfyui-h3-lowvram-findings`

---

## 2. GitHub「About」一句话描述（≤350 字符）

**英文：**
```
Six measured traps running MiniMax H3 on an 8GB laptop GPU: a prompt-timeline bug that
silently drops the subject, SageAttention being a pessimisation on sm_120, SaveVideo's
dead H.264 path, an 80% LoRA, and why 16GB of RAM (not 8GB of VRAM) is the wall.
```

**中文：**
```
8GB 笔记本跑 MiniMax H3 的六个实测陷阱：丢主体的提示词时间线 bug、sm_120 上 SageAttention
是负优化、SaveVideo 的 H.264 通道是死的、LoRA 只有 80% 生效，以及为什么真正的墙是
16GB 内存而不是 8GB 显存。全部附复现方式。
```

---

## 3. Topics（标签，可直接粘）

```
comfyui  minimax  minimax-h3  video-generation  low-vram  sm120  blackwell
sageattention  attention  lora  quantization  nvidia  rtx5060  rtx4060
benchmark  windows  pytorch  generative-ai  diffusion  text-to-video
```

---

## 4. Release 说明（v0.1.0）

```markdown
## v0.1.0 — 六个实测陷阱

全部在 RTX 5060 Laptop 8GB (sm_120) + 15.26 GiB RAM / Windows 11 / ComfyUI 0.35.0 /
PyTorch 2.14.0+cu130 上实测。

### 发现

1. **提示词时间线不覆盖会导致 10 秒片子 7 秒丢主体**
   只写 `[Shot 1]` + 运镜无落点 → 模型自己填空景。20 步官方基线可完整复现。
   修复后主体全程在画面内（同 seed 单变量对照）。

2. **sm_120 上 SageAttention 是负优化，且会架空后端选择**
   H3 真实几何实测：kitchen INT8 22.8ms/err 0.0164 vs sage 29.0ms/0.0387 vs SDPA 143.1ms。
   sage patch 直接调 `_sageattn_int8_fp8_nhd`，绕过 `optimized_attention`。

3. **`SaveVideo` 的 H.264 重编码在本环境不可用**
   `avcodec_open2("libx264", {})` 连空参数都失败。唯一可用模式 `format=auto` 不重编码。
   改用 `VHS_VideoCombine`（走外部 ffmpeg），crf 实测可用：码率 2067 → 10393 kb/s。

4. **turbo LoRA 在 pruned 基座上只有 80.3% 生效**
   259 个模块中 51 个 `adaln_proj.linear` 因形状（2688 vs 8）被跳过 —— 而 AdaLN 正是
   步数蒸馏的作用通路。

5. **真正的瓶颈是 16GB 内存，不是 8GB 显存**
   首次 0.9–6 秒/block，连续 1.7 小时后 212 秒/block。判据：`nvidia-smi` 功耗
   正常 64–98W，换页时平线 34–36W 而利用率仍显示 99%。

6. **8 步 vs 20 步：构图几乎一致，时间 14.3 vs 33.8–43.5 分钟**

### 交付物

- `scripts/` 9 个可复现脚本（含秒级输出节点探针，不加载模型）
- `workflows/` 三档工作流（Quality 20 步 / Balanced 8 步 / Fast 4 步）
- `evidence/` 9 份证据（对照图 + 14 行 × 33 字段实测 CSV）
- `videos/` 三段成片（修前 / 修后 20 步 / 修后 8 步高码率）

### 已知局限（重要）

- **零 4060 实测** —— 文中 4060 数据引自社区，且 sm_89/sm_120 后端可用性不同
- 6 的对比被编码器差异污染，单 seed
- `ref_image_size=max` 未测
- 9:16 是因内存不足放弃，非因其错误
- 本会话中有 5 条结论被作者自己推翻，README 已逐条标注

### 许可

脚本 MIT。不含任何模型权重、LoRA 或自定义节点代码。
```

---

## 5. 发帖文案（中文社区 / B站动态 / 小红书）

```
【8GB 笔记本跑 MiniMax H3，我踩了六个坑，每个都有实测数据】

硬件：RTX 5060 Laptop 8GB + 16GB 内存

1️⃣ 10 秒的片子到 7 秒角色突然消失 —— 折腾半天发现是我的提示词问题
   只写了 [Shot 1]，后 4 秒没内容可描述，模型就开始编空景
   而且运镜「小幅度向前推进」没有落点 → 模型一路推过头把主体推出画面
   ★ 20 步官方基线照样复现，所以跟步数、加速节点完全无关

2️⃣ sm_120 上装 SageAttention 是负优化
   实测 kitchen INT8 22.8ms/误差 0.0164  vs  sage 29.0ms/误差 0.0387
   又快又准。而且 sage patch 会让「注意力后端选择」彻底失效

3️⃣ SaveVideo 的 H.264 重编码在这套环境是坏的
   avcodec_open2("libx264", {}) 连空参数都失败
   换成 VHS_VideoCombine（走外部 ffmpeg）→ 码率 2067 → 10393 kb/s（5倍）

4️⃣ turbo LoRA 只有 80.3% 生效
   259 个模块里 51 个 adaln_proj 因形状不匹配被跳过
   而 AdaLN 正是步数蒸馏要改的通路

5️⃣ 卡住你的不是 8GB 显存，是 16GB 内存
   刚重启 0.9–6 秒/层，连续跑 1.7 小时后 212 秒/层
   判断方法：看 nvidia-smi 功耗。正常 64–98W，换页时平线 34W 但利用率还是 99%

6️⃣ 8 步 vs 20 步：构图几乎一样，时间 14.3 分钟 vs 33.8–43.5 分钟

全部附复现脚本和原始数据，GitHub 链接在评论区
⚠️ 文中明确标注了未验证项（4060 零实测等），请勿外推
```

---

## 6. 发帖文案（英文 / X / Reddit）

```
Six things I measured running MiniMax H3 on an 8GB laptop GPU (RTX 5060, sm_120),
each with a reproduction — and each one I got wrong first:

1. A 10s clip silently loses its subject at ~7s. It's the PROMPT: only [Shot 1] written,
   so the last 4 seconds have nothing to describe, and the camera move had no target so
   it pushes until the subject leaves frame. A 20-step official baseline reproduces it.

2. SageAttention is a pessimisation on sm_120. Measured on H3's real geometry:
   kitchen INT8 22.8ms/err 0.0164 vs sage 29.0ms/0.0387 vs SDPA 143.1ms.
   Also: the sage patch bypasses optimized_attention, so ModelAttentionBackend does nothing.

3. SaveVideo's H.264 re-encode is DEAD here — avcodec_open2("libx264", {}) fails with no
   options at all. Only format=auto works, and that doesn't re-encode.
   VHS_VideoCombine (external ffmpeg) works: bitrate 2067 -> 10393 kb/s.

4. The turbo LoRA applies only 80.3% to the pruned base. 51 of 259 modules are
   adaln_proj.linear, failing on 2688-vs-8 — and AdaLN is exactly the timestep path a
   step-distilled LoRA needs to modify.

5. The wall is 16GB of RAM, not 8GB of VRAM: 0.9-6 s/block fresh vs 212 s/block after
   ~1.7h. The tell is nvidia-smi power — 64-98W normally, a flat 34-36W while utilization
   still reads 99%.

6. 8 steps vs 20 steps: near-identical composition, 14.3 min vs 33.8-43.5 min.

Scripts, workflows, raw CSVs and clips in the repo. Unverified items are labelled.
```

---

## 7. 上传前的自查清单

- [ ] 看过 `videos/03-...8step-vhs-crf12-with-audio.mp4`，确认画质满意
- [ ] 决定用 `README.md`（英文）还是 `README.zh-CN.md`（中文）作为主文档
- [ ] 解压 zip → `git init` → `git add .` → 首次提交
- [ ] 提交信息可用：`feat: six measured MiniMax H3 traps on an 8GB laptop GPU`
- [ ] 18 MB 的视频若嫌大，可只保留 `01-` 与 `03-`（对照 + 最佳）
- [ ] 复制第 2 节的 About 描述、第 3 节的 Topics
- [ ] 打 tag `v0.1.0` 并把第 4 节作为 Release 说明
- [ ] **不要**声称附带的三个工作流是最优解 —— README 里已经写明
