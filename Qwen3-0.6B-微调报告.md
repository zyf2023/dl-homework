# Qwen3-0.6B 监督微调（SFT / LoRA）技术报告

> **文档说明**
> 本文是一份可直接执行、可填写结果的完整技术报告框架。
> - 配置、代码、超参数均为可直接运行的真实值。
> - 标注 `【待填】` 的是必须由实际训练产生的结果，**不做任何虚构**。
> - 标注 `【参考】` 的来自公开实践或官方文档，仅供参考，不代表本机实测。
>
> 版本：v1.0　｜　适用硬件：RTX 5090 32GB（主）/ RTX 4070 12GB（辅）

---

## 0. 摘要

本报告记录在 Qwen3-0.6B 上执行监督微调（Supervised Fine-Tuning, SFT）的完整流程，包括模型选型依据、数据构建规范、LoRA 参数高效微调方法、超参数设计逻辑、训练监控、评测协议与消融实验设计。

**核心结论（跑完实验后填写）**

| 指标 | 数值 |
|---|---|
| 基座模型 | Qwen/Qwen3-0.6B |
| 微调方式 | LoRA（r=16, α=32, target=all） |
| 训练数据量 | 【待填】条 |
| 可训练参数 | 【待填】/ 约 752M（占比【待填】%） |
| 训练耗时 | 【待填】 |
| 峰值显存 | 【待填】GB |
| 测试集指标 | 【待填】 |
| 相对 zero-shot 提升 | 【待填】 |

---

## 1. 模型选型

### 1.1 型号澄清

社区常说的 "Qwen 0.7B" 实际指向 **Qwen3-0.6B**。官方并未发布 0.7B 规格。

| 项目 | Qwen3-0.6B-Base | Qwen3-0.6B（Instruct） |
|---|---|---|
| 总参数量 | 596M | **752M** |
| 非嵌入参数 | 0.44B | 0.44B |
| 层数 | 28 | 28 |
| 注意力头（Q/KV） | 16 / 8（GQA） | 16 / 8（GQA） |
| 词表绑定（Tie Embedding） | Yes | Yes |
| 上下文长度 | 32K | 32K（config 中 max_position_embeddings=40K） |
| 训练阶段 | 仅预训练 | 预训练 + 后训练 |
| 许可证 | Apache-2.0 | Apache-2.0 |

> **为什么 Instruct 版参数更多（752M vs 596M）**：Instruct 版额外包含后训练阶段的嵌入/输出层调整，且部分实现保留独立 lm_head，导致按 safetensors 统计为 752M。

### 1.2 Base 还是 Instruct？

| 选择 | 适用场景 | 建议 |
|---|---|---|
| **Qwen3-0.6B（Instruct）** | 已有对话能力，想让模型适配特定任务/风格/领域 | ✅ **默认选这个** |
| Qwen3-0.6B-Base | 想从零构建对话能力、或做继续预训练（CPT） | 需要大量数据，不适合小样本微调 |

**本项目选择 Qwen3-0.6B（Instruct）**，因为它已具备 instruction following 能力，微调只需"对齐到目标任务"，数据量需求低一个数量级。

### 1.3 为什么选 0.6B 而不是更大的模型

| 考量 | 说明 |
|---|---|
| **迭代速度** | 单卡几十分钟到几小时即可完成一轮，可以快速做数据/超参消融 |
| **显存友好** | 32GB 显存可开大 batch 甚至全参微调；12GB 显存用 QLoRA 也能跑 |
| **成本可控** | 数据配比、prompt 模板、评测口径这些"脏活"在 0.6B 上试错最便宜 |
| **能力上限** | 0.6B 天然有上限，复杂推理任务会吃力——这是它作为"方法论验证平台"的定位，不是最终模型 |

**推荐做法**：先在 0.6B 上把**数据格式、训练超参、评测脚本**三件事全部调通并形成 SOP，再把同一套流程迁移到 4B/8B。

### 1.4 与同期同尺寸模型对比（官方基准）

| 模型 | 参数量 | 定位 |
|---|---|---|
| Qwen2.5-0.5B | 0.5B | 上一代 |
| **Qwen3-0.6B** | 0.6B | ✅ 本项目基座 |
| Gemma-3-1B | 1B | 参数更大 |

> Qwen 官方称 Qwen3-4B 已能匹敌 Qwen2.5-72B-Instruct；0.6B 在同尺寸中属第一梯队（详见 Qwen3 Technical Report, arXiv:2505.09388）。

---

## 2. 任务定义与数据集

### 2.1 任务定义模板

```
任务名称：【待填】
输入格式：【待填，例如：一段用户问题 / 一段文本 / 一张图的描述】
输出格式：【待填，例如：JSON / 短答案 / 一段解释】
成功标准：【待填，例如：F1 ≥ X / 人工打分 ≥ 4/5】
```

**推荐先做"判别式任务"再上"生成式任务"**：

| 任务类型 | 评价指标 | 难度 | 建议顺序 |
|---|---|---|---|
| 文本分类（如意图识别、情感分类） | Accuracy / F1 | ⭐ | 先做这个验证管线 |
| 信息抽取（NER、结构化输出） | F1 / 精确匹配 | ⭐⭐ | 第二步 |
| 开放问答 / 摘要 | ROUGE / BLEU / 人工 | ⭐⭐⭐ | 第三步 |

### 2.2 数据格式（LLaMA-Factory Alpaca 格式）

`data/train.json`：

```json
[
  {
    "instruction": "判断下面这句话的情感倾向，只输出 positive / negative / neutral 三个词之一。",
    "input": "这家店的服务态度真的很差，等了半小时没人理。",
    "output": "negative"
  },
  {
    "instruction": "判断下面这句话的情感倾向，只输出 positive / negative / neutral 三个词之一。",
    "input": "包装精美，物流很快，非常满意！",
    "output": "positive"
  }
]
```

**字段说明**

| 字段 | 必填 | 说明 |
|---|---|---|
| `instruction` | ✅ | 任务指令，**同一个任务内应保持措辞一致** |
| `input` | ⭕ | 待处理内容，无输入时留空字符串 |
| `output` | ✅ | 期望输出，**格式必须完全统一**（大小写、标点、换行） |

**如果需要多轮对话**，改用 ShareGPT 格式（`conversations` 字段），但**小样本 SFT 首选单轮 Alpaca 格式**，因为噪声更小、更容易诊断。

### 2.3 数据规模建议

| 数据量 | 效果 | 说明 |
|---|---|---|
| < 200 条 | ❌ 不可靠 | 模型会直接"背题" |
| **500–2000 条** | ✅ **小任务足够** | 本项目推荐区间 |
| 2000–10000 条 | ✅✅ 更稳 | 覆盖更多边界情况 |
| > 10000 条 | ⚠️ 需检查重复 | 增量收益递减，先做去重 |

> **经验法则**：数据量不足时，模型多跑几遍 epoch 就开始记忆而非泛化。**质量 > 数量**，500 条精标数据远胜 5000 条脏数据。

### 2.4 数据划分

```
train : val : test = 8 : 1 : 1
```

| 集合 | 用途 | 纪律 |
|---|---|---|
| train | 训练 | 可重复使用 |
| val | 训练中选 checkpoint、调超参 | 可多次查看 |
| **test** | **最终报告数字** | 🔒 **实验做完之前绝对不许看** |

**必须保证三个集合无重复样本、无同源样本（例如同一篇文章切出来的多个片段不能跨集合）。** 否则指标虚高。

### 2.5 数据质检清单

跑训练前逐项检查：

- [ ] `output` 字段格式统一（大小写、标点、JSON 结构）
- [ ] 无空 `output`、无重复样本（`instruction+input` 去重）
- [ ] 每个类别/标签样本量均衡（偏差不超过 3:1）
- [ ] 训练集与验证集无泄漏（可用 MinHash / 精确匹配查）
- [ ] 长度分布合理，超过 `cutoff_len` 的样本要统计占比
- [ ] `instruction` 措辞统一（不要一半"请判断"一半"帮我看看"）

---

## 3. 实验环境

### 3.1 硬件

| 项目 | 主训练机 | 辅助机 |
|---|---|---|
| GPU | **RTX 5090 32GB** | RTX 4070 12GB |
| CPU | Intel Core Ultra 7 270K Plus（24 核） | i7-13700KF |
| 内存 | 64GB DDR5-6400 | 32GB |
| 存储 | 1TB SSD + 4TB PCIe4.0 | — |
| 电源 | 振华 1200W 钛金 ATX3.1 | — |

> ⚠️ 5090 满载 600W，供电话必须用**电源自带的原生 12V-2x6 线**，插到底听"咔"声，接口附近 3–4cm 保持直线，不用显卡附带的 4×8pin 转接头。

### 3.2 软件版本（推荐组合）

```bash
# 基础环境
Python            3.11
CUDA              12.4+（需匹配 5090 / Blackwell 架构）
PyTorch           2.5+（必须支持 sm_120）
Transformers      4.51+
Accelerate        1.0+
PEFT              0.13+
bitsandbytes      0.44+（QLoRA 才需要）
flash-attn        2.7+（可选，加速用）

# 训练框架
LLaMA-Factory     0.9.3+
```

**验证安装是否可用（第一步必做）**

```bash
# 第1步：确认 PyTorch 能识别 GPU
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

# 第2步：确认 compute capability（5090 应为 sm_120）
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# 第3步：确认能跑一次矩阵乘（最重要，很多"装了但用不了"就是这步挂）
python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print((a@a).sum().item())"
```

> ⚠️ **如果你的 PyTorch 版本早于 2.7，可能不支持 sm_120（Blackwell）**，会报 `no kernel image is available`。届时升级 PyTorch 到支持 5090 的版本，或用 nightly 构建。

### 3.3 显存估算（Qwen3-0.6B）

| 训练方式 | 权重 | 优化器 | 激活值 | 总计（粗估） | 12GB 卡能跑吗 |
|---|---|---|---|---|---|
| **LoRA**（bf16） | 1.5 GB | 极小（只存 adapter） | 1–4 GB | **4–8 GB** | ✅ 能 |
| **QLoRA**（4bit） | 0.4 GB | 极小 | 1–3 GB | **2–5 GB** | ✅ 轻松 |
| **全参微调**（bf16） | 1.5 GB | 12 GB（AdamW） | 2–6 GB | **15–24 GB** | ⚠️ 勉强/不行 |

> 实际占用随 `cutoff_len`、`batch_size`、是否开启 activation checkpointing 变化很大。**`cutoff_len` 是显存第一影响因子。**

---

## 4. 微调方法

### 4.1 三种方法对比

| 方法 | 可训练参数 | 显存 | 效果上限 | 适用 |
|---|---|---|---|---|
| **Full FT** | 100% | 高 | 最高 | 数据多、算力足 |
| **LoRA** | 0.5%–2% | 低 | 接近全参 | ✅ **本项目选择** |
| **QLoRA** | 0.5%–2% | 最低 | 略低于 LoRA | 显存受限时 |

### 4.2 LoRA 原理

对原始线性层权重 `W₀ ∈ ℝ^{d×k}`，LoRA 冻结 `W₀`，只训练一个低秩增量：

```
W = W₀ + ΔW = W₀ + (α / r) · B · A
```

其中：

| 符号 | 维度 | 说明 |
|---|---|---|
| `A` | r × k | 用高斯随机初始化 |
| `B` | d × r | **初始化为 0**（保证训练起点等价于原模型） |
| `r` | — | rank，控制容量 |
| `α` | — | 缩放系数，实际缩放为 `α / r` |

**可训练参数量**：`r × (d + k)`，远小于 `d × k`。

**为什么 `B` 初始化为 0**：保证 `ΔW = 0`，微调开始时模型输出与基座完全一致，不会因为我们插入了 adapter 就先把模型搞坏。

### 4.3 为什么选 LoRA

| 理由 | 说明 |
|---|---|
| **快速迭代** | 一轮训练几十分钟，可以跑 10+ 组消融 |
| **显存友好** | 4070 12GB 也能跑，两台机器可以并行做不同实验 |
| **可插拔** | adapter 只有几十 MB，可以给多个任务各存一个，共享同一份基座 |
| **风险低** | 基座权重不动，不容易灾难性遗忘（相对全参） |
| **便于对比** | 合并/不合并两种形态都能评测，不影响部署 |

### 4.4 target_modules 怎么选

| 配置 | 说明 | 参数量 | 建议 |
|---|---|---|---|
| `q_proj,v_proj` | 只改注意力最核心的两层 | 最少 | 数据少时用 |
| `q_proj,k_proj,v_proj,o_proj` | 全部注意力层 | 中 | 常用 |
| **`all`** | 注意力 + MLP（gate/up/down） | 最多 | ✅ **推荐起点** |

> 0.6B 模型参数量小，`all` 也不会占用太多显存，能力上限更高。**先用 `all`，如果过拟合再收缩。**

---

## 5. 超参数配置与理由

### 5.1 主配置表

| 参数 | 取值 | 理由 | 可调范围 |
|---|---|---|---|
| `lora_rank` (r) | **16** | 0.6B 模型上容量与效率的平衡点 | 8 / 16 / 32 |
| `lora_alpha` | **32** | 惯例取 `2 × rank`，缩放系数 = 2.0 | 16 / 32 / 64 |
| `lora_dropout` | **0.05** | 轻微正则，防止小数据集过拟合 | 0 / 0.05 / 0.1 |
| `lora_target` | **all** | 覆盖注意力 + MLP | — |
| `learning_rate` | **1.0e-4** | LoRA 常用起点；过高震荡，过低收敛慢 | 5e-5 / 1e-4 / 2e-4 |
| `lr_scheduler_type` | **cosine** | 后期平滑衰减，收敛更稳 | linear / cosine |
| `warmup_ratio` | **0.03** | 短预热，避免初期梯度爆炸 | 0.01 / 0.03 / 0.1 |
| `num_train_epochs` | **3** | 500–2000 条数据的常用值 | 2 / 3 / 5 |
| `cutoff_len` | **1024** | 覆盖大部分样本，控制显存 | 512 / 1024 / 2048 |
| `per_device_train_batch_size` | **8**（5090）/ **2**（4070） | 按显存调整 | — |
| `gradient_accumulation_steps` | **4**（5090）/ **16**（4070） | 补足有效 batch | — |
| **effective batch size** | **32** | 三者乘积固定为 32 | 16 / 32 / 64 |
| `bf16` | **true** | Blackwell 原生支持，显存省一半，数值比 fp16 稳 | — |
| `optim` | **adamw_torch** | 稳定、通用 | adamw_8bit（省显存） |
| `max_grad_norm` | **1.0** | 梯度裁剪，防跳变 | 0.5 / 1.0 |
| `seed` | **42** | 保证可复现 | — |
| `enable_thinking` | **false** | Qwen3 的 thinking 模式会输出 `...`，SFT 阶段关闭可避免学到冗长推理 | — |
| `template` | **qwen3_nothink** | 必须与模型匹配，否则格式错乱 | — |

### 5.2 关键参数的作用机理

**effective batch size 是真正的"批量"**

```
effective_batch = per_device_batch × grad_accum × gpu_count
```

梯度累积的实现：先累积 N 次前向+反向的梯度，再统一次 `optimizer.step()`。
它**等价于**大 batch（除了 BatchNorm 类的统计量，Transformer 里没有这个问题）。

**为什么固定为 32**：
- 学习率与 batch size 存在耦合，换机器时保持 effective batch 不变，学习率才可以沿用。
- 5090 用 `8×4`，4070 用 `2×16`，**训练动力学一致**，两台机器的结果可直接对比——这一点对"5090 跑主实验、4070 跑消融"非常重要。

**cutoff_len 是第一显存杀手**

激活值显存近似随序列长度**线性**增长（开启 flash-attn 后），随 batch 线性增长。从 1024 提到 2048，显存可能翻倍。

**bf16 而不是 fp16**

Blackwell（5090）原生支持 bf16，与 fp16 同速度但数值范围大得多，不需要 loss scaling，**小模型微调几乎不会出现 inf/nan**。

### 5.3 不需要调的参数（新手常踩的坑）

| 参数 | 建议 | 原因 |
|---|---|---|
| 学习率调度器的 `min_lr` | 保持默认（≈ lr 的 10%） | 调到 0 会让最后阶段完全学不动 |
| `weight_decay` | 保持默认 | LoRA 参数少，正则收益不明显 |
| `neftune_noise_alpha` | 先不开 | 有增益但会给结果加不确定因素，先建立基线 |
| `packing` | 先关闭 | 打包会改变有效样本边界，调试期先关 |

---

## 6. 训练过程与监控

### 6.1 启动后第一眼要看的三行

```
Num examples = 【待填】
Total optimization steps = 【待填】
Number of trainable parameters = 【待填】
```

**自查公式**

```
总步数 = ceil(样本数 / effective_batch) × epochs

例：1600 条 / 32 × 3 轮 = 50 × 3 = 150 步
```

如果**实际步数与手算对不上**，说明数据没有被完全读入（格式错误、被过滤、`max_samples` 截断），要立刻停下来查。

**可训练参数占比自查**：LoRA `all` 在 0.6B 上通常落在 **1%–3%** 区间。
- 如果 < 0.1% → `lora_target` 没生效
- 如果 > 10% → 可能误用了 full 模式

> 【参考】某公开实践在 Qwen3-0.6B + LoRA(r=8, all) + 1600 条数据下：trainable params = 5,046,272，占比 0.8395%，150 步，V100-32GB 训练耗时 519.8 秒（约 8 分 40 秒）。

### 6.2 Loss 曲线形态判读

| 现象 | 可能原因 | 处理 |
|---|---|---|
| **平稳下降，后期趋缓** | ✅ 正常 | 继续 |
| 前 10 步还在 2.0 以上，之后骤降 | ✅ 正常（warmup + B 初始为 0） | 继续 |
| **Loss 长时间不动（> 50 步）** | 学习率过低 / 数据模板错误 / 全部样本被截断 | 检查 `template`、`cutoff_len`、样本长度 |
| **Loss 抖动剧烈（±0.5 以上）** | 学习率过高 / batch 太小 | lr ÷ 2；增大 grad_accum |
| **Loss 降到很低（< 0.3）但验证集变差** | ❌ 过拟合 | 减 epoch、加 dropout、加数据 |
| **Loss 变成 NaN** | 学习率过高 / fp16 溢出 | 换 bf16；lr ÷ 10 |

### 6.3 训练速度与耗时

| 硬件 | 配置 | 相对速度 |
|---|---|---|
| RTX 5090 32GB | LoRA, bs=8×4 | 基准 1.0×（【待填】steps/s） |
| RTX 4070 12GB | LoRA, bs=2×16 | 约 0.3–0.4× |
| RTX 3090 24GB | LoRA, bs=4×8 | 【参考】0.6B 上通常 < 30 分钟/轮 |
| Mac M4（无 CUDA） | fp32 CPU/MPS | 极慢，仅调试用 |

> 【参考】公开实践：R9 3090 上 Qwen3-0.6B 全参微调约 2 小时，LoRA 可缩短到 30 分钟以内。

### 6.4 监控命令

```bash
# 每秒刷新，观察显存与功耗
nvidia-smi -l 1

# 只看关键指标
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2

# 训练全程后台运行并记录日志
nohup llamafactory-cli train config.yaml > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**健康状态参考**

| 指标 | 正常范围 | 异常信号 |
|---|---|---|
| GPU 利用率 | 85%–100% | 长期 < 50% → 数据加载是瓶颈（调 `dataloader_num_workers`） |
| 显存占用 | 稳定不增长 | 持续增长 → 有张量没释放 |
| 温度 | < 80°C | > 85°C → 降功耗墙或优化风道 |
| 功耗 | 稳定 | 剧烈波动 → 可能是 batch 大小不一导致 |

---

## 7. 评测协议

### 7.1 评测原则（写论文/报告必须遵守）

```
1. 固定 test set，所有模型用同一个
2. 固定解码参数：greedy（do_sample=False），max_new_tokens 固定
3. 关闭 thinking 模式，所有模型一致
4. 同一个 prompt 模板，与训练时一致
5. 报告均值 ± 标准差（≥3 个随机种子）
6. 不根据 test 结果调整任何超参（那是 val 的活）
```

### 7.2 指标选择

| 任务类型 | 主指标 | 辅助指标 |
|---|---|---|
| 分类 | Accuracy、Macro-F1 | 各类别 precision/recall |
| 抽取 | Exact Match、Entity F1 | — |
| 生成 | ROUGE-L / BLEU | 人工 1–5 分、长度分布 |
| 通用 | — | 困惑度（perplexity）仅作参考 |

**必须报告类别分布**：Macro-F1 与 Accuracy 差距大时，说明类别不均衡，此时 Macro-F1 才是真实水平。

### 7.3 必做的三组对照

| 组别 | 模型 | 目的 |
|---|---|---|
| A | Qwen3-0.6B zero-shot（不微调） | 基线，证明任务本身不难/难 |
| B | Qwen3-0.6B + 本文 SFT | 主结果 |
| C | 更大模型（如 Qwen3-4B）zero-shot | 上限参考，说明 0.6B 差距 |

**没有 A 组，你的"SFT 有效"就没有说服力。** 报告里 A→B 的差值才是你的贡献。

---

## 8. 结果

### 8.1 主结果表（跑完填写）

| 配置 | Accuracy | Macro-F1 | 训练耗时 | 峰值显存 |
|---|---|---|---|---|
| A. zero-shot 基座 | 【待填】 | 【待填】 | — | — |
| **B. LoRA SFT（本方案）** | **【待填】** | **【待填】** | 【待填】 | 【待填】GB |
| C. 更大模型 zero-shot（可选） | 【待填】 | 【待填】 | — | — |

### 8.2 消融实验设计

建议在 4070 上并行跑（不影响 5090 主实验）：

| 编号 | 变量 | 设置 | 目标 |
|---|---|---|---|
| E1 | `lora_rank` | 8 / 16 / 32 | rank 的边际收益 |
| E2 | `learning_rate` | 5e-5 / 1e-4 / 2e-4 | 学习率敏感性 |
| E3 | `epochs` | 2 / 3 / 5 | 过拟合拐点 |
| E4 | `lora_target` | `q,v` / 全部注意力 / `all` | MLP 是否必要 |
| E5 | 数据量 | 25% / 50% / 100% | 数据规模曲线 |
| E6 | 数据格式 | 统一 instruction vs 多样化 | 指令多样性收益 |

**消融表模板**

| 实验 | 变化 | Accuracy | Macro-F1 | 结论 |
|---|---|---|---|---|
| 基线 | r=16, lr=1e-4, ep=3 | 【待填】 | 【待填】 | — |
| E1-a | r=8 | 【待填】 | 【待填】 | 【待填】 |
| E1-b | r=32 | 【待填】 | 【待填】 | 【待填】 |
| … | | | | |

### 8.3 错误分析（这部分最体现功力）

把测试集的错误样本全部人工过一遍，按下表归类统计：

| 错误类型 | 占比 | 典型例子 | 改进方向 |
|---|---|---|---|
| 格式错误（没按要求输出） | 【待填】% | 【待填】 | 增加格式约束样本 |
| 标签混淆（相近类别分不清） | 【待填】% | 【待填】 | 补充边界样本 |
| 完全无关回答 | 【待填】% | 【待填】 | 检查是否模板不匹配 |
| 输入过长被截断 | 【待填】% | 【待填】 | 提高 `cutoff_len` |
| 过度推理（输出冗长） | 【待填】% | 【待填】 | 关闭 thinking，加"只输出X"约束 |

> **报告里一定要有错误分析。** 只报一个准确率数字的报告，价值有限；能说清"错在哪、为什么错、怎么改"的，才是研究。

---

## 9. 复现步骤

### 9.1 完整流程

```bash
# ── 第1步：装环境 ──────────────────────────
conda create -n qwen_sft python=3.11 -y
conda activate qwen_sft
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.51" "peft>=0.13" accelerate datasets
pip install llamafactory

# ── 第2步：下载模型 ────────────────────────
# 国内推荐用 ModelScope 镜像（HuggingFace 可能连不上）
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B

# ── 第3步：验证环境 ────────────────────────
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); a=torch.randn(4096,4096,device='cuda'); print('OK', (a@a).shape)"

# ── 第4步：放入数据 ────────────────────────
cp data/train.json data/val.json data/test.json  LLaMA-Factory/data/
# 然后在 LLaMA-Factory/data/dataset_info.json 中登记（见 9.3）

# ── 第5步：启动训练 ────────────────────────
llamafactory-cli train qwen3_0.6b_lora_sft.yaml

# ── 第6步：合并 adapter ────────────────────
llamafactory-cli export merge_config.yaml

# ── 第7步：评测 ────────────────────────────
python eval.py --model ./output/merged --test data/test.json
```

### 9.2 训练配置文件（`qwen3_0.6b_lora_sft.yaml`）

```yaml
# ═══════════ 模型 ═══════════
model_name_or_path: ./models/Qwen3-0.6B
trust_remote_code: true

# ═══════════ 微调方式 ═══════════
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

# ═══════════ 数据 ═══════════
dataset: my_task_train          # 需在 dataset_info.json 中登记
template: qwen3_nothink         # ⚠️ 必须与模型匹配
cutoff_len: 1024
preprocessing_num_workers: 8
dataloader_num_workers: 4
overwrite_cache: true

# ═══════════ 输出 ═══════════
output_dir: ./output/qwen3_0.6b_lora
logging_steps: 10
save_steps: 100
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: none

# ═══════════ 训练超参 ═══════════
per_device_train_batch_size: 8          # 5090 可到 8~16；4070 改成 2
gradient_accumulation_steps: 4          # 与上者相乘 × 卡数 = 32
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
bf16: true
optim: adamw_torch
seed: 42
ddp_timeout: 180000000

# ═══════════ 验证 ═══════════
val_size: 0.1
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 100
```

### 9.3 数据集登记（`data/dataset_info.json`）

```json
{
  "my_task_train": {
    "file_name": "train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "my_task_val": {
    "file_name": "val.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 9.4 合并导出配置（`merge_config.yaml`）

```yaml
model_name_or_path: ./models/Qwen3-0.6B
adapter_name_or_path: ./output/qwen3_0.6b_lora
template: qwen3_nothink
finetuning_type: lora
export_dir: ./output/merged
export_size: 2
export_device: cpu          # 用 CPU 合并，不抢 GPU
export_legacy_format: false
```

### 9.5 推理验证片段

```python
# 第1步：加载模型与分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./output/merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="bfloat16", device_map="cuda"
)

# 第2步：构造与训练时完全一致的 prompt（用 chat template）
instruction = "判断下面这句话的情感倾向，只输出 positive / negative / neutral 三个词之一。"
text = "这家店的服务态度真的很差，等了半小时没人理。"

messages = [{"role": "user", "content": f"{instruction}\n\n{text}"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)

# 第3步：greedy 解码，保证可复现
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False,          # greedy
    temperature=None,
    top_p=None,
    pad_token_id=tokenizer.eos_token_id,
)

# 第4步：只取新生成的部分
answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer.strip())
```

---

## 10. 常见问题与排查

### 10.1 报错速查

| 报错 / 现象 | 原因 | 解决 |
|---|---|---|
| `no kernel image is available` | PyTorch 不支持 sm_120（5090） | 升级 PyTorch 到支持 Blackwell 的版本 |
| `CUDA out of memory`（加载权重时） | 显存被其他进程占用 / 精度过高 | `nvidia-smi` 查占用；改用 bf16 或 QLoRA |
| `CUDA out of memory`（训练中） | `cutoff_len` 或 batch 太大 | 先降 batch，再降 `cutoff_len` |
| Loss 不降 | 学习率过低 / template 不匹配 / 数据格式错 | 检查日志里前几条样本的 token 化结果 |
| 输出格式混乱、夹杂模板标记 | `template` 与模型不匹配 | 换成 `qwen3_nothink` |
| 输出里有 `思考过程` | 没关闭 thinking | `enable_thinking: false` + `template: qwen3_nothink` |
| 训练完模型变傻（通用能力崩溃） | 过拟合 / 学习率过高 / 数据单一 | 降 epoch、降 lr、混入通用数据 |
| Windows 上 `Can't pickle local object` | 多进程 DataLoader 问题 | `dataloader_num_workers: 0` |

### 10.2 五个必须避免的错误做法

| ❌ 错误 | 后果 | ✅ 正确 |
|---|---|---|
| 用 test set 调超参 | 指标虚高，不可复现 | 只用 val 调参 |
| 训练集里混入验证集样本 | 指标虚高 | 严格去重 + 查泄漏 |
| 不同实验用不同 `effective_batch` | 结果不可比 | 固定为 32 |
| 只报一个随机种子的结果 | 结论不稳健 | ≥3 个种子取均值±标准差 |
| 训练完直接说"效果很好" | 没有说服力 | 必须有 zero-shot 基线和错误分析 |

### 10.3 显存优化优先级

```
第1步：降 per_device_train_batch_size（最有效）
第2步：降 cutoff_len（显存第一杀手）
第3步：开启 gradient_checkpointing（省 30-50%，慢 20%）
第4步：改 QLoRA（4bit 加载，最省）
第5步：用 adamw_8bit 优化器
```

---

## 11. 结论与下一步

### 11.1 本报告达成（跑完后填写）

- 【待填】在 Qwen3-0.6B 上完成 LoRA SFT，测试集 Macro-F1 从 【待填】 提升到 【待填】
- 【待填】验证了 rank / 学习率 / 数据量 三个维度的敏感性
- 【待填】形成了可复用的训练-评测 SOP，可迁移到 1.7B / 4B

### 11.2 后续方向

| 优先级 | 方向 | 说明 |
|---|---|---|
| 🥇 | **扩大数据** | 同一套代码，数据从 1k → 5k 通常是收益最大的动作 |
| 🥈 | **换更大模型** | 0.6B 验证流程 → 4B/8B 复现，看能力上限 |
| 🥉 | **DPO / RLHF** | 有偏好数据后，做偏好对齐 |
| 4 | **量化部署** | 导出 GGUF，看推理速度与实际可用性 |
| 5 | **多任务混合** | 一个 adapter 覆盖多个任务，观察任务间干扰 |

### 11.3 与 CV 主线的结合点（针对本项目）

> 若后续要接图像方向，可考虑：用 0.6B 做**图像描述的后置文本处理**（如把检测结果转成结构化报告），或作为**长文本知识库的轻量问答基线**，与视觉模型串联成 pipeline。

---

## 附录 A：术语表

| 术语 | 全称 | 含义 |
|---|---|---|
| SFT | Supervised Fine-Tuning | 用"输入-期望输出"对训练模型 |
| LoRA | Low-Rank Adaptation | 低秩适配器，只训练少量参数 |
| QLoRA | Quantized LoRA | 基座 4bit 量化后做 LoRA |
| adapter | — | LoRA 训练出的增量权重文件 |
| effective batch | — | per_device × grad_accum × 卡数 |
| cutoff_len | — | 单条样本最大 token 数 |
| GQA | Grouped Query Attention | Q 头多于 KV 头，省显存 |
| warmup | — | 学习率从 0 逐步升到设定值的阶段 |

## 附录 B：一键自查脚本

```bash
# 把这段存成 precheck.sh，训练前跑一遍
echo "=== 1. GPU ===";        nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== 2. PyTorch ===";    python -c "import torch; print(torch.__version__, torch.version.cuda)"
echo "=== 3. 算力可用 ===";   python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print('OK')"
echo "=== 4. 数据条数 ===";   python -c "import json; print(len(json.load(open('data/train.json'))))"
echo "=== 5. 版本 ===";       pip list 2>/dev/null | grep -Ei "transformers|peft|accelerate|llamafactory"
```

---

**报告结束**

> 本文档中标注 `【待填】` 的位置，均需由实际训练日志、评测输出填充。
> 请勿使用估算值或推测值替代真实实验数据。
