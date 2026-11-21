# Qwen3-0.6B 监督微调（LoRA）技术方案

**文档类型**：实验方案设计（Experiment Design）
**目标硬件**：RTX 5090 32GB（主）/ RTX 4070 12GB（辅助消融）
**计划框架**：LLaMA-Factory 0.9.3 + PyTorch 2.7 + Transformers 4.51

> 本文档描述**待执行的实验方案**，不包含实验结果。
> 所有数值均为设计参数或验收标准，实际结果需按第 9、10 节流程产出后另行记录。

---

## 1. 目标与动机

### 1.1 为什么做这件事

在有限算力下，判断"小模型 + 参数高效微调"能在一个具体任务上做到什么程度，并建立一套可迁移的训练—评测 SOP。

这不是追求 SOTA，而是打通三件事：

| 目标 | 验收方式 |
|---|---|
| ① 跑通完整链路 | 从数据到评测可一键复现 |
| ② 摸清超参敏感度 | 消融矩阵覆盖 5 个维度 |
| ③ 形成可迁移流程 | 同一套配置能平移到 1.7B / 4B |

### 1.2 为什么选 0.6B 作为起点

| 考量 | 说明 |
|---|---|
| 迭代速度 | 单轮训练在分钟级，一晚可跑数十组对比 |
| 显存友好 | 消费级显卡即可完成，两台机器可并行做不同实验 |
| 定位 | 0.6B 是"方法论验证平台"，用于把数据/超参/评测流程调通 |
| 上限认知 | 复杂推理会吃力，但它验证的是流程而非能力天花板 |

**策略**：先在 0.6B 上把格式、超参、评测三件事全部固化，再整体平移到更大模型。反过来（先上大模型再调流程）试错成本高一个数量级。

---

## 2. 模型选型

### 2.1 型号澄清

社区常说的 "Qwen 0.7B" 实际指向 **Qwen3-0.6B**，官方未发布 0.7B 规格。

| 项目 | Qwen3-0.6B |
|---|---|
| 总参数量 | 752M |
| 非嵌入参数 | 0.44B |
| 层数 | 28 |
| 注意力头（Q/KV） | 16 / 8（GQA） |
| 隐层维度 | 1024 |
| FFN 中间维度 | 3072 |
| 词表绑定（Tie Embedding） | Yes |
| 上下文长度 | 32K |
| 许可证 | Apache-2.0（可商用） |

### 2.2 Base 还是 Instruct

| 选择 | 适用场景 | 结论 |
|---|---|---|
| **Qwen3-0.6B（Instruct）** | 已具备指令跟随能力，只需对齐到具体任务 | ✅ 采用 |
| Qwen3-0.6B-Base | 从零构建对话能力，或做继续预训练 | 需要海量数据，不适合小样本 SFT |

**理由**：Instruct 版自带 instruction following，千级数据即可对齐到目标任务；Base 版要学会"听指令"本身就需要大得多的数据量。

---

## 3. 任务定义与数据方案

### 3.1 任务定义表

| 项目 | 内容 |
|---|---|
| 任务类型 | 【按实际任务填写，建议优先选择判别式任务】 |
| 输入 | 【待定】 |
| 输出 | 【待定，格式必须严格统一】 |
| 主指标 | Accuracy / Macro-F1（分类）或 Entity F1（抽取） |
| 验收标准 | 见 §11 |

**任务类型选择建议（按实施难度排序）**

| 类型 | 指标 | 难度 | 建议顺序 |
|---|---|---|---|
| 文本分类（意图/情感/主题） | Accuracy、Macro-F1 | ⭐ | 先做，用于验证管线 |
| 信息抽取（实体/结构化输出） | F1、精确匹配 | ⭐⭐ | 第二步 |
| 开放生成（摘要/问答） | ROUGE、人工评分 | ⭐⭐⭐ | 第三步 |

### 3.2 数据格式规范（Alpaca）

```json
[
  {
    "instruction": "判断下面这句话的情感倾向，只输出 positive / negative / neutral 三个词之一。",
    "input": "这家店的服务态度真的很差，等了半小时没人理。",
    "output": "negative"
  }
]
```

| 字段 | 必填 | 规范 |
|---|---|---|
| `instruction` | ✅ | 同一任务内**措辞必须一致**，避免"请判断"与"帮我看看"混用 |
| `input` | ⭕ | 待处理内容，无输入时留空字符串 |
| `output` | ✅ | 格式完全统一：大小写、标点、换行、JSON 结构 |

> 多轮对话改用 ShareGPT 格式（`conversations` 字段）。但小样本 SFT **优先单轮 Alpaca**：噪声更小，出问题更容易定位。

### 3.3 数据规模规划

| 数据量 | 可用性 | 说明 |
|---|---|---|
| < 200 条 | ❌ | 模型直接记忆，无泛化能力 |
| **500–2000 条** | ✅ **本方案目标区间** | 小任务足够 |
| 2000–10000 条 | ✅✅ | 更稳，覆盖更多边界 |
| > 10000 条 | ⚠️ | 需先去重，否则增量收益递减 |

### 3.4 数据划分

```
train : val : test = 8 : 1 : 1
```

| 集合 | 用途 | 纪律 |
|---|---|---|
| train | 训练 | 可重复使用 |
| val | 选 checkpoint、调超参 | 可多次查看 |
| **test** | **最终报告数字** | 🔒 实验定稿前不许查看 |

**硬性要求**：三个集合之间无重复、无同源（同一篇文章切出的多个片段不得跨集合），否则指标虚高。

### 3.5 数据质检清单（训练前逐项勾选）

- [ ] `output` 格式统一（大小写、标点、JSON 结构）
- [ ] 无空 `output`；按 `instruction + input` 去重
- [ ] 类别均衡度：最大类 / 最小类 ≤ 3 : 1
- [ ] 训练/验证/测试无泄漏（精确匹配 + MinHash 双重校验）
- [ ] 超过 `cutoff_len` 的样本占比 < 1%
- [ ] `instruction` 措辞单一化

---

## 4. 实验环境

### 4.1 硬件

| 项目 | 主训练机 | 辅助机 |
|---|---|---|
| GPU | RTX 5090 32GB | RTX 4070 12GB |
| CPU | Intel Core Ultra 7 270K Plus（24 核） | i7-13700KF |
| 内存 | 64GB DDR5-6400 | 32GB |
| 存储 | 1TB SSD + 4TB PCIe 4.0 | — |

> **供电注意**：5090 满载 600W，必须使用电源原生的 12V-2x6 线，**不使用**显卡附带的 4×8pin 转接头；插到底至"咔"声，接口附近 3–4cm 保持直线。

### 4.2 软件版本

```bash
Python        3.11
CUDA          12.4+
PyTorch       2.7.0+      # 必须支持 sm_120（Blackwell），否则会报 no kernel image
Transformers  4.51.3
PEFT          0.13.2
Accelerate    1.0+
LLaMA-Factory 0.9.3
```

### 4.3 环境自检（训练前必跑）

```bash
# 第1步：确认 GPU 可见
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

# 第2步：确认计算能力（5090 应为 sm_120）
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# 第3步：实跑一次矩阵乘（很多"装了但用不了"就挂在这步）
python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print('OK', (a@a).shape)"
```

### 4.4 显存预算（Qwen3-0.6B）

| 训练方式 | 权重 | 优化器状态 | 激活值 | 合计估算 | 12GB 卡 |
|---|---|---|---|---|---|
| LoRA（bf16） | 1.5 GB | 极小（仅 adapter） | 1–4 GB | 4–8 GB | ✅ |
| QLoRA（4bit） | 0.4 GB | 极小 | 1–3 GB | 2–5 GB | ✅ |
| 全参微调（bf16） | 1.5 GB | ~12 GB（AdamW） | 2–6 GB | 15–24 GB | ⚠️ 紧张 |

> `cutoff_len` 是显存的第一影响因子，调整优先级高于 batch size。

---

## 5. 微调方法

### 5.1 方法选型

| 方法 | 可训练参数 | 显存 | 效果上限 | 本项目 |
|---|---|---|---|---|
| Full FT | 100% | 高 | 最高 | ✗（数据量不匹配） |
| **LoRA** | ~1%–2% | 低 | 接近全参 | ✅ 采用 |
| QLoRA | ~1%–2% | 最低 | 略低于 LoRA | ✗（32GB 无需量化） |

### 5.2 LoRA 原理

对线性层权重 `W₀ ∈ ℝ^{d×k}`，冻结 `W₀`，只训练低秩增量：

```
W = W₀ + ΔW = W₀ + (α / r) · B · A
```

| 符号 | 维度 | 初始化 |
|---|---|---|
| `A` | r × k | 高斯随机 |
| `B` | d × r | **全零** |
| `r` | — | rank |
| `α` | — | 缩放系数，实际缩放 `α / r` |

**为什么 B 初始化为 0**：保证训练起点 `ΔW = 0`，模型输出与基座完全一致，插入 adapter 本身不会破坏原模型。

### 5.3 可训练参数量推导（r = 16，target = all）

Qwen3-0.6B：d = 1024，FFN = 3072，28 层，16 Q 头 / 8 KV 头，head_dim = 128。

| 模块 | 形状 | LoRA 参数（r=16） |
|---|---|---|
| q_proj | 1024 × 2048 | 16 × (1024 + 2048) = 49,152 |
| k_proj | 1024 × 1024 | 16 × (1024 + 1024) = 32,768 |
| v_proj | 1024 × 1024 | 16 × (1024 + 1024) = 32,768 |
| o_proj | 2048 × 1024 | 16 × (2048 + 1024) = 49,152 |
| gate_proj | 1024 × 3072 | 16 × (1024 + 3072) = 65,536 |
| up_proj | 1024 × 3072 | 16 × (1024 + 3072) = 65,536 |
| down_proj | 3072 × 1024 | 16 × (3072 + 1024) = 65,536 |
| **单层合计** | | **360,448** |

```
总计 ≈ 360,448 × 28 = 10,092,544
占 752M 总参数 ≈ 1.34%
```

> **用途**：训练日志里的 `Number of trainable parameters` 应与该值一致。
> 偏差过大说明 `lora_target` 未覆盖全部 7 个模块——这是排查配置错误最快的信号。

### 5.4 target_modules 选择依据

| 配置 | 覆盖范围 | 建议 |
|---|---|---|
| `q_proj, v_proj` | 注意力核心两路 | 数据极少时可用 |
| `q,k,v,o` | 全部注意力 | 常规起点 |
| **`all`** | 注意力 + MLP（7 模块） | ✅ **本方案采用** |

**理由**：0.6B 参数量小，全覆盖的显存代价可承受；而 MLP 层承载了大量任务相关变换，排除它会显著限制适配能力。

---

## 6. 超参数配置

### 6.1 主配置

| 参数 | 取值 | 依据 | 备选范围 |
|---|---|---|---|
| `lora_rank` | 16 | 容量与效率平衡点，预设拐点 | 8 / 16 / 32 |
| `lora_alpha` | 32 | 惯例取 2 × rank，缩放系数 2.0 | 16 / 32 / 64 |
| `lora_dropout` | 0.05 | 小数据集轻正则 | 0 / 0.05 / 0.1 |
| `lora_target` | all | 覆盖注意力 + MLP | — |
| `learning_rate` | 1.0e-4 | LoRA 常用起点 | 5e-5 / 1e-4 / 2e-4 |
| `lr_scheduler_type` | cosine | 后期平滑衰减 | linear / cosine |
| `warmup_ratio` | 0.03 | 短预热，避免初期梯度冲击 | 0.01 / 0.03 / 0.1 |
| `num_train_epochs` | 3 | 千级数据的常规区间 | 2 / 3 / 5 |
| `cutoff_len` | 1024 | 覆盖绝大多数样本 | 512 / 1024 / 2048 |
| `per_device_train_batch_size` | 8（5090）/ 2（4070） | 按显存 | — |
| `gradient_accumulation_steps` | 4（5090）/ 16（4070） | 补足有效批次 | — |
| **effective batch** | **32** | 三者的乘积，固定不变 | 16 / 32 / 64 |
| `bf16` | true | Blackwell 原生支持，数值稳定 | — |
| `optim` | adamw_torch | 稳定通用 | adamw_8bit |
| `max_grad_norm` | 1.0 | 梯度裁剪 | 0.5 / 1.0 |
| `seed` | 42 | 可复现 | — |
| `template` | qwen3_nothink | 必须与模型匹配 | — |
| `enable_thinking` | false | 避免学到冗长推理轨迹 | — |

### 6.2 为什么固定 effective batch = 32

```
effective_batch = per_device_batch × grad_accum × gpu_count
```

学习率与批量存在耦合关系。**换机器时保持 effective batch 不变，学习率才可沿用**：

| 机器 | per_device | grad_accum | 结果 |
|---|---|---|---|
| 5090 | 8 | 4 | 32 |
| 4070 | 2 | 16 | 32 |

两台机器训练动力学一致，结果可直接横向对比——这是"5090 跑主实验、4070 跑消融"方案成立的前提。

### 6.3 预期训练步数（供启动后核对）

```
每轮步数 = ceil(样本数 / 32)
总步数   = 每轮步数 × epoch 数
```

启动日志中的 `Total optimization steps` 必须与手算一致。**不一致说明数据未被完整读入**（格式错误 / 被过滤 / `max_samples` 截断），应立即停止排查。

### 6.4 刻意保持默认的参数

| 参数 | 处理 | 原因 |
|---|---|---|
| `min_lr` | 默认（≈ lr 的 10%） | 设为 0 会使末段完全学不动 |
| `weight_decay` | 默认 | LoRA 参数少，正则收益不显著 |
| `neftune_noise_alpha` | 关闭 | 有增益但引入额外变量，先建基线 |
| `packing` | 关闭 | 打包改变样本边界，调试期先禁用 |

---

## 7. 训练监控与判读标准

### 7.1 启动后必看三项

| 日志字段 | 作用 |
|---|---|
| `Num examples` | 与实际数据条数比对 |
| `Total optimization steps` | 与 §6.3 手算值比对 |
| `Number of trainable parameters` | 与 §5.3 推导值比对 |

### 7.2 Loss 曲线判读表

| 现象 | 判断 | 处置 |
|---|---|---|
| 平稳下降、后期趋缓 | ✅ 正常 | 继续 |
| 前 20 步陡降 | ✅ 正常（warmup 结束 + B 从 0 起步） | 继续 |
| 连续 50 步以上不动 | ⚠️ 异常 | 查 `template`、`cutoff_len`、样本长度分布 |
| 抖动幅度 > 0.5 | ⚠️ 学习率偏高 / 批量偏小 | lr ÷ 2；增大 grad_accum |
| 训练 Loss 降但验证 Loss 回升 | ⚠️ 过拟合 | 取更早的 checkpoint；减 epoch；加 dropout |
| 出现 NaN | ⚠️ 数值溢出 | 改用 bf16；lr ÷ 10 |

### 7.3 checkpoint 选择原则

**默认不使用最后一个 checkpoint。**

```
以验证集 Loss 最低点作为交付权重
若验证 Loss 在末段回升，说明已进入过拟合区
```

交付时必须记录：选中的 step、对应的验证指标、以及"为什么不是最后一个"。

### 7.4 资源监控

```bash
# 实时观察
nvidia-smi -l 1

# 关键指标轮询
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2

# 后台训练并留存日志
nohup llamafactory-cli train config.yaml > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

| 指标 | 健康范围 | 异常含义 |
|---|---|---|
| GPU 利用率 | 85%–100% | 长期 < 50% → 数据加载瓶颈 |
| 显存 | 稳定不增长 | 持续增长 → 张量未释放 |
| 温度 | < 80°C | > 85°C → 降功耗墙或优化风道 |

---

## 8. 评测协议

### 8.1 六条纪律

```
① 固定 test set，所有模型共用同一份
② 固定解码参数：greedy（do_sample=False），max_new_tokens 固定
③ 统一关闭 thinking 模式
④ prompt 模板与训练时完全一致
⑤ 报告均值 ± 标准差（≥ 3 个随机种子）
⑥ 绝不依据 test 结果回调超参
```

### 8.2 必做的三组对照

| 组别 | 模型 | 作用 |
|---|---|---|
| **A** | 基座 zero-shot | 基线；没有 A 组，"微调有效"就无说服力 |
| **B** | 基座 + 本方案 SFT | 主结果 |
| **C** | 更大模型 zero-shot（如 4B） | 上限参考 |

**A → B 的差值才是本方案贡献。**

### 8.3 指标选择

| 任务 | 主指标 | 辅助 |
|---|---|---|
| 分类 | Accuracy、Macro-F1 | 各类别 P / R / F1 |
| 抽取 | Entity F1、精确匹配 | — |
| 生成 | ROUGE-L | 人工评分、输出长度分布 |

> 类别不均衡时，Macro-F1 才是真实水平；Acc 与 Macro-F1 差距大，须在报告中说明原因。

---

## 9. 实验计划（消融矩阵）

设计 6 组消融，可在 4070 上并行推进，不占用 5090 主实验。

| 编号 | 变量 | 水平 | 待回答问题 |
|---|---|---|---|
| E1 | `lora_rank` | 8 / 16 / 32 | rank 的边际收益拐点在哪 |
| E2 | `learning_rate` | 5e-5 / 1e-4 / 2e-4 | 本任务的学习率敏感度 |
| E3 | `epochs` | 2 / 3 / 5 | 过拟合起始轮次 |
| E4 | `lora_target` | q,v / 全部注意力 / all | MLP 层是否必要 |
| E5 | 数据量 | 25% / 50% / 100% | 数据是否仍是瓶颈 |
| E6 | 数据格式 | 单一 instruction / 多样化表述 | 指令多样性的影响 |

**记录模板**

| 实验 | 变化 | Acc | Macro-F1 | 峰值显存 | 耗时 | 结论 |
|---|---|---|---|---|---|---|
| 基线 | r=16, lr=1e-4, ep=3 | | | | | — |
| E1-a | r=8 | | | | | |
| E1-b | r=32 | | | | | |
| … | | | | | | |

---

## 10. 风险与排查

### 10.1 报错对照表

| 现象 | 根因 | 处置 |
|---|---|---|
| `no kernel image is available` | PyTorch 不支持 sm_120 | 升级至支持 Blackwell 的版本 |
| 加载权重时 OOM | 显存被占用 / 精度过高 | 查占用；改 bf16 或 QLoRA |
| 训练中 OOM | `cutoff_len` 或 batch 过大 | 先降 batch，再降 `cutoff_len` |
| Loss 不下降 | lr 过低 / 模板不匹配 / 数据格式错 | 打印前几条样本的 tokenize 结果 |
| 输出夹杂模板标记 | `template` 与模型不匹配 | 改用 `qwen3_nothink` |
| 输出含思考过程 | thinking 未关闭 | `enable_thinking: false` |
| 微调后通用能力明显下降 | 过拟合 / lr 过高 / 数据过窄 | 减 epoch、降 lr、混入通用数据 |
| Windows 报 `Can't pickle local object` | 多进程 DataLoader | `dataloader_num_workers: 0` |

### 10.2 显存优化优先级

```
① 降 per_device_train_batch_size   （最直接）
② 降 cutoff_len                    （第一影响因子）
③ 开启 gradient_checkpointing      （省 30–50%，慢约 20%）
④ 转 QLoRA（4bit 加载）            （最省）
⑤ 使用 adamw_8bit 优化器
```

### 10.3 五条红线

| ❌ 禁止 | 后果 |
|---|---|
| 用 test set 调超参 | 指标虚高且不可复现 |
| 训练集混入验证/测试样本 | 指标严重虚高 |
| 不同实验使用不同 effective batch | 结果不可比 |
| 只报单一随机种子 | 结论不稳健 |
| 仅汇报指标、不做错误分析 | 无法定位改进方向 |

---

## 11. 预期产出与验收标准

### 11.1 交付物清单

| 交付物 | 形式 |
|---|---|
| 训练配置 | `yaml` 配置文件（可复现） |
| LoRA adapter | 权重目录（含选中的 step） |
| 评测脚本 | 可一键跑 test set |
| 实验记录表 | 消融矩阵完整填写 |
| 错误分析 | 错误样本分类统计 + 典型案例 |

### 11.2 验收标准（方案目标，非实测结果）

| 项目 | 达标线 |
|---|---|
| 相对 zero-shot 基线 | Macro-F1 提升 ≥ 5 个百分点 |
| 训练可复现性 | 同配置重跑，指标波动 ≤ 2pp |
| 消融完整性 | 6 组实验全部完成并记录 |
| 错误分析 | 覆盖 ≥ 90% 的错误样本并归类 |

### 11.3 后续扩展路径

| 优先级 | 方向 | 说明 |
|---|---|---|
| 🥇 | 数据扩容 | 同一套代码，数据翻倍通常是收益最大的单点动作 |
| 🥈 | 模型升级 | 0.6B 流程平移到 1.7B / 4B，看能力上限 |
| 🥉 | 偏好对齐 | 有偏好数据后接 DPO |
| 4 | 量化部署 | 导出 GGUF，评估推理速度与可用性 |
| 5 | 多任务混合 | 单 adapter 覆盖多任务，观察任务间干扰 |

---

## 12. 复现步骤

```bash
# ── 第1步：环境 ────────────────────────────
conda create -n qwen_sft python=3.11 -y
conda activate qwen_sft
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.51" "peft>=0.13" accelerate datasets
pip install llamafactory

# ── 第2步：模型 ────────────────────────────
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B

# ── 第3步：环境自检 ────────────────────────
python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print('OK', a.shape)"

# ── 第4步：数据 ────────────────────────────
cp data/train.json data/val.json data/test.json  LLaMA-Factory/data/
# 在 LLaMA-Factory/data/dataset_info.json 中登记

# ── 第5步：训练 ────────────────────────────
llamafactory-cli train qwen3_0.6b_lora_sft.yaml

# ── 第6步：合并且导出 ──────────────────────
llamafactory-cli export merge_config.yaml

# ── 第7步：评测 ────────────────────────────
python eval.py --model ./output/merged --test data/test.json
```

### 训练配置

```yaml
model_name_or_path: ./models/Qwen3-0.6B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

dataset: my_task_train
template: qwen3_nothink
cutoff_len: 1024
preprocessing_num_workers: 8
dataloader_num_workers: 4
overwrite_cache: true

output_dir: ./output/qwen3_0.6b_lora
logging_steps: 10
save_steps: 20
save_total_limit: 10
plot_loss: true
overwrite_output_dir: true
report_to: none

per_device_train_batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
bf16: true
optim: adamw_torch
seed: 42
ddp_timeout: 180000000

val_size: 0.1
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 20
```

### 数据集登记（`data/dataset_info.json`）

```json
{
  "my_task_train": {
    "file_name": "train.json",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  },
  "my_task_val": {
    "file_name": "val.json",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  }
}
```

### 合并导出

```yaml
model_name_or_path: ./models/Qwen3-0.6B
adapter_name_or_path: ./output/qwen3_0.6b_lora
template: qwen3_nothink
finetuning_type: lora
export_dir: ./output/merged
export_size: 2
export_device: cpu
export_legacy_format: false
```

---

## 附录 A：术语表

| 术语 | 全称 | 含义 |
|---|---|---|
| SFT | Supervised Fine-Tuning | 用"输入-期望输出"对训练模型 |
| LoRA | Low-Rank Adaptation | 低秩适配器，仅训练少量参数 |
| QLoRA | Quantized LoRA | 基座 4bit 量化后做 LoRA |
| adapter | — | LoRA 训练产生的增量权重 |
| effective batch | — | per_device × grad_accum × 卡数 |
| cutoff_len | — | 单条样本最大 token 数 |
| GQA | Grouped Query Attention | Q 头数多于 KV 头，节省显存 |
| warmup | — | 学习率从低值逐步升到设定值的阶段 |

## 附录 B：训练前自检脚本

```bash
#!/bin/bash
# precheck.sh — 训练前跑一遍
echo "=== 1. GPU ===";      nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== 2. PyTorch ===";  python -c "import torch; print(torch.__version__, torch.version.cuda)"
echo "=== 3. 算力可用 ==="; python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print('OK')"
echo "=== 4. 数据条数 ==="; python -c "import json; print(len(json.load(open('data/train.json'))))"
echo "=== 5. 依赖版本 ==="; pip list 2>/dev/null | grep -Ei "transformers|peft|accelerate|llamafactory"
```
