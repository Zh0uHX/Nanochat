# 简历项目表述（科研 / 企业实习）

## 推荐项目名

**可复现的端到端大语言模型训练系统：分布式 SFT 数据管线与精确断点恢复**

时间建议写为 `2025.12–至今`；如果只希望呈现已完成阶段，可写
`2025.12–2026.07`。项目需明确标注“基于开源 nanochat 二次开发”，避免把
Transformer、Muon、Value Embedding、KV Cache 等上游能力写成个人原创。

## 通用技术栈

Python、PyTorch、DDP、Muon/AdamW、BF16、Tokenizer、KV Cache、pytest、GitHub
Actions

## 科研实习版

- 基于 nanochat 设计**有状态分布式 SFT 装箱器**，实现 sequential、
  first-fit、length-bucket 与 best-fit 四种策略；采用确定性 rank 分片并保存
  预取缓冲区、epoch、游标与配置指纹，使中断前后的后续 token batch 可精确复现。
- 修复原 SFT 管线丢弃 tokenizer 逐 token 监督掩码的问题，将 mask 与 token
  同步装箱和恢复，确保用户提示、工具输出与 padding 不参与 assistant-only
  next-token loss；加入超长样本 `truncate/error` 策略及截断、利用率统计。
- 设计固定预算消融协议，对比四种装箱策略的 padding ratio、CPU p50/p95
  延迟、训练吞吐、MFU、验证 BPB 与下游任务指标；已提供可复现 benchmark
  与结果模板，GPU 结论在完成多 seed 实验前不作宣称。
- 完善实验溯源：checkpoint/eval 记录配置 SHA-256、Git commit、dirty/diff
  状态及包含未跟踪源码的工作树内容哈希，并对历史评测报告与 checkpoint
  step 不一致的结果作显式排除。

## 企业 / 系统实习版

- 重构分布式 SFT 数据管线，实现确定性 rank 分片、buffer-aware 精确恢复、
  四种 packing 策略及 assistant-only loss mask，消除重启后的数据漂移和
  prompt token 误训练风险。
- 将 checkpoint 拆分为模型、rank-sharded optimizer 与 rank-local packer
  状态；采用同目录临时文件原子替换，并在全部 rank 文件通过校验后发布完成
  标记，避免将中断产生的半写入 checkpoint 误判为可恢复状态。
- 为 Muon/AdamW 增加 `torch.compile` 快路径与 eager 兼容路径，提供数值一致性
  测试和 CUDA microbenchmark；A100 默认使用经目标环境验证后的实现，避免
  将兼容性假设硬编码进训练脚本。
- 建立 CPU/CI 回归、packing/optimizer/KV-cache benchmark 与 8×A100
  运行脚本；历史完成 1.68B 参数模型的 7.806B-token BF16 训练（8×A100
  80GB，记录 MFU 45.86%），新实现的性能收益待受控实验单独验证。

## 一页简历压缩版

- 基于 nanochat 重构分布式 SFT：自研有状态 best-fit/length-bucket packer，
  保存预取缓冲区与 rank 游标，实现确定性数据分片和后续 batch 精确恢复。
- 贯通 tokenizer supervision mask、packing 与 loss，排除 prompt/tool-output/
  padding token；实现超长样本策略及 packing efficiency、truncation 指标。
- 实现 rank-sharded optimizer/packer checkpoint、原子写入和全 rank 完成
  标记，并以配置/Git/工作树哈希保证训练与评测可追溯。
- 完成 1.68B 模型 7.806B-token BF16 训练（8×A100 80GB，MFU 45.86%）；
  建立 packing、优化器和 KV Cache 消融框架，性能增益待多 seed GPU 实验验证。

## 面试时应能解释

1. 为什么只保存 dataset cursor 不能恢复 best-fit packer：预取缓冲区已经读取但
   尚未消费，cursor 与下一批数据不是一一对应。
2. 为什么 mask 要按 target 位置使用：输入为 `tokens[:-1]`，目标为
   `tokens[1:]`，因此 loss mask 同样取原 token mask 的 `[1:]`。
3. 为什么需要 checkpoint 完成标记：原子文件只能保证单文件完整，不能保证
   多 rank 的文件集合完整；barrier 后的 marker 才定义一次有效提交。
4. 装箱率与模型质量为何要分开评估：packing 改变样本邻接顺序，吞吐提升不等于
   validation BPB 或下游准确率不变。

## 不应写入简历的表述

- “自主设计 Transformer / Value Embedding / RMSNorm / KV Cache / Muon”：
  这些属于上游实现。
- “支持 FSDP、FP16”：当前项目没有对应实现或验证。
- “packing 提升训练速度 X% / KV Cache 加速 X 倍”：在目标 GPU、固定
  checkpoint 和受控协议产生结果文件前，不应填入数字。
- 使用 step 169150 的历史 CORE/BPB：该报告与现存 step 14889 checkpoint
  不匹配，不能作为已验证结果。
