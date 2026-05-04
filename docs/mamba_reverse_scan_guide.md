# Mamba_Hybrid_Degree_Noise_Bucket 机制与反向扫描接入指南

## 现有分支机制（`graphgps/layer/gps_layer.py`）

`Mamba_Hybrid_Degree_Noise_Bucket` 分支核心流程：

1. 计算每个节点的度 `deg`。
2. 加入随机噪声 `deg_noise`，形成扰动排序键 `deg + noise`。
3. 随机分桶（`NUM_BUCKETS` 个桶），将节点打散到多个子序列。
4. 每个桶内按 `(batch, noisy_degree)` 排序。
5. 将每个桶分别 `to_dense_batch`，送入 `self.self_attn`（Mamba）编码。
6. 将各桶输出拼接后，按原全局节点 index 反排回原图顺序。
7. 推理阶段重复 5 次随机噪声+分桶，并取平均以降低随机性。

这种做法等价于：
- 用 degree/noise 构造「局部可重复、全局随机」的序列视角；
- 用 bucket 降低超长序列建模难度并增加顺序扰动多样性；
- 用 test-time averaging 稳定输出。

## 如何加入“反向扫描 Mamba”（bidirectional scan）

推荐最小侵入实现：在每个桶内同时跑**正向序列**和**反向序列**，再融合。

### 关键点

- 正向：`y_fwd = self.self_attn(h_dense)`
- 反向输入：`x_rev = torch.flip(h_dense, dims=[1])`
- 反向输出翻回正序：`y_rev = torch.flip(self.self_attn(x_rev), dims=[1])`
- 融合：`y = 0.5 * (y_fwd + y_rev)`（或拼接后线性层）

注意：
- 必须在 dense/padded 维度（`dim=1`）翻转；
- 翻转后依旧使用同一 `mask` 取有效节点：`[mask]`；
- 分桶后回填原顺序逻辑保持不变。

### 可直接替换的桶内编码片段

将桶循环中的：

```python
h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
h_dense = self.self_attn(h_dense)[mask]
```

替换为：

```python
h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])

# forward scan
y_fwd = self.self_attn(h_dense)

# reverse scan（沿序列维翻转）
h_dense_rev = torch.flip(h_dense, dims=[1])
y_rev = self.self_attn(h_dense_rev)
y_rev = torch.flip(y_rev, dims=[1])

# fuse
y = 0.5 * (y_fwd + y_rev)

h_dense = y[mask]
```

### 进一步建议

- 若你希望保持参数量可控，可先复用同一个 `self.self_attn` 同权双向扫描（如上）。
- 若要增强表达能力，可新增 `self.self_attn_rev`（独立参数）并在 `__init__` 初始化。
- 若担心反向信息过强，可使用可学习门控：
  `y = alpha * y_fwd + (1 - alpha) * y_rev`，其中 `alpha = sigmoid(w)`。

## 独立反向 Mamba vs 复用同一个 Mamba

- **参数量与显存**
  - 复用同一个 Mamba：参数不增加，训练显存和优化器状态开销更小。
  - 独立反向 Mamba：参数约翻倍（前向一套 + 反向一套），显存与训练开销更高。

- **表达能力**
  - 复用同一个 Mamba：本质是“同一核”看两个方向，带有对称约束，容量更受限。
  - 独立反向 Mamba：前后方向可学习不同动态（例如前向偏局部、反向偏全局），更灵活。

- **正则化与过拟合**
  - 复用同一个 Mamba：参数共享天然带正则效果，小数据集通常更稳。
  - 独立反向 Mamba：容量更大，若数据量不足更容易过拟合，需要更强正则。

- **训练稳定性**
  - 复用同一个 Mamba：优化目标更集中，通常更容易收敛。
  - 独立反向 Mamba：需要同时协调两套动力系统，可能更依赖学习率/权重衰减调参。

- **实践建议**
  - 数据规模小或资源紧：优先“复用同一个 Mamba”。
  - 数据规模大、追求上限：尝试“独立反向 Mamba”，再配合 gated fusion 与正则策略。

## FAQ：为什么“独立反向 Mamba”和“复用同一个 Mamba”看起来参数量一样？

常见原因通常不是实现原理问题，而是**配置或统计口径**问题：

1. **没有真正开启反向分支**
   - 若 `enable_reverse_mamba=False`，不会创建反向 Mamba，自然参数不变。

2. **统计时机不对**
   - 需要在模型初始化后统计：`sum(p.numel() for p in model.parameters())`。
   - 如果你统计的是旧实例（未开启 reverse），结果会一样。

3. **统计口径只看某个子模块**
   - 若只统计 `self.self_attn`，不会包含 `self.self_attn_reverse`。
   - 应统计整个 `GPSLayer` 或整个模型。

4. **分支未命中**
   - 当前实现仅在 Mamba 类型且开启 reverse 时创建独立反向模块。
   - 若模型类型不在该条件内，反向独立参数不会被创建。

快速自检建议：

- 打印 `hasattr(layer, 'self_attn_reverse') and layer.self_attn_reverse is not None`；
- 分别打印 `sum(p.numel() for p in layer.self_attn.parameters())` 与
  `sum(p.numel() for p in layer.self_attn_reverse.parameters())`；
- 再打印 `sum(p.numel() for p in layer.parameters())` 对比是否接近翻倍。
