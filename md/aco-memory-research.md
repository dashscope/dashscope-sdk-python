# 蚁群算法（ACO）应用于智能体记忆检索 —— 定向调研

> ⚠️ 保密：含未申请专利的技术方案，申请递交前勿公开、勿提交任何 git 仓库。
> 来源：deep-research 工作流（runId: wf_13fb4357-cbf，5 角度、22 来源、94 条 claim 抽取、25 条对抗验证**全部确认**、0 驳回）。ACO 教科书内容经 Dorigo & Stützle 原书 PDF（MIT Press, 321 页）逐字核对。
> 证据强度标注：【教科书】= 经典定义性材料；【benchmark】= 公开基准实测；【宣称】= 论文/项目方自报；【代码核实】= 开源仓库逐行验证。

## 0. 一页结论

1. **直接先例存在（关键问题的直接回答）**：开源系统 [sentAInce](https://github.com/dcnconsult/sentAInce)（Apache-2.0，2026-07 创建、2026-09 活跃）公开实现了"agent 行为在持久记忆图上沉积信息素、落点分类后信息素引导游走取 top-k 上下文注入"。**存在性证据强（代码级核实），有效性证据为零**（单人项目、3 stars、无同行评审、自身对照实验未达显著性门槛）。
2. **学术上显式信息素路线基本空白**：2026-01 版 GraphRAG 功能综述全文无 pheromone/ant colony/stigmergy（缺席证据）；最接近的 agent 记忆加权工作 HAGE（2026-05）用 RL 而非信息素、全文无相关术语。
3. **机制工具箱全部有成熟参照**：经典教科书给出转移/挥发/MMAS 公式与参数；落点定位与 τ/η 融合有 NAACL 2015 独立 benchmark 支撑（融合后 91.77%，单信号显著更差——信号互补"缺一不可"有实证）。
4. **风险闭环完整**：马太效应有小规模实证观察，MMAS 界/挥发/聚合约束/精英强化缓解手段齐备；**对抗性污染防御在所有已核实来源中完全空白**（既是风险也是专利空位）。
5. **专利含义**：宽泛框架（行为沉积信息素 + 落点 + 游走取上下文）已被 sentAInce 覆盖，独权必须收窄到具体区别特征组合；窗口在快速收窄（sentAInce 两个月前才创建），要申请必须快。详见第 7 节。

## 1. 目标场景与设计约束

智能体（终端编程助手为代表）在持久化记忆图谱上运行。给定一个 query：落点定位 → 信息素引导的邻域游走 → top-k 紧凑子图（~10^3 tokens）注入上下文。约束：检索代价与库规模解耦；图谱伴随会话自动构建、在线演化；冷启动可用；防高频路径垄断。

## 2. ACO 经典机制【教科书，经原书逐字核实】

来源：Dorigo & Stützle《Ant Colony Optimization》（MIT Press）原书 PDF + Scholarpedia。

### 2.1 核心公式

- **随机比例转移规则**（式 3.2）：p_ij^k = τ_ij^α · η_ij^β / Σ_l τ_il^α · η_il^β。α=0 退化为随机贪婪；β=0 导致纯信息素自放大停滞——**α/β 都不可缺**。
- **蒸发规则**（式 3.3）：τ ← (1−ρ)·τ，0<ρ≤1。使未被选择的边指数衰减，防止无限累积、实现"遗忘坏决策"。

### 2.2 经典参数区间（原书 Box 3.1，TSP 无局部搜索）

| 参数 | AS/EAS | ASrank | MMAS | ACS |
|---|---|---|---|---|
| α | 1 | 1 | 1 | 1 |
| β | 2~5 | 2~5 | 2~5 | 2~5 |
| ρ | 0.5 | 0.1 | **0.02** | 0.1 |
| m（蚂蚁数） | n | n | n | 10 |
| 其他 | EAS: e=n | w=6 | —— | q0=0.9, ξ=0.1 |

### 2.3 MMAS（防早熟收敛的关键变体）

- 每边信息素钳制在 [τmin, τmax]；τmax = 1/(ρ·C^bs) 随最优解更新，τmin = τmax/a；
- **实验证明 τmin 比 τmax 更关键**（Stützle 1999）——τmin 保证任何边始终保留非零被探索概率，这正是防马太效应的理论机制；
- 接近停滞时触发**信息素重初始化**（重置到 τmax）；
- 初始化约定：MMAS 将信息素**初始化到 τmax**（促进早期探索），与 AS 的低常数初始化相反——冷启动设计应采 MMAS 约定。

### 2.4 概念映射表（ACO → 记忆图谱）

| ACO 概念 | 本方案对应 |
|---|---|
| 蚂蚁 | 一次 query 的上下文提取游走 |
| 节点 | 知识实体（代码符号/文档/知识点/事项） |
| 边信息素 τ | 行为边权：共现检索频次 × 采纳率 |
| 启发值 η | 静态结构边强 × 与当前 query 的语义相似度 |
| 路径质量 | 任务完成信号（资格迹/桶队式沿路径折扣回传 γ^Δ） |
| 挥发 ρ | 时间衰减（长期记忆取 MMAS 式低挥发 0.02–0.1） |
| MMAS [τmin,τmax] | 防老边垄断、保新边机会；停滞重初始化 |
| ACS q0=0.9 / ξ=0.1 | 游走利用/探索配比；经过即微量局部挥发 |

## 3. 先例排查（关键问题：有无直接先例）

### 3.1 直接先例：sentAInce【代码核实，存在性强证据】

[sentAInce](https://github.com/dcnconsult/sentAInce)（Apache-2.0, v0.1.12，2026-07 创建、2026-09 活跃，3 stars / 1 fork，单人项目）：

- **沉积**：仅当动作被验证成功（Bash exit 0）时沿路径在 verb-altitude 有向图上沉积 τ——"earned memory"：读取不沉积、重复不沉积、失败不沉积；含衰减、剪枝下限、每类边上限、资格迹信用分配（γ^Δ，bucket-brigade 式）；
- **检索**：UserPromptSubmit 将 prompt 分类到 goal-class（落点），splice(k=8) 从该类最强边源节点做**贪婪 widest-path 游走**（`cur = max(outs, key=lambda x: x[1])[0]`），top-k τ 边渲染为 additionalContext 注入；
- **与本案同构点**：落点 + 信息素引导邻域游走取上下文（投票 2-1，分歧仅在"落点"对应关系的解释强度，代码事实无争议）；
- **重大限定**：无同行评审、无外部采用、自身对照 flock 实验未达显著性门槛——**"先例存在"证据强，"先例有效"证据为零**；
- **法律意义**：公开 GitHub 仓库构成现有技术，宽泛框架独权不可行（见第 7 节）；Apache-2.0 许可使其代码可作为工程参照。

### 3.2 功能等价但机制不同：HAGE【benchmark 级论文，代码无关】

[HAGE（arXiv 2605.09942，2026-05）](https://arxiv.org/html/2605.09942v2)：在 agent 记忆图谱上用**下游任务奖励的 RL（策略梯度 + 边锚定正则化）**联合优化路由行为与边表示——"由检索/路由行为的结果强化图边权供后续 query 使用"在功能层面已有先例。
- 游走打分 S(n_j|n_i,q) = λ·cos(v_j,q) + (1−λ)·w_ij(q)：语义相似 × 学习边权的融合，与 τ/η 融合在**抽象设计模式**层面同构，可直接参照；
- **两点不可照抄**：其 w_ij(q) 是 query-conditioned 动态表示（非 ACO 式持久标量）；融合是学习系数线性加和（非 τ^α·η^β 乘性）；
- **全文零信息素术语**——支持"显式 ACO 路线在学术文献空白"的判断。

### 3.3 ACO×LLM 先例但场景不同：ACO-ToT【宣称，存在性代码级】

[ACO-ToT（arXiv 2501.19278，2025-01）](https://arxiv.org/abs/2501.19278)：经典转移规则与挥发迁移到 LLM 思维树推理；GSM8K/ARC/MATH 达 84.2%/88.9%/22.6%（对最弱基线 CoT 提升 28.6 点，**但对最强基线 IRPO 仅 +1.8~2.6 点**，论文自报、单模型、无复现）。**关键限定：信息素仅作用于单次推理任务内部的推理树，不跨 query 持久化**——经两次全文核查确认未应用于记忆/检索/KG 场景，不构成核心设想先例。

### 3.4 IR/推荐领域的历史先例【一手论文机制核实】

| 工作 | 机制 | 与本案差异 |
|---|---|---|
| **NaïveRank**（约 2014，Brunel 大学库） | (query,document) 对上沉积信息素：点击即 +1，排名完全由信息素决定；指数半衰期挥发 φW·(1/2)^((t−φT)/δ) 在每次强化前执行，明确用于对抗正反馈自强化 | 载体是点击日志的 query→doc 二部结构，非记忆图谱邻域游走；注意作者在自己的实验中**关闭了挥发**，防收敛是设计 rationale 而非实验结论 |
| Web 搜索个性化（Procedia CS 2021） | 每用户偏好画像，画像不足时借用他人画像（stigmergy 类比） | 载体为用户画像而非图边权；摘要无真实 ACO 算子（部分类比） |
| Ant Collaborative Filtering（Appl. Sci. 2020） | 信息素交换建模评分、挥发建模 concept drift | 推荐场景，非 agent 记忆 |

### 3.5 主流文献的缺席证据与静态基线

2026-01 版 [GraphRAG 功能综述（arXiv 2504.10499v2）](https://arxiv.org/html/2504.10499v2) 768KB 全文 grep：pheromone/ant colony/stigmergy 零命中。主流范式恰为本案的静态对照组：query 实体对齐到 KG 节点后作为 **Personalized PageRank 种子做邻域扩散**（HippoRAG, NeurIPS 2024）——静态/扩散权重而非信息素权重。
**限定**：缺席证据限于综述覆盖范围；**本轮未系统检索专利库**，工业内部实现未知。

## 4. 算法设计草案

### 4.1 设计依据的独立 benchmark 证据链

[Pershina/He/Grishman, NAACL 2015（N15-1026）](https://aclanthology.org/N15-1026.pdf)：PPR 实体消歧在 AIDA 27,816 个提及上达 **91.77% micro**（超当时 SOTA）。消融证明**信号互补缺一不可**：单用静态启发值 62.61%、单用图游走一致性 85.56%、融合后 91.77%——为 τ（游走证据）× η（语义/静态启发）融合设计提供独立支撑。聚合约束 c1/c2（忽略同实体竞争候选、每实体只取最大贡献候选）将准确率从 90.26% 提至 91.77%，**机制是过滤错误候选注入游走的噪声**——直接对应抗污染设计。
限定：其最终组合为加性 score 且 iSim 兼作 PPR 重启分布，对乘性 τ^α·η^β 的支撑在信号互补性层面而非公式同一。

### 4.2 在线更新（伴随会话自动执行）

```
on tool_call(query, hits, adopted, task_context):
    for (a, b) in co_occurred(hits):                    # 共现沉积
        Δτ_ab += adopt_weight(adopted) * γ^depth(a, b)

on task_done(success):                                  # 精英强化：仅成功路径
    if success:
        for edge, Δ in retrieval_path.edges_with_distance():
            τ_edge += Q * γ^Δ                           # 资格迹信用回传

on tick(Δt):                                            # 或懒挥发：读取时按 Δt 折算
    τ ← clip(τ * (1−ρ)^(Δt/τ_unit), τmin, τmax)         # MMAS 钳制
    if stagnation_detected(): τ ← τmax                  # 停滞重初始化（MMAS）
```

### 4.3 查询提取（落点 + 信息素引导游走）

```
extract(query, token_budget):
    seeds = entity_link(query)                          # 落点（PPR 消歧式，见 4.1）
    if not seeds:
        ctx = fallback_grep_vector(query)               # 回退通道
        write_back_candidate_nodes(ctx)                 # 新实体回写，图谱自动生长
        return ctx
    frontier, ctx, cost = seeds, [], 0
    while cost < token_budget:
        cand = edges(frontier)
        cand = apply_aggregation_constraints(cand)      # c1/c2 抗噪（4.1）
        p_ij ∝ τ_ij^α · η_ij^β                          # τ=行为边权, η=语义×静态
        next = argmax(p) if rand() < q0 else sample(p)  # ACS 利用/探索
        ctx.append(next); cost += tokens(next)
        frontier = next
    return ctx
```

### 4.4 参数初始值（教科书 Box 3.1 核实值 → 本场景外推）

| 参数 | 教科书经典值 | 本场景建议 | 依据 |
|---|---|---|---|
| α | 1 | **1.0** | Box 3.1 |
| β | 2~5 | **2** | Box 3.1 下沿；长期行为信号应占主导 |
| ρ | AS 0.5 / ACS 0.1 / MMAS 0.02 | **0.02–0.1** | 长期在线记忆取 MMAS 式低挥发 |
| q0 | 0.9 (ACS) | **0.85–0.9** | ACS 经典 |
| ξ（局部挥发） | 0.1 (ACS) | 0.1（可选） | ACS 经典 |
| [τmin,τmax] | τmax=1/(ρ·C^bs)，τmin=τmax/a | MMAS 钳制；**τmin 更关键**（保探索下限） | Stützle 1999 实验结论 |
| τ0 初始化 | AS 低常数 / **MMAS 初始化到 τmax** | **τ0=τmax** | MMAS 约定：高初始值促进早期探索，缓解冷启动 |
| 游走预算 | —— | ~10^3 tokens | HippoRAG2 PPR 紧凑子图实测量级（上轮调研） |

## 5. 评测设计

- **必含基线**（有 benchmark 依据）：A 纯向量 RAG；B 静态 Personalized PageRank 图（HippoRAG 式，主流范式对照）；C 信息素加权（本案）。
- **冷启动/覆盖率是受控变量**：KG-GraphRAG 因建图覆盖率仅 ~65%（HotPotQA 65.8%、NQ 65.5% 的答案实体入图）在 QA 上普遍逊于向量 RAG（HotPotQA F1：KG-GraphRAG-Triplets 25.02 vs RAG 60.04）——**这是稀疏/冷启动风险的 benchmark 级直接证据**，评测必须控制图覆盖率变量（[arXiv 2502.11371v2 / OpenReview K6N6gCCYcb](https://arxiv.org/html/2502.11371v2)，ACM 已发表）。
- 指标：recall@k / MRR、token 成本、延迟、**收敛轮数**（行为加权需多少轮交互才显著优于静态基线 B）、抗噪恢复。
- 消融（参照 ACO-ToT 对照设计范式）：去挥发、去 MMAS 钳制、去精英回传、τ-only（β=0，应现自放大停滞）、η-only（α=0）。
- 自建任务集（代码场景无现成 benchmark）：跨文件符号定位 ×20、影响面分析 ×20、架构问答 ×20。

## 6. 风险与缓解（实证观察—缓解闭环）

| 风险 | 实证证据 | 缓解（均有支撑） |
|---|---|---|
| 正反馈失控/马太效应 | NaïveRank 8 用户 ×6 任务实验：目标页升至首位后被更多点击、信息素持续增加，可能收敛次优（作者自述初步结果、hedged，且其自评在该静态领域 negligible）【宣称】 | MMAS [τmin,τmax]（τmin 保探索下限，实验证明比 τmax 更关键）+ 停滞重初始化【教科书】；指数半衰期挥发【宣称】 |
| 冷启动/覆盖稀疏 | KG 覆盖率 ~65% 致检索失效（HotPotQA F1 25.02 vs 60.04）【benchmark】 | 静态底图 + τ0=τmax 高初始探索 + 回退通道与回写生长 |
| 错误候选噪声 | 聚合约束 c1/c2 使准确率 90.26→91.77%【benchmark】 | 游走前过滤低置信节点贡献 |
| 规模扩展 | 无已发表方案 | 懒挥发（读取时折算）+ 局部子图更新 + SQLite 单文件 |
| **对抗性污染** | **所有已核实来源中完全无针对性防御研究（空白）** | 本案自拟：精英强化仅限任务成功路径；采纳信号校验；τmin 保底使污染边可随挥发恢复——**空白即专利空位（见第 7 节）** |

## 7. 专利新颖性含义（修正版，重要）

**宽泛框架已被覆盖**：sentAInce 公开了"agent 行为成功沉积信息素于持久记忆结构 + 落点分类 + 信息素引导贪婪游走取 top-k 上下文注入"——"行为信息素记忆"作为大概念的独权不再可行。

**仍可主张的区别特征组合**（独权须收窄至此层级）：

| 区别维度 | sentAInce | 本案 |
|---|---|---|
| 转移规则 | 贪婪 widest-path（max τ） | **随机比例规则 τ^α·η^β + ACS q0 利用/探索**（教科书规则，含探索防停滞） |
| 抗早熟约束 | 每类边上限 + 剪枝下限 | **MMAS [τmin,τmax] + 停滞重初始化**（有理论依据的界） |
| 落点粒度 | goal-class 粗分类 | **实体链接到具体节点**（PPR 消歧式，91.77% benchmark） |
| 沉积信号 | 仅验证成功（exit 0） | **共现频次 × 采纳率 × 成功奖励多信号**；失败/读取的差异化处理可再设计 |
| 图内容 | verb-altitude 动作图（"什么动作有效"） | **知识实体图**（代码符号/文档/知识点/事项）+ 静态/行为双层边 |
| 上下文构造 | 固定 k=8 splice | **token 预算约束的紧凑子图**（~10^3 tokens，PPR 式） |
| 失效管理 | 衰减 + 剪枝 | **双时序失效（保留历史）+ 变更级联失效传播** |
| 场景 | 编程 CLI 单场景 | 多场景泛化（编程/办公/学习/生活） |

**附加空位**：对抗性污染防御在所有已核实来源（含 sentAInce）中均为空白，可作分案储备。

**两点诚实警示**：
1. **有效性证据缺失是全行业现状**（唯一直接先例自身实验未达显著性）——专利不要求实证优越性，但产品投入前必须用第 5 节的评测自行验证"信息素加权 > 静态 PPR"；
2. **窗口在收窄**：sentAInce 两个月前创建、HAGE 四个月前发表，该领域演进以月计。且本轮未系统检索专利库——正式递交前的 CNIPA/incoPat 查新不可省。

## 8. 开放问题

1. 信息素在线沉积（sentAInce 式）与 RL 离线训练（HAGE 式）在同一记忆图谱上的头对头对比完全缺失——无 recall@k、成本、收敛速度、抗噪的任何实验证据；
2. 专利与工业界未公开实现未知（本轮未系统检索专利库）；
3. 理论性质未明：稳态下 τ 加权邻域游走与 Personalized PageRank 的数学关系（是否等价于某种自适应加权 PPR）、收敛条件，均无已发表分析——**这本身是可发表的研究点**；
4. 百万级边规模的挥发调度成本（全边周期蒸发 vs 访问时惰性衰减）无已发表工程方案。

## 9. 来源清单

- Dorigo & Stützle, *Ant Colony Optimization*, MIT Press（[原书 PDF](https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf)）；[Scholarpedia: ACO](http://www.scholarpedia.org/article/Ant_colony_optimization)
- [sentAInce（直接先例，代码核实）](https://github.com/dcnconsult/sentAInce)
- [HAGE, arXiv 2605.09942](https://arxiv.org/html/2605.09942v2)
- [ACO-ToT, arXiv 2501.19278](https://arxiv.org/abs/2501.19278)
- [NaïveRank（Brunel 大学库全文）](https://bura.brunel.ac.uk/bitstream/2438/13248/1/Fulltext.pdf)
- [Web 搜索个性化（Procedia CS 2021）](https://www.sciencedirect.com/science/article/pii/S1877050921011686/pdf?md5=9980107be61f5ba6365ba6cb56a0b274&pid=1-s2.0-S1877050921011686-main.pdf)；[Ant Collaborative Filtering (Appl. Sci. 2020)](https://www.mdpi.com/2076-3417/10/20/7245)
- [GraphRAG 功能综述 2026-01, arXiv 2504.10499v2](https://arxiv.org/html/2504.10499v2)；[HippoRAG, arXiv 2405.14831](https://arxiv.org/abs/2405.14831)
- [Pershina et al., NAACL 2015 (N15-1026)](https://aclanthology.org/N15-1026.pdf)
- [GraphRAG 系统评测（OpenReview K6N6gCCYcb / arXiv 2502.11371v2）](https://arxiv.org/html/2502.11371v2)
- [Bucket Brigade 机制（IDSIA 存档）](https://people.idsia.ch/~juergen/bucketbrigade/node2.html)
