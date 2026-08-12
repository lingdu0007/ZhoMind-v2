# 知识库 First Edition 候选证据

## 范围

这份有界公开投影记录六个声明主题下三十条已复核候选 corpus。authority corpus、Acceptance Set 文本、source snapshot、Gold Evidence 和完整 editorial artifact 仍保留在 Private Editorial Repository。

## 发布门禁

认证 production Compose 与 live-provider run 完成了 120 个归一化 case，但未通过 Knowledge Base Release Gate：direct top-three coverage 为 3.33%，paraphrase 与 combined-condition coverage 为 0%，三个 Boundary Query 产生 unsupported answer，answerable case 的 exact openable citation snapshot coverage 为 17.78%。三十条 target entry 全部保持未发布，因此该候选不能标记为 `First Edition`。

## 生产限制

在有界 observation window 内，provider 有四个 case 超时、84 个 case 被限流。这些 case 记录为 `Generation Unavailable`；它们导致 entry 验收失败，但不计为 unsupported answer。该结果不是 retrieval-only diagnostic，也不以 disposable local 结果替代 production evidence。

## 维护

monthly source check、90-day Version Mapping review、six-month Stable Principle review 和 quarterly sampled re-acceptance 只创建 review work。记录的 90-day run 创建了 30 条 Version Mapping review work 和 6 条 quarterly re-acceptance work，自动发布数为 0。

## 限制

结果只适用于记录的 source revision、hash、model identity、timeout 与 observation window。Private corpus prose、query、answer、excerpt、user、host、credential 和 operational detail 均被排除。
