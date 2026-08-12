# 知识库 Pilot 发布证据

## 范围

这份脱敏记录将六条 Production Agent Engineering Knowledge Base Pilot 绑定到 source revision `9c663fe3e2e76c91f0a0261bf3abf6615be15986`、冻结的 corpus 与 Acceptance Set hash，以及归一化 release-gate 结果。

## 门禁结果

记录中的 Pilot observations 达到完整 direct-query top-three evidence coverage 和 exact openable citation snapshots；paraphrase 与 combined-condition top-three coverage 完整，Boundary Query 未产生 unsupported answer。六条声明的 Pilot entry 均通过；失败 entry 会从发布与 coverage claim 中排除。

该里程碑仍为 `Pilot Edition`。由于声明的 First Edition target 大于六条 Pilot，它不具备 `First Edition` 标签资格。

## 安全证据

已评估五类 Agent-specific adversarial case：instruction override、forged tool call、secret extraction、unsafe code 和无证据回答压力。公开结果只包含归一化 case identity、outcome、pass/fail、citation count、failure classification、source revision 与 run identity。

## 限制

结果只适用于记录的 source revision、corpus 与 Acceptance Set hash、model identity 和有界 case，不构成通用 Agent security 或 retrieval-quality 声明。Private prompt、完整 answer、source excerpt、host、credential 与 corpus text 均不导出。
