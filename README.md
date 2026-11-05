# IMLS-Splatting 镜面表面重建：研究交接与重新规划

本仓库汇集截至 **2026-11-05** 的研究任务书、实验记录、代表图和相关源码，供网页端 GPT 制定新方案，再由 Codex 根据方案修改代码并测试。

## 从这里开始

**首先完整阅读：[研究任务书与项目交接](WEB_GPT_RESEARCH_BRIEF_ZH.md)。**

允许延续现有方法，也允许推翻 V4/V6/V32 等历史路线重新设计；重点解决不透明镜面物体与背景交界的缺损、不连续及错误表面问题。旧版本与无关工程必须保留。

可直接将下面的请求交给网页端 GPT：

> 请先完整阅读本仓库的 WEB_GPT_RESEARCH_BRIEF_ZH.md，再按其中要求查阅证据、源码和相关论文。针对不透明镜面物体的边缘破损重新规划，允许推翻现有方案，不局限于叠加局部损失。请分析根因、比较实质不同的路线并自我质疑，最终提供自包含、可下载的 IMPLEMENTATION_PLAN_ZH.md，详细到可以让 Codex 按文档修改代码和测试。遵循任务书第11节的交付规范，明确未验证假设与工程保护要求。无法读取的文件或未运行的实验须如实说明。

## 证据与源码

- [边界修复首轮实验结果](evidence/BOUNDARY_PROBE_RESULTS_ZH.md)：三个新候选均未达到预设改善门槛。
- [配对指标和选择结果](evidence/BOUNDARY_PROBE_SELECTION.json)、[完整探针指标](evidence/BOUNDARY_PROBE_SUMMARY.json)、[等格点扩域诊断](evidence/BOUNDARY_DOMAIN_AUDIT.json)。
- [RefReal完整结果](evidence/REFREAL_RESULTS_ZH.md)。
- [历史版本方法与代码记录](evidence/versions/)：优先阅读勘误，历史计划不代表本轮必须沿用。
- [相关源码](source/)：当前工作树的只读选摘，不是全依赖可运行发行版，也不代表全部历史版本源码。
- [六场景代表图](figures/)和[图像说明](figures/README_ZH.md)：首测试视角00000，不是最佳/最差视角筛选；图中指标表使用legacy LPIPS，标准口径见任务书。
- [附件说明](ATTACHMENTS_ZH.md)、[文件来源与SHA256](MANIFEST.json)。

**指标勘误：**旧报告的 `boundary_f2` / F2 指2像素容差下的边界F1，即 F@2px，不是 F-beta=2。新版任务书已明确区分。

## 阅读边界与来源

基础方法：[SilenKZYoung/IMLS-Splatting](https://github.com/SilenKZYoung/IMLS-Splatting)；前置初始化基础：[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)。其他借鉴关系、论文链接和独立实现范围见任务书及方法记录。

本仓库不是上述官方项目，也不为历史负结果或未来方法宣称成功。选摘源码保留已有版权标识；本次未统一授予新的软件许可证，各第三方部分的使用条件应核对其来源。未包含账号凭据、训练数据、完整模型、编译依赖和原项目Git历史。

旧附件中的本地/服务器路径用于后续执行端定位；网页端无法凭路径访问文件。旧文档指向未打包文件的链接不能视为材料已提供。若网页端不能完整读取GitHub，可用页面的 **Code → Download ZIP** 下载后上传所需文件。
