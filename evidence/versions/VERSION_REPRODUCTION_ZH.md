# Ref-NeRF修改版本复现说明

2026-09-28勘误：V39发现旧训练导出按shuffle后的uid编号，旧离线几何工具却按原metadata顺序配相机。依赖这种配对的历史几何筛选证据须重算；历史归档用于追溯旧行为，不代表这些监督有效。新实验必须使用`views.json`原始帧身份检查，详见两份累计记录的V39部分。测试相机未shuffle，当前未发现该问题影响V32测试RGB配对。

## 保存策略

本项目同时采用两种方式保留版本：

1. `version_snapshots/IMLS-Splatting_v4_code.tar.gz`与`IMLS-Splatting_v6_code.tar.gz`是排除数据、权重、训练输出和缓存的独立完整源码归档，SHA-256记录在同目录。
2. V7以后保留版本执行脚本和默认关闭的消融参数，但这些并不能保证当前代码精确复现任意旧版本：CUDA修复、掩码读取、插值及默认参数都可能改变行为。严格复现应使用当时的完整源码快照、对应CUDA构建、相同检查点和环境；不能仅把当前代码加上旧脚本就称为“精确旧版本”。

服务器中的`/root/autodl-tmp/gaussian-shader`及`gaussian_shader4090`环境明确不属于本项目，任何归档、同步、清理和实验均不得涉及。

## 主要版本入口

|版本|入口脚本|用途/状态|
|---|---|---|
|V4|`run_refnerf_v4_finalize.sh`|几何修复及法线/深度最终图；独立归档可复现|
|V6|`run_refnerf_v6_exact.sh`|直接球面反射贴图；独立归档可复现|
|V7|`run_refnerf_v7_geometry.sh`、`run_refnerf_v7_appearance_full.sh`|几何—反射两阶段解耦|
|V8|`run_refnerf_v8_zdisparity.sh`|camera-Z视差消融，已否决|
|V9|`run_refnerf_v9_bilateral_probe.sh`|点域双边平面投影消融，已否决|
|V10|`run_refnerf_v10_mesh_taubin_refine.sh`后接V7外观脚本|历史几何/外观参考基线；非当前最终版本|
|V11|`run_refnerf_v11_normal_cvar_refine.sh`|最差法线风险消融，已否决|
|V12|`run_refnerf_v12_sdf_taubin_refine.sh`|隐式场Taubin消融，已否决|
|V13|`run_refnerf_v13_reflection_detail.sh`|高分辨率方向贴图消融，收益可忽略|
|V14|`run_refnerf_v14_specular_normal.sh`|反射专用切向微法线消融，收益可忽略|
|V15|`run_refnerf_v15_silhouette_release.sh`|强化不透明轮廓召回消融，局部改善但整体否决|
|V16|`run_refnerf_v16_visual_hull_probe.sh`|训练alpha外轮廓补点；car有效但跨场景未通过，作为消融|
|V17|`run_refnerf_v17_hull_geometry_refine.sh`|外壳支撑几何校正消融，已否决|
|V18|`run_refnerf_v18_hull_appearance.sh`|外壳支撑后的统一反射外观重拟合，已否决|
|V25|`run_refnerf_v25_adaptive_surface_support_probe.sh`后接V22外观脚本|保留的car/coffee阶段版本；非六场景最终版|
|V32|`run_refnerf_v32_alpha_audit.sh`|独立alpha及白底合成修复；同代码零更新和100次几何更新对照，非已达标版本|
|V33|`run_refnerf_v33_pixel_normal_probe.sh`|IMLS逐像素几何法线；收益极小，已否决，开关默认0|
|V34|`run_refnerf_v34_surface_continuity_probe.sh`|训练法线连续/切平面约束；RGB退化，已否决|
|V35|`run_refnerf_v35_reflection_probe.sh`|固定几何反射分解；配对收益不可见，已否决|
|V36候选|`run_refnerf_v36_surface_support_probe.sh`、`run_refnerf_v36_domain_pair.sh`|训练alpha/GT法线/第一表面筛选缺边点；car/coffee同网格20视角探针有PSNR收益，但未完成六场景，非正式归档版本|
|V37数值修复|`run_refnerf_v36_domain_pair.sh`、`test_mesh_domain_parity.py`|视觉外壳统一读取ball独立alpha，自动扩域保持旧网格奇偶以避免半体素错位；六场景20视角门禁已完成但未达到目标|
|V38|`run_refnerf_v38_visible_contour.sh`|可见轮廓距离200步配对；轮廓微增、法线/LPIPS退步，未采纳；默认权重0|

每个脚本通过`SCENES`选择场景，但一轮正式结论必须对所有场景使用同一组方法超参数；禁止从不同版本逐场景拼接最优指标。复现前应确认脚本引用的前置PLY存在。保留的训练结果只应包括每个被采纳阶段的最终候选；否决探针输出可删除，但脚本、代码开关、指标与两份累计记录保留。

截至2026-09-27，V32六场景源码归档为`version_snapshots/IMLS-Splatting_v32_alpha_audit_20260926_code.tar.gz`，对应六场景200测试视角输出为`output/{scene}_v32_mask_geometry_candidate`。V33–V35未达到值得正式留档的门槛。V36另有`version_snapshots/IMLS-Splatting_v36_first_surface_candidate_20260927_code.tar.gz`作为**候选源码**，并保留car/coffee筛选源点与`analysis_next_round/v36_surface_support`证据；它不是六场景最终模型。重跑V36需要V32输出PLY以及训练数据，并按脚本先导出全部训练视角几何缓冲；源码归档不包含数据、权重或输出。历史`metrics.py`的LPIPS输入口径不标准；标准[-1,1] LPIPS-VGG的重算工具是`evaluate_lpips_standard.py`，V6/V32六场景每场景200视角结果分别保存在对应`renders_iter_1/test/standard_lpips_vgg.json`。

V37另有`version_snapshots/IMLS-Splatting_v37_domain_parity_20260927_code.tar.gz`，保存ball侧车alpha一致性入口、自动扩域奇偶对齐修复、测试和六场景门禁脚本。六场景门禁结果在`analysis_next_round/v37_six_scene_gate/summary.json`，只含各场景固定前20视角；不能冒充V32的200视角最终评估。服务器保留car/coffee/toaster的V36筛选源PLY，其余临时结果已定向删除。V37如需复现完整20视角门禁，仍需V32检查点、Ref-NeRF数据集和这三份筛选源；其源码归档本身不包含任何模型权重或输出。


## 未完成版本不留档

2026-09-29按用户要求：未完成六场景统一验证的局部结果不正式记录或归档。V39的模型、图像、指标、实验结果叙述与独立源码归档已撤回清理；保留工作树代码及必要的代码修复说明，不代表采纳该方法。

不要再引用V39结果路径或快照。保留的实验脚本依赖已删除的临时输入，不能直接续跑；新实验须重新生成所需输入，并完成六场景统一验证后再决定是否留档。
