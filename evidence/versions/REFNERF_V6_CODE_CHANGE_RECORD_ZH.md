# IMLS-Splatting V6 代码修改记录

本文记录相对原始IMLS-Splatting、并累积V4/V5改动后的V6新增内容。逐行差异以工作区代码为准。

## `models/models.py`

- `rendernet.__init__`新增`reflection_texture`：形状`[1,3,64,128]`的可学习球面RGB贴图，全零初始化。
- 新增`reflection_gate`：输入逐点192维表面特征和`n_dot_v`，输出单通道门控；权重为零、偏置为-2。
- 新增`reflection_texture_scale=0.75`，限制新支路最大影响。
- `rendernet.forward`复用反射方向经纬度，在经度方向循环padding后调用`grid_sample`。
- 采样RGB经`tanh`限幅，与`sigmoid(reflection_gate)`相乘后加入最终颜色logit。
- 旧检查点缺少这些键时通过既有`strict=False`加载；零贴图保证初始RGB不变。

## `scene/gaussian_model.py`

- `GaussianModel.training_setup`新增`reflection_map_params`优化组。
- `reflection_texture`和`reflection_gate.*`不再与旧`environment_mlp`/V5潜变量方向分解共用学习率。
- 优化组读取`reflection_map_lr`；未显式配置时回退到环境学习率。

## `arguments/__init__.py`

- `OptimizationParams`新增`reflection_map_lr=0.001`，保持默认训练接口向后兼容。

## `run_refnerf_v6_exact.sh`

- 六场景统一执行入口，默认顺序为ball、car、coffee、helmet、teapot、toaster。
- 从各场景`*_final_v4/point_cloud/iteration_1`开始。
- 固定位置、法线、尺度、特征及所有旧颜色网络，仅训练V6反射贴图与门控。
- 统一训练3000步，`reflection_map_lr=0.01`，图像损失为L1/SSIM并每20步加入0.005权重、256分辨率LPIPS。
- 训练后用SSAA2导出全部测试/训练视角、最终法线组合和深度组合图，并运行RGB与几何诊断指标。
- 脚本保留V4每个场景已采用的网格和法线平滑参数，防止改变提取几何。
- 使用`test ! -e`保护输出，避免意外覆盖已有结果。

## 评估与产物

- AutoDL最终目录：`output/{scene}_v6_exact_candidate`。
- AutoDL日志：`run_refnerf_v6_exact_ball.log`、`run_refnerf_v6_exact_remaining.log`。
- 本地对比目录：`comparison_refnerf_v6`，含六场景RGB、法线、深度组合图及`metrics.csv`、`metrics.json`。
- 失败的错误网格实验、初版V6验证目录和可由最终候选复现的训练中间目录已经清理；V4结果未覆盖。
