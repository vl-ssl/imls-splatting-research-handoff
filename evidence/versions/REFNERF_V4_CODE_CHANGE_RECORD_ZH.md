# IMLS-Splatting 原版到统一 V4：代码修改记录

本文是 `REFNERF_V4_METHOD_RECORD_ZH.md` 的代码级补充。比较基线为当前仓库 Git 中的原始 IMLS-Splatting，记录截至统一 V4 的全部功能性修改文件、主要函数和行为；不逐行抄录 diff。

## 1. 参数和环境

### `arguments/__init__.py`

- 修改 `OptimizationParams.__init__`。
- 新增损失参数：`lambda_normal`、`lambda_depth`、`lambda_mask`、`lambda_lpips`、`lpips_interval`、`lpips_resolution`。
- 新增反射控制：`reflection_grad_scale`、`mesh_normal_weight`。
- 新增 V4 表面修复参数：`sdf_close_weight`、`sdf_smooth_weight`、`coverage_hole_fill_min_neighbors`、`coverage_hole_fill_iterations`、`mesh_component_min_faces`、`mesh_component_min_fraction`、`normal_smooth_weight`、`normal_smooth_iterations`、`normal_smooth_cosine`。默认值均不改变旧行为。
- 新增最终导出参数：`save_final_geometry_maps`、`final_render_train`、`depth_model_path`。
- 新增外观分支学习率：`residual_lr`、`position_residual_lr`、`environment_lr` 和 `freeze_base_rendernet`。

### `environment.yml`

- Python 版本更新为 3.10；旧 CUDA/PyTorch 固定项与本地子模块自动安装项改为注释，由服务器现有 CUDA/PyTorch 环境和手工编译扩展负责。
- 固定 `transformers==4.46.3`，避免 Transformers 5.x 与服务器 PyTorch 2.4 的兼容问题，并支持本地 Depth Anything V2 权重。

## 2. 3DGS 前置初始化

### `gaussian-splatting/arguments/__init__.py`

- 修改 `OptimizationParams.__init__`，将 `densify_from_iter` 从 500 提前到 100，使初始点覆盖更早细化。

### `gaussian-splatting/scene/dataset_readers.py`

- 修改 `readCamerasFromTransforms`，把 RGBA 数组转换类型由有符号 `np.byte` 修正为 `np.uint8`，避免高像素值溢出。

### `gaussian-splatting/utils/loss_utils.py`

- 新增 `compute_opacity_loss(model, lambda_opacity=0.01)`，对高斯 opacity 的平方均值做正则。

### `gaussian-splatting/train.py`

- 修改 `training`：导入并加入 `compute_opacity_loss`；原 entropy 项不再加入总损失，透明度平方正则与原 L1/DSSIM 图像损失共同反向传播。

## 3. 数据集、相机和法线 GT

### `scene/dataset_readers.py`

- 扩展 `CameraInfo`，新增 `gt_normal`。
- 修改 `readCamerasFromTransforms`：优先查找与 RGB 同目录的 `<name>_normal.png`，其次尝试把 `images` 目录映射到 `normals`；存在时读取 RGB 法线图，不存在时保留 `None`。
- 同时把 RGBA 图像数组转换改为 `np.uint8`。

### `utils/camera_utils.py`

- 新增 `resize_normal_map`：支持 NumPy/Tensor 和 HWC/CHW 输入；将 `[0,255]` 或 `[0,1]` 转成向量空间，双线性缩放后重新单位化，再编码回 `[0,1]`。
- 修改 `loadCam`：与 RGB 使用同一目标分辨率处理法线，并把结果传给 `Camera`。

### `scene/cameras.py`

- 修改 `Camera.__init__`，新增 `gt_normal` 参数和 `original_normal`。
- 统一法线数值范围与 CHW 布局，并使用 alpha mask 清除背景法线，避免边缘背景噪声参与监督和导出。

## 4. 反射网络

### `models/models.py`

- 重写 `positional_encoding` 的设备/类型处理和张量拼接，功能仍为正弦余弦 Fourier 编码。
- 扩展 `rendernet.__init__`：保留原 `spatial_mlp` 与 `directional_mlp` 的结构和参数兼容性；`btn_freq` 改为 buffer；增加 `reflection_grad_scale`。
- 新增零初始化 `residual_mlp`，输入原方向特征，输出受 `0.15*tanh` 限制的颜色残差。
- 新增零初始化 `position_residual_mlp`，输入方向特征、世界表面坐标和四频位置编码，输出受 `0.5*tanh` 限制的表面材料残差。
- 新增零初始化 `environment_mlp`，输入反射方向、六频方向编码与 `N·V`，输出受 `0.75*tanh` 限制的环境反射残差。
- `reflect` 改为静态函数。
- 修改 `spatial_mlp_forward` 与 `forward`：只处理有效 alpha 像素；归一化视线/法线/反射方向；按比例截断传向几何的反射梯度；组合基础颜色与三个残差分支；返回 `rgb_map`、`diffuse_map` 和 `specular_map`。旧检查点缺少新增分支时，零初始化保证初始输出连续。

## 5. 高斯模型与优化器

### `scene/gaussian_model.py`

- 修改 `GaussianModel.__init__`：向 `rendernet` 传入 `reflection_grad_scale`。
- 修改 PLY/检查点加载相关路径，使旧 `_op` 检查点可在新增残差分支存在时继续加载，缺失的新参数保持零初始化。
- 修改 `training_setup`：把基础渲染网络、普通残差、位置残差和环境残差拆成四个优化器组；`freeze_base_rendernet` 时基础网络学习率为零。
- 修复 `update_learning_rate` 在循环内部提前 `return` 的错误，使所有参数组都能更新学习率。
- 修改 `_prune_optimizer` 和 `_densify_optimizer`：排除四类网络参数组，只对逐点的 xyz、normal、scaling、feature 做裁剪或扩增。
- 修改 `densify_points`，保证扩增后的 feature 参数正确回写到 `_feature`。

## 6. IMLS 网格、法线、深度和材质图

### `rasterizer.py`

- 新增 `compute_grid_gradients`：对三维 SDF 做复制边界的中心差分，按 `x/y/z` 顺序输出归一化梯度。
- 扩展 `imlsplatting.__init__`，缓存连通分量掩码与签名，避免相同拓扑重复做 CPU 连通性分析。
- 新增 `repair_sdf`：可选三维形态闭运算与均值平滑。
- 新增 `fill_isolated_coverage_holes`：统计 26 邻域覆盖数，只填充满足阈值的空体素，并同步插值 SDF 和特征；阈值为零时保持原始未覆盖体素正偏移逻辑。
- 新增 `filter_small_mesh_components`：按最小面数或相对最大分量比例过滤 Marching Cubes 的小型孤立组件。
- 新增 `smooth_mesh_normals`：建立无向网格边，只在相邻法线余弦达到阈值时做迭代邻域平均和归一化。
- 修改 `marchingcube`：使用已经统一修复过的 `surface_sdfs`，保证提取表面与梯度法线来自同一 SDF。
- 修改 `rasterize`：所有路径都按 SSAA 倍率设置栅格尺寸，不再只在 `refer` 路径放大。
- 修改 `shading`：计算相机到世界表面点的射线距离；输出 alpha、depth、image_normals、image_gb_pos，并透传 diffuse/specular 等材质图；SSAA 下统一下采样全部 G-buffer。
- 修改 `__call__`，新增 `step`：确保 CUDA 输入连续；完成覆盖修复、SDF 修复、Marching Cubes、分量过滤；在网格顶点采样 SDF 梯度法线并与网格法线混合；再做可选法线平滑、反投影和着色。`mesh_normal_weight=-1` 使用旧的 5000–10000 步衰减调度，否则使用固定权重。

## 7. 损失和训练主循环

### `utils/loss_utils.py`

- 新增 `normal_loss`：把编码法线转换到 `[-1,1]` 后归一化，在可选前景 mask 内计算角度损失，并加入小权重鲁棒向量误差。
- 新增 `_pearson_correlation`。
- 新增 `pearson_depth_loss`：只在有效前景内计算预测深度和单目深度的 Pearson 相关损失，以减弱绝对尺度差异；保留转换后的深度供诊断。V4 最终配置不启用该损失。

### `train.py`

- 新增 `_map_to_uint8`：按有效区域的 1%–99% 分位稳健归一化诊断图。
- 新增 `_save_depth_comparison`：把预测射线距离转换为逆深度，与 Depth Anything 结果左右拼接；无有效数据时跳过。
- 新增 `_save_normal_comparison`：把预测法线和 GT 法线左右拼接；尺寸或数据不匹配时跳过。
- 扩展 `training`：
  - `lambda_depth>0` 时才预计算训练相机的 Depth Anything 伪深度，并以 CPU float16 暂存后释放模型显存；失败时不终止训练。
  - 可选加载一次 VGG-LPIPS，以 `lpips_interval` 间隔和 `lpips_resolution` 分辨率参与训练。
  - 在 L1/SSIM 基础上加入可选 alpha mask、法线、LPIPS 和 Pearson 深度损失，并显示各项日志。
  - 调用栅格器时传递当前 `iteration`，供法线混合调度使用。
  - 删除按固定训练阶段保存深度调试图的实际执行路径，改为训练结束统一导出。
  - 最终按 UID 排序遍历 train/test 全部视角，保存 RGB/GT；可选保存预测逆深度/Depth Anything 和预测法线/法线 GT 组合。缺图或单张失败时只跳过对应项。

## 8. 深度、评估和比较工具

### `utils/depth_interactor.py`（新增）

- 新增 `OnlineDepthEstimator`。
- `__init__` 从指定本地目录或环境变量加载 Hugging Face Depth Anything V2，目录存在时强制 `local_files_only`。
- `infer_depth` 把相机 RGB 转成模型输入，插值回原分辨率并归一化为 `[0,1]` 的相对逆深度图。
- `clear_memory` 删除模型和处理器并释放 CUDA 缓存。

### `metrics.py`（新增）

- `readImages` 按文件名对齐 `renders` 与 `gt`。
- `evaluate` 只评估 test split，逐视角计算 SSIM、PSNR、VGG-LPIPS，并写入 `results.json` 和 `per_view.json`。
- VGG-LPIPS 在整个调用中只初始化一次，不再逐图反复加载。

### `tools/build_refnerf_comparison.py`（新增）

- 通过 `FINAL_SUFFIX`、`FINAL_LABEL` 和 `COMPARISON_DIR` 选择统一最终版本。
- `result_file`、`render_dir` 自动选择输出中对应的最终迭代目录。
- `make_contact_sheet` 生成“原始 IMLS / `_op` / V4 / GT”的同视角横向拼图。
- `main` 汇总六场景 JSON/CSV 指标，并从 V4 复制代表性的法线与深度组合图；不存在时跳过。

### `tools/repair_imls_point_normals.py`（新增）

- `normalized` 批量单位化法线。
- `repair_normals` 使用体素分组的一致法线修复角度离群点。
- `repair_normals_knn` 使用带距离权重的 KNN 邻域修复；低一致性邻域不处理，以保护薄结构和相反表面。
- `main` 读取并原样保留 PLY 的全部属性，只替换 nx/ny/nz；还可用 `min_scale_ratio` 对异常小尺度做可选下限修复。

## 9. 最终执行文件

### `run_refnerf_v4_finalize.sh`（新增，最终保留）

- 定义六场景正式 V4 检查点映射。
- 六场景使用同一 `train.py` 和反射模型；只设置场景级网格/法线平滑超参数。
- 最终导出使用 SSAA=2、纯网格法线，关闭上采样/重采样，全部学习率为零并冻结基础网络。
- 对每个场景导出全部视角并运行 `metrics.py`。脚本支持通过 `PYTHON`、`DATASET_ROOT`、`OUTPUT_ROOT`、`DEPTH_MODEL_PATH` 和 `CUDA_VISIBLE_DEVICES` 覆盖环境配置。

## 10. 结果与复现位置

- AutoDL 正式结果：`/root/autodl-tmp/output/<scene>_final_v4`
- AutoDL 对比结果：`/root/autodl-tmp/comparison_refnerf_v4`
- 本地对比结果：`comparison_refnerf_v4`
- 最终表格：`outputs/019fad96-refnerf-v4/RefNeRF_V4_results.xlsx`

正式保留的中间检查点是最终六场景映射所依赖的阶段目录；V1/V2/V3 失败消融、未采用的 V4 消融、临时预览、备份代码、缓存、编译产物和旧日志均不属于统一 V4 方法。
