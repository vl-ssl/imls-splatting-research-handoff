# IMLS-Splatting累计代码修改记录（原始版至V39实验）

更新日期：2026-09-28。本文用于复现和继续开发，覆盖相对原始IMLS-Splatting的累计修改文件、主要函数和功能；不逐行抄录。方法动机、实验结论与第三方参考见`REFNERF_V22_METHOD_RECORD_ZH.md`。历史分节保留当时实验状态；最新状态见文末。V39发现训练导出帧身份错配，涉及旧离线补点/筛点/剔除证据，详见文末勘误；V32仍是最后完整六场景200视角版本。

## 1. 主项目文件

|文件|修改的函数/类|累计内容|
|---|---|---|
|`arguments/__init__.py`|`OptimizationParams.__init__`|法线、Depth Anything、Ref-NeRF视差、mask/LPIPS；几何独立训练；轮廓IoU/召回/边界/多尺度；GT法线连续/梯度/CVaR；点锚定/切向；自动域；覆盖/SDF/网格组件/Taubin修复；各反射分支学习率和正则；最终几何图导出；V19支持度、V21辅助门控；V22 `factorized_reflection`及`reflection_factor_lr`。新增项默认关闭，脚本显式启用|
|`models/models.py`|`positional_encoding`；`rendernet.__init__ / reflect / spatial_mlp_forward / forward`|原空间/方向MLP上增加零初始化方向残差、世界位置残差、环境MLP、方向材质分解、直接RGB反射贴图和门控、高分辨率细节贴图、反射微法线；输出材质诊断；V22增加`sample_reflection_mip`、方向特征金字塔、表面粗糙度MLP、材料外积式读出和门控。新读出零初始化，开关关闭时历史路径不执行|
|`scene/gaussian_model.py`|`GaussianModel.__init__ / create_from_pcd / create_from_3dgs / create_from_ply / save_ply / construct_list_of_attributes`|IMLS点/网络加载保存、旧反射贴图插值兼容、点级`imls_confidence`字段；V22把粗糙度因子开关传入renderer，历史ckpt缺新键时保持零读出初始化|
|同上|`training_setup / update_learning_rate / _prune_optimizer / _densify_optimizer / densify_points / prune_points`|点参数与全局网络分组学习率，防止全局网络被逐点裁剪；新增各反射优化组；V22新增`reflection_factor`组，并把它加入裁剪/致密化排除集合；点增删时同步confidence，长度异常改为显式报错而非静默全1|
|`rasterizer.py`|`compute_grid_gradients`；`imlsplatting.repair_sdf / regularize_low_confidence_sdf / fill_isolated_coverage_holes / filter_small_mesh_components / smooth_mesh_normals / smooth_mesh_vertices`|最终SDF梯度、覆盖洞、组件、低支持度、窄带扩散和法线/网格Taubin消融|
|同上|`marchingcube / rasterize / shading / __call__`|IMLS→SDF→MC→组件→法线混合→像素着色；输出世界位置、欧氏深度、camera-Z及材质图；V20/V21辅助点矩合并；V22 SSAA后重新单位化世界法线，`normal_world`与alpha分离，`image_normals`不再预乘alpha；透传roughness和factorized reflection图|
|`train.py`|`prepare_output_and_logger / training`|RGB L1/SSIM/LPIPS；法线、DA深度、真实视差、mask、轮廓、多尺度、CVaR、切平面、点锚定、反射TV/微法线；几何冻结/几何only；自动网格域；全视角最终输出。V22法线损失使用GT与可靠预测覆盖交集，reflection TV包含因子纹理|
|同上|`_map_to_uint8 / _save_depth_comparison / _save_normal_comparison / _save_geometry_raw`|最终深度/法线组合；V22取消camera-Z的`abs`，独立alpha合成正常图，保存未量化`pred_normal_world / pred_alpha / pred_camera_z`及可用GT|
|`utils/loss_utils.py`|`normal_loss / _pearson_correlation / pearson_depth_loss / pearson_disparity_loss`|编码法线的鲁棒角度损失；DA逆深度相关；正camera-Z到视差的尺度对齐、Huber/梯度/相关组合|
|`utils/surface_regularization.py`（新增）|`surface_tangent_loss`|只在训练GT法线连续的不透明区域约束多尺度3D切向位移，不把镜面RGB轮廓当几何边|
|`utils/depth_interactor.py`（新增）|`OnlineDepthEstimator.__init__ / infer_depth / clear_memory`|本地Depth Anything加载、推理、缩放和显存释放；权重为0时不参与训练|
|`scene/cameras.py`|`Camera.__init__`|保存GT法线、alpha、DA相对深度、Ref-NeRF视差等辅助量|
|`scene/dataset_readers.py`|`CameraInfo / readCamerasFromTransforms / readNerfSyntheticInfo`|读取RGBA、法线、视差并保持相机对齐|
|`utils/camera_utils.py`|`resize_normal_map / resize_scalar_map / loadCam`|法线、标量图缩放与相机辅助量传递|
|`metrics.py`（新增）|`readImages / evaluate`|test逐视角SSIM/PSNR/LPIPS VGG0.1及均值JSON；LPIPS实例复用|

## 2. CUDA IMLS扩展（V22首次修改）

|文件|函数|V22内容|
|---|---|---|
|`submodules/diffsplatting/splatter.cu`|`SplatterForward`|新增每体素`int32 out_counts`临时张量，传入核心前向；Python返回接口保持不变|
|`submodules/diffsplatting/src/forward.cu`|`splatCUDA / FORWARD::splat`|每个phase不再独立拒绝单点，而是累积有效点全局计数和未归一化矩|
|同上|`weightingCUDA / FORWARD::weighting`|按全局count>1统一保留/置零；线程数向上取整并检查`thread_idx<num_cell`，修复尾体素|
|`submodules/diffsplatting/src/forward.h`|`splat / weighting`声明|增加`out_counts`参数|
|`submodules/diffsplatting/src/splatter_impl.h`|`Splatter::forward`声明|增加计数缓冲参数|
|`submodules/diffsplatting/src/splatter_impl.cu`|`Splatter::forward`|把计数缓冲传到累积和归一化两阶段|
|`submodules/diffsplatting/src/backward.cu`|未改|全局count不合格时前向将`out_sums`置0，原`compute_grad_flag`不生成反向；合格时全体点反向与新前向一致。已用feature有限差分验证；位置/法线/尺度完整gradcheck仍待补充|

## 3. 前置3DGS子项目

|文件|函数|累计内容/当前审查结论|
|---|---|---|
|`gaussian-splatting/arguments/__init__.py`|`OptimizationParams.__init__`|`densify_from_iter`由500改100|
|`gaussian-splatting/scene/dataset_readers.py`|`readCamerasFromTransforms`|RGBA数组由有符号`np.byte`改为正确`np.uint8`|
|`gaussian-splatting/utils/loss_utils.py`|`compute_opacity_loss`|新增`lambda*mean(opacity²)`；审查确认它压低opacity，不等价于前景累积alpha监督|
|`gaussian-splatting/train.py`|`training`|关闭原entropy调用并固定加入0.01的opacity平方项。V22-G0计划新增显式legacy/off/accumulated-alpha三路对照，当前尚未修改这部分语义|
|`environment.yml`|依赖配置|Python3.10、transformers4.46.3；旧torch/CUDA/扩展条目被注释。它不是完整环境锁，服务器实测环境需另存|

## 4. 评估、诊断和数据工具

|文件|函数/用途|
|---|---|
|`tools/build_refnerf_comparison.py`|`result_file / render_dir / make_contact_sheet / main`：原始、op、候选、GT以及法线/深度对比和指标汇总|
|`tools/evaluate_geometry_maps.py`|`evaluate`：V22优先读取`geometry_raw/*.npz`，用独立alpha、原始有符号法线和共同腐蚀前景计算IoU、缺失/伪前景、法线均值/P95；无raw时保留标记为LEGACY的PNG估计|
|`tools/refnerf_next/diagnostics.py`（新增）|`imls_query / erode_mask / geometry_metrics / depth_diagnostics`：CPU参考IMLS、旧phase规则模拟、独立alpha法线指标和正深度检查|
|`tools/refnerf_next/run_numerical_audit.py`（新增）|源码/二进制SHA、尾体素、33点排列、feature有限差分；拒绝覆盖报告，禁止在受保护项目/环境运行|
|`tools/refnerf_next/test_diagnostics.py`（新增）|14项NumPy单元测试；本地和服务器均通过|
|`tools/refnerf_next/run_server_audit.sh`（新增）|GPU有活动进程或查询失败即退出；只做审计，不安装/训练/终止进程|
|`tools/repair_imls_point_normals.py`|点法线共识与KNN修复消融|
|`tools/diagnose_refnerf_disparity.py`|ray distance和camera-Z视差回投诊断|
|`tools/bilateral_smooth_imls_points.py`|V9法线感知点位平滑消融|
|`tools/augment_visual_hull_points.py`|V16–V21训练mask视觉外壳、缺失证据、GT训练法线、confidence PLY及检查点复制|
|`tools/filter_visibility_support_points.py`（V24/V25新增）|`parse_args / project_grid / sample_map / calibrate_flip / copy_rows_to_dtype / main`：从V16候选中提取只被训练视角支持的缺失轮廓点；联合训练alpha缺失、原始camera-Z近表面、GT法线一致性、原始点距离及前置冲突计数。V25增加按`missing_views`与`near_views`自适应的前置冲突预算；输出筛选PLY、JSON和NPZ证据。明确不读取测试视角和RGB|
|`tools/prune_opaque_surface_conflicts.py`（V26/V27新增、已否决）|`parse_args / main`并复用V24/V25投影工具：审计原始IMLS点在所有训练视角中是否曾成为可信第一表面；对“无近表面支持且反复位于可信表面之前或GT背景内”的点生成剔除候选，保存逐点证据。V27增加`require_both_conflicts`，要求两类矛盾同时成立。不读取测试视角和RGB；激进与保守规则均使car/coffee退化|
|`utils/surface_regularization.py`（V29新增）|`_sample_volume / imls_bidirectional_consistency_loss`：在显式点、法线两侧和两条切向上三线性采样当前IMLS场；计算零水平集、正负符号和切向导数一致性。有效门仅由分离梯度的IMLS核支撑决定，不读取RGB或测试视角|
|`utils/surface_regularization.py`（V30新增、已否决）|`apply_bounded_implicit_residual / implicit_residual_regularization`：把低分辨率`tanh`残差网格三线性上采样到IMLS场，按提取体素限制幅值，并返回幅值/TV正则。默认禁用，保留消融|
|`scene/gaussian_model.py`（V30扩展）|可选`implicit_sdf_grid`参数进入模型检查点和独立优化器参数组；剪枝/增点明确排除该非逐点参数，旧检查点缺少它时以零初始化兼容加载|
|`rasterizer.py`（V30扩展）|在IMLS修复后、Marching Cubes前可选叠加独立连续隐式残差；零分辨率或零幅值保持旧路径|
|`tools/test_surface_tangent.py`|切平面约束CPU测试|
|`tools/build_code_snapshots.ps1 / build_current_code_snapshot.ps1`|历史V4/V6和当前树归档辅助；不能仅凭标签还原任意版本|

## 5. 主要执行脚本与版本状态

- `run_refnerf_v4_finalize.sh`、`run_refnerf_v6_exact.sh`：V4/V6入口；有独立源码归档。
- `run_refnerf_v5_*.sh`：域、法线共识、SDF、切平面和外观消融。
- `run_refnerf_v7_*.sh`：几何—外观两阶段、视差/法线梯度消融；`appearance_full`也作为V10后续外观入口。
- `run_refnerf_v8_zdisparity.sh`、`run_refnerf_v9_bilateral_probe.sh`：否决消融。
- `run_refnerf_v10_mesh_taubin_probe.sh / refine.sh`：弱网格Taubin探针与300步几何精修。
- `run_refnerf_v11...`至`run_refnerf_v15...`：CVaR、SDF扩散、细节贴图、微法线、轮廓释放消融。
- `run_refnerf_v16...`至`run_refnerf_v21...`：视觉外壳、几何校正、外观、支持度、辅助点权重/门控。
- `run_refnerf_v22_kernel_probe.sh`（新增）：固定V10检查点和外观，只比较新CUDA核；car/coffee已完成。
- V22修复后几何用`SCENES='car coffee' TAG='v22_kernel_rawnormal_refine' run_refnerf_v10_mesh_taubin_refine.sh`完成统一300步精修。
- `run_refnerf_v22_appearance.sh`（新增，V24更新）：`METHOD=baseline|factorized`在同一几何、同一2500步预算下完成旧反射头与粗糙度方向因子分支消融；V24增加`RUN_TAG`和含`{scene}`占位符的`SOURCE_PLY_TEMPLATE`，使V24/V25可复用完全相同的外观预算。两场景均输出全部train/test视角的RGB、未预乘alpha法线、camera-Z/深度对照和独立alpha原始数组。脚本内含受保护项目/环境拒绝检查。
- `run_refnerf_v23_visibility_support_probe.sh`：将V16全部视觉外壳点放到V22数值修复路径做统一探针。car轮廓改善但coffee显著退化，已否决；完整大输出、旧日志与重复诊断包已定向删除，保留脚本用于消融重建。
- `run_refnerf_v24_first_surface_support_probe.sh`：固定前置冲突上限12的训练视角第一表面筛选及car/coffee探针；car接纳368点、coffee接纳0点。
- `run_refnerf_v25_adaptive_surface_support_probe.sh`：在V24证据上使用`front_conflicts <= max(12, 3*missing_views + near_views)`的统一自适应预算；car接纳656点、coffee接纳0点。直接几何探针已完成，外观训练状态见方法记录。
- `run_refnerf_v26_opaque_conflict_probe.sh`：从V25源点出发测试“前置或背景冲突”剔除，car/coffee均退化，已否决。
- `run_refnerf_v29_bidirectional_surface.sh`（已否决）：从六场景V6候选出发，使用同一套点—IMLS双向一致性、训练alpha轮廓和GT法线做几何验证，再以零学习率导出测试RGB/法线/原始几何。固定符号300步版使ball崩溃；方向无关100步版在ball混合改善、car三项退化，因此停止。脚本保留用于消融，不作为最终执行方案。
- `run_refnerf_v30_implicit_surface.sh / run_refnerf_v30_appearance_recovery.sh`（已否决）：前者冻结点与外观，只训练`48³`、两体素幅值的连续隐式残差；后者冻结V30几何重新优化原反射头以排除外观失配。ball几何与外观恢复门禁均低于V6，脚本仅保留作消融。
- `run_refnerf_v31_deferred_normal_propagation.sh`（已否决）：冻结V6几何及既有外观，仅训练切空间反射微法线；用训练GT连续域对微法线偏移做多尺度屏幕空间传播。car改进小于记录门槛且LPIPS略退，脚本只保留作消融。
- `run_refnerf_v27_conservative_opaque_prune_probe.sh`：改为强前置与强背景冲突同时成立，删除点大幅减少，但两场景仍退化，已否决。
- `run_refnerf_v28_reflective_surface_continuity_probe.sh`：不删点，在高度法线一致邻接上加强弱Taubin网格连续化并保持边界不动；car/coffee均未改善，已否决。

## 6. 版本留档与验证

服务器已保存：

- `IMLS-Splatting_v21_audited_20260922_code.tar.gz`及SHA-256；旧CUDA二进制`diffsplatting_v21_audited_20260922.so`及SHA-256。
- `IMLS-Splatting_v22_kernel_corrected_20260922_code.tar.gz`及SHA-256；修复二进制`diffsplatting_v22_kernel_corrected_20260922.so`及SHA-256。
- `IMLS-Splatting_v25_adaptive_surface_20260922_code.tar.gz`，SHA-256为`41c5be74329f4697c3792a1d544ab238ed7cbe70f8ed7f299652088759ef4b19`；归档包含V25可复现路径并显式排除建立于其后的未验证V26脚本/工具。

V22 CUDA审计修复前后的JSON/日志保存在`analysis_next_round/server_audits/`。本轮不覆盖V4/V6/V18归档，不接触`/root/autodl-tmp/gaussian-shader`或`gaussian_shader4090`。

已验证：Python语法、14项CPU测试、renderer新分支启用前RGB完全一致、服务器CUDA参数梯度、CUDA尾部/顺序/feature有限差分、V22 car/coffee重训几何、旧头对照和粗糙度因子分支；V23全视觉外壳否决；V24固定冲突筛选与V25自适应冲突筛选的car/coffee几何及同预算外观；V26/V27原点冲突剔除和V28表面连续化均否决。V25把car提高到0.912367/21.02768/0.077555并降低缺失前景，coffee保持V22，但目视上仍未完全解决car缺边、coffee多层托盘和倒影轮廓问题。待验证：全部CUDA梯度类别、显式点—隐式闭合表面双向约束和六场景统一结果。

中间暂存目录`.v22_upload`在逐文件放入正确目录并完成语法测试后已定向删除；没有删除数据集、V4/V6完整阶段结果或V25最终检查点。2026-09-22完成第二次收敛清理：删除V7–V24、V26–V28共46个服务器训练/探针目录、43份旧日志和12项旧证据/诊断内容；随后又删除12个被最终V4/V6结果取代的V4/V5调参目录和9份V5过程日志。服务器结果目录现仅含原始/`_op`、六场景V4最终、六场景V6最终，以及V25的`car/coffee`筛选source与最终factorized candidate；V25另保留两份日志、最终JSON和视觉样本。V18/V22/V25源码归档、全部执行脚本及本文记录继续保留；被清理的训练文件本身不可直接恢复，但可据此重建。受保护的`/root/autodl-tmp/gaussian-shader`及其环境从未访问或修改。

V29新增参数位于`arguments/__init__.py`：`lambda_imls_surface`、`lambda_imls_sign`、`lambda_imls_tangent`、`imls_consistency_interval`、`imls_consistency_samples`、`imls_consistency_offset_voxels`和`imls_consistency_support_floor`，默认值均保持旧版本路径不变；另增加`skip_final_render`和`save_final_depth_maps`，用于把临时阶段检查点保存、法线/原始几何导出和Depth Anything全视角诊断解耦，避免重复生成大文件。`train.py::training`按设定间隔调用双向一致性损失并写入TensorBoard。`tools/refnerf_next/test_surface_regularization.py`覆盖平面零损失、正反两种定向、同号拒绝和无支撑禁用；本地与服务器共18项测试通过，服务器CUDA扰动场反向梯度有限且非零。固定符号版和方向无关版均被真实场景否决，5个V29完整/部分输出目录、两次中断输出及全部V29日志均已删除；参数与工具以默认关闭状态保留作消融，不建立V29源码归档。

V30新增参数：`implicit_sdf_grid_resolution`、`implicit_sdf_amplitude_voxels`、`implicit_sdf_lr`、`lambda_implicit_sdf_magnitude`和`lambda_implicit_sdf_tv`；默认均为零/禁用。`tools/refnerf_next/test_implicit_sdf_residual.py`验证零初始化严格保持旧表面、残差体素幅值上界、有限梯度和TV检测；累计21项测试通过，服务器两步GPU探针确认`implicit_sdf_grid`写入检查点并产生有限非零更新。ball几何和外观恢复均否决后，4个正式输出目录、集成探针和全部日志均已清理；不建立V30源码归档。

V31在`arguments/__init__.py`新增`lambda_reflection_normal_tv`、`reflection_normal_tv_cosine`和`reflection_normal_tv_erode`，默认关闭。`train.py::training`在GT法线连续、腐蚀后的不透明内部区域对`reflection_normal_offset_map`做1/4像素尺度Smooth-L1传播，并记录TensorBoard；真实`image_normals`和最终法线对照不使用该偏移。car门禁无实质改善后，2个V31输出目录和日志已清理，不建立V31源码归档。

## V32增量与历史解释勘误（2026-09-26）

本文前面原始IMLS到V31的累计文件/函数记录仍适用。撤回“固定符号版使ball崩溃”的确定性解释：ball侧车alpha未读取，旧IoU不可信；历史RGB指标保留，但缺少同代码零更新对照。V30是稠密有界残差，不是独立闭合SDF或低秩场。详细方法解释见累计方法记录第15节。

|文件|函数/内容|
|---|---|
|`utils/refnerf_alpha.py`（新增）|`read_refnerf_image`：原生alpha优先；否则读取同名`_alpha.png`，验证尺寸和预合成白底（8位254–255容差）。`composite_reference`：统一CHW张量合成，避免RGB白底图重复乘alpha；白底GT保持不变。|
|`scene/dataset_readers.py`|`CameraInfo`新增可选`gt_alpha / image_background`，兼容其他数据读取器；`readCamerasFromTransforms`使用新加载器并传递两个字段。|
|`utils/camera_utils.py`|`loadCam`将侧车mask按同一分辨率缩放，优先于RGB转RGBA得到的全1alpha，并把原合成背景传给Camera。|
|`scene/cameras.py`|`Camera.__init__`新增默认None的`image_background`元数据。原法线和深度处理保留。|
|`train.py`|`training`训练GT合成、最终渲染GT合成都改用`composite_reference`；没有替换RGB GT，没有改变模型架构。|
|`tools/refnerf_next/test_refnerf_alpha.py`（新增）|覆盖侧车加载/白底RGB逐值不变、黑底转换、原生RGBA优先、尺寸异常、未知背景拒绝、原生straight alpha公式不变。累计26项CPU测试在本地/服务器通过。|
|`run_refnerf_v32_alpha_audit.sh`（新增）|同V6源检查点依次运行当前代码零更新对照、冻结外观100次几何更新、最终测试评估。STEPS=101适配现有保存/优化器顺序（最后一次不更新）。保留原V6提取参数，不允许覆盖已存在输出；包含受保护项目/环境拒绝检查。默认ball，SCENES可选择六场景。|
|两份`REFNERF_V22_*_RECORD_ZH.md`|累计记录追加V32并标明历史结论勘误；文件名延续V22以保持已有链接。|

此修复不操作3DGS前置结果、不修改数据集图像、不涉及`gaussian-shader`或`gaussian_shader4090`。历史V4/V6/V25归档不覆盖。失败的首个V32启动仅产生cfg_args（由于两处254量化像素触发严格白底检查），已定向删除并重启；不保留无效训练结果。

后续执行/保留工具：`run_refnerf_v32_remaining.sh`依次验证剩余五场景，沿用完全相同的V32损失/训练预算，不拼接版本。`tools/compact_v32_audit.py::main`只允许六个固定场景名，检查路径非重定向、两组200视角评估全部完成及最终候选检查点存在；复制完整数值和四个固定视角到`analysis_next_round/v32/<scene>`，然后删除该场景V32零更新对照及重复训练目录，保留最终候选并写清理清单。任何检查失败则停止，绝不跨出当前项目删除数据。`VERSION_REPRODUCTION_ZH.md`和快照README同步澄清：旧脚本配合新CUDA/读取器并不能保证严格复现旧版，需使用对应源代码归档。

本轮ball完成200视角控制及候选评估，候选PSNR30.6441、SSIM0.965123、legacy LPIPS0.124887、法线MAE0.538961°，作为阶段结果保留，非六场景最终模型。Python语法检查与26项CPU测试通过；代码差异检查仍报告历史修改中的空白行尾空格，本轮未对整个旧工作树进行无关格式化。

## V32六场景收尾及V33候选增量（2026-09-27）

V32六场景均完成200个测试视角的同代码控制/候选评估，表格与视觉结论见方法记录第16节。保留六个候选、所有指标与固定代表视角；控制和重复训练目录已由`tools/compact_v32_audit.py::main`逐场景验证后删除，共6.75 GB。V32归档的SHA-256为`d404dec74d715ea3e73d45e84c2bb4ac3e9c624ee0a6122318144a54c6da04c5`。

|文件|函数与改动|
|---|---|
|`utils/implicit_pixel_normals.py`（新增）|`blend_implicit_pixel_normals`：在像素世界坐标查询[1,3,D,H,W] IMLS梯度体，使用与提取相同的归一化映射、三线性采样和法线定向；对场外、低alpha、零梯度或矛盾方向退回网格法线。权重0直接返回原张量，保持旧路径。|
|`rasterizer.py`|`imlsplatting.__call__`在开关启用时暂存当前提取用的梯度体；`shading`在反射MLP之前查询并混合逐像素几何法线，同时改变输出法线与反射方向。原网格法线和V32默认路径保留。|
|`arguments/__init__.py`|`OptimizationParams`新增`implicit_pixel_normal_weight`，默认0；`final_render_test_limit`默认0，仅短程探针使用20，不改变正式全视角导出。|
|`train.py`|`training`最终导出按可选测试视角上限截取排序后的列表；默认0仍导出全部视角。|
|`test_v33_pixel_normals.py`（新增）|验证零开关张量完全保留、平面场法线方向、背景与方向不可靠时的回退。|
|`run_refnerf_v33_pixel_normal_probe.sh`（新增）|同一V32候选检查点、SSAA1、20测试视角，car/coffee分别对照权重0和0.7；保留指标和原始几何以决定是否值得做完整六场景训练。|

本轮V33仍为候选实验，不纳入已验证的V32归档。代码未复制3DGS-DR或GS-ROR²；思想联系与差异见方法记录第16节。实验未证明改善前，保留V32作为当前统一结果。

V33 car/coffee固定20视角比较仅有极小RGB收益，coffee原始法线误差上升，故否决；四个实验目录在确认20视角指标完整后删除约681MB。`compact_v33_pixel_normal_probe.py::main`检查固定场景与变体、报告计数/指标、复制数值及视角00000/00010图像后，定向删除其四个探针目录；它只在本项目根目录下执行。对应证据在`analysis_next_round/v33_pixel_normal`，V33仍默认关闭供消融，V32归档不覆盖。

`run_refnerf_v34_surface_continuity_probe.sh`在V32最终候选上执行统一的训练法线连续/表面切平面损失，car/coffee先做100次训练及20固定测试视角门禁。该脚本调用`train.py::training`和已有`utils/surface_regularization.py::surface_tangent_loss`，没有改动这些损失实现。目标、权重和区别见方法记录第17节；当前尚无通过门禁的结果。

`run_refnerf_v35_reflection_probe.sh`复用`models/models.py::RenderNet.forward/sample_reflection_mip`及已有`factorized_reflection`开关，冻结V32几何、成对训练原反射头与因子化残差头500次，使用`train.py::training`的RGB/LPIPS损失，最终按相同SSAA1与20测试视角输出指标与几何图。此时该脚本只是可执行对照，未宣称优于V32；两份相邻实验保留各自结果，不按场景择优拼接。与Ref-GS思路联系在方法记录第18节明确标注。

V34最终两场景的原始法线MAE均改善约0.18–0.22°，但SSIM、PSNR、legacy LPIPS全部退步，因此否决。`compact_v34_surface_probe.py::main`检查car/coffee固定20视角评估已完整，再复制指标与视角00000/00010代表图并定向删除四个可重算目录约487MB。`evaluate_lpips_standard.py::evaluate/main`逐视角读取RGB与GT，将[0,1]变换到[-1,1]后按官方标准计算LPIPS-VGG并另存JSON，不覆盖历史结果；待GPU空闲后批量重算。

V35成对20视角完成，细节与否决理由见方法记录第18节。`run_refnerf_v35_reflection_probe.sh`只调用已有`models/models.py`因子化反射开关，不额外修改RenderNet或几何参数；同源检查点训练500次并以同一设置导出。`compact_v35_reflection_probe.py::main`（新增）先验证car/coffee两种变体各20视角的RGB和几何指标齐全，再保存`results.json`、`per_view.json`、`geometry_metrics.json`、新口径`standard_lpips_vgg.json`及视角00000/00010的render/GT/法线比较；随后仅删除八个具名V35训练/探针目录，约1.10GB。`evaluate_lpips_standard.py::evaluate/main`已在四组V35真实输出上运行，计算标准LPIPS-VGG；当前六场景V32完整输出也按同一代码复核。V35的因子化方法与Ref-GS方向/材质分解思想相关；本轮复用此前独立实现，没有复制其源码。V32源码归档不覆盖，也不为效果极小的V35建立正式版本归档。

`evaluate_lpips_standard.py::evaluate/main`进一步对V6/V32各六场景、每场景200视角完成标准口径重算，在原`renders_iter_1/test/standard_lpips_vgg.json`独立保存逐视角和均值，不覆写历史`results.json`。V32标准LPIPS等权平均为0.109774，V6为0.118169；coffee由0.144467退至0.145958，勘误写入方法记录第19节。

`run_refnerf_v36_surface_support_probe.sh`（新增，候选）：用当前V32检查点和`train.py::training`最终导出分支生成全部训练视角未压缩几何缓冲与法线/GT比较；调用已有`tools/augment_visual_hull_points.py::main/load_frames`产生训练alpha缺边候选，再调用`tools/filter_visibility_support_points.py::main`以训练camera-Z、GT法线和前置冲突筛选；最后以相同V32冻结外观参数导出固定20测试视角并计算RGB/几何指标。其筛选规则为统一参数，不按测试指标挑场景。当前只运行car的第一轮门禁；未验证coffee/六场景，不建立V36源码归档或最终结果声明。ball的RGB+独立alpha侧车读取需要与V32数据审计保持一致，未修正旧外壳工具前不把ball加入V36运行。

car筛选产生935个支撑点，但固定提取盒外占约98.4%，固定域探针几乎完全不变。`run_refnerf_v36_domain_pair.sh`（新增）因此在相同`train.py::training`、`--auto_mesh_domain`、SSAA1、20测试视角下成对导出V32原点集和V36筛选加点集；只把两者之间差异归因于支撑点，自动扩域本身另与固定域基线对比。`train.py`既有自动域实现先按点集最大范围扩大提取网格并保持原体素间距，本轮未修改该函数。V36仍是探针而非值得保存的最终版本。

car最初自动域原点集/加点集比较显示PSNR差`+0.5546dB`、IoU差`+0.01339`、法线MAE差`-1.485°`，但两个输入各自扩大域至不同尺寸，不能把差值全归因于935点。`run_refnerf_v36_domain_pair.sh`新增`MATCHED_MESHSCALE/MATCHED_GRID`参数；提供两者时关闭自动域，两组强制使用同一固定物理尺寸和体素间距，输出`matched_domain`新标签以免覆盖旧探针。`run_refnerf_v36_refine_pair.sh`（新增）准备将同一V32几何训练损失/学习率和100次更新分别应用于原点集与加点集，再按同SSAA1/20视角评估；仅在严格域配对表明真实收益后运行。只追加执行脚本，不修改原有损失实现、反射网络或GT法线保留机制。未完成时不得将该脚本结果写成正式版本。

car严格同域`grid282/meshscale2.314118`复核确认加点本身带来PSNR`+0.5551dB`、IoU`+0.0134`及法线MAE`-1.485°`，方法记录第19节给出完整数值。`run_refnerf_v36_domain_pair.sh`另加`VARIANTS`，允许远端中断时只重跑缺失变体，避免覆盖完整对照。一次远端脚本更新恰逢其执行，coffee后半任务出现shell读取错误；检查后确认其RGB/几何结果已保存，失败只发生在脚本尾部。没有删除已完成输出，后续以文件实际完整性而非进程退出码判断；实验文件上传禁止覆盖正在执行的同名脚本。

`run_refnerf_v36_refine_pair.sh`随后增加与零更新配对完全一致的`MATCHED_GRID/MATCHED_MESHSCALE`选项，训练、导出与两组点集固定同一个网格；输出标签含`matched_domain`，不会覆盖初次自动域探针。这样100步后的差异不再混入各点集单独改变提取盒的影响。

V36 car精修配对完成：加点仍明显优于同训练预算原点集，但比不精修的加点集在SSIM、legacy和标准LPIPS上退化；coffee未重复这一无益训练。`compact_v36_support_probe.py::main/checked_directory`（新增）仅定位car/coffee这轮具名输出，先验证固定20视角RGB/几何指标、保留全部JSON与视角00000/00010的render/GT/法线图和筛选证据，再定向删除训练视角几何缓存、视觉外壳候选、重复的固定/自动/匹配域评估与car精修训练目录。保留两场景最终筛选源PLY/检查点供下一轮研究，不覆盖V32六场景候选、不触及数据集或受保护项目。V36尚无六场景最终验证，也没有完成不透明镜面倒影轮廓的改进证明；仅建立**候选源码快照**供消融和回滚，不作为正式最终模型。

上述V36整理已实际执行，18个具名服务器目录合计清理`3,226,418,218`字节；`analysis_next_round/v36_surface_support`完整证据已同步本地，重复传输压缩包随后删除。本轮新增/修改的执行文件包括`run_refnerf_v36_surface_support_probe.sh`、`run_refnerf_v36_domain_pair.sh`、`run_refnerf_v36_refine_pair.sh`和`compact_v36_support_probe.py`；V33默认关闭的像素隐式法线代码、V32侧车掩码修复及更早所有累计修改仍按上文保留。未触碰`gaussian-shader`项目或其环境。

`tools/build_current_code_snapshot.ps1`以标签`v36_first_surface_candidate_20260927`保存当前兼容代码树，输出`version_snapshots/IMLS-Splatting_v36_first_surface_candidate_20260927_code.tar.gz`并更新`SHA256SUMS.txt`；源码归档排除数据、权重与输出，真正重建探针仍需要V32检查点和保留的V36筛选源PLY。

## V37统一六场景门禁与数值修复（2026-09-27）

V37不更换反射MLP或删除原法线/深度路径。它完成V36第一表面支撑在六场景的同规则筛选，发现自动域奇偶变化引起的半体素采样错位，再用同点集toaster对照与六场景固定20视角门禁验证。完整数值与不足见方法记录第20节。

|文件|函数/修改内容|
|---|---|
|`augment_visual_hull_points_refnerf.py`（新增）|`load_frames`复用`utils.refnerf_alpha.read_refnerf_image`，对RGB+`*_alpha.png`加载真实掩码，对原生RGBA保留原alpha；仅把已存在`tools/augment_visual_hull_points.py`的`load_frames`入口替换为审计版后调用其`main`，视觉外壳/法线/筛选算法不变。服务器ball首帧前景占比0.502934，car仍读原生alpha。|
|`run_refnerf_v36_surface_support_probe.sh`|调用上述审计版外壳入口，使ball与其他场景遵守同一前景定义；训练视角几何缓冲与筛选流程、统一阈值不变。|
|`utils/mesh_domain.py`（新增）|`aligned_mesh_domain`根据旧分辨率/物理尺度和所需尺度扩展立方域，保持旧体素间距且新旧网格分辨率同奇偶；旧格点在新网格中保持整数索引偏移，不发生半体素重采样。拒绝无效网格。|
|`train.py::training`|在现有`auto_mesh_domain`分支中调用`aligned_mesh_domain`代替简单`ceil+1`，仍以点集半径和原有margin确定所需尺度；只有显式开启自动域时行为改变，V32固定域默认路径不变。保留原法线损失、深度开关、最终法线/深度导出。|
|`test_mesh_domain_parity.py`（新增）|4项CPU测试覆盖不扩域时精确保留、扩域后分辨率/体素/整数偏移、奇数原网格与无效输入。服务器再与V33两项像素法线测试合计6项通过。|
|`run_refnerf_v36_domain_pair.sh`|增加可选`DOMAIN_TAG`并保留`VARIANTS`，可独立命名/补跑奇偶网格配对，不覆盖V36已有结果；同一脚本手工固定281/282网格量化数值问题。|
|`compact_v36_noop_scene.py`（新增）|对helmet/teapot等筛出0点的场景先检查筛选证据与20视角RGB/几何指标，再保存JSON和两张代表图，删除四个具名重复输出目录；非0点则拒绝删除。|
|`evaluate_lpips_standard.py`|实现未变；本轮继续对六场景20视角及toaster奇偶配对写出标准[-1,1] LPIPS-VGG，旧结果JSON保持不变。|
|`compact_v37_six_gate.py`（新增）|`main/read_evaluation/copy_evaluation/checked_output`固定检查六场景门禁每组20视角RGB、法线几何与标准LPIPS，核对各场景筛选点数，写`summary.json`和代表图；验证后只删15个具名ball/helmet/teapot/toaster临时目录，保留car/coffee/toaster筛选源与两份累计记录。|

上述入口和参数均记录为同一方法，不按场景拼训练模型。六场景门禁只有小幅PSNR/标准LPIPS增益，SSIM不升，因此V32仍是最后一个200测试视角完整版本。V37格点对齐是值得保留的数值修复；V36筛选点并非完成的六场景最终方法。相关思路标注见方法记录第20节，不含第三方代码复制。后续仍需仅保留压缩后的证据和选中的支撑源点，清除本轮生成的重复20视角目录。

清理和同步已执行：`compact_v37_six_gate.py`定向删除15个重复目录共`3,936,660,194`字节；`compact_v36_noop_scene.py`提前删除helmet/teapot零新增点原始阶段共约`1,245,001,302`字节。六场景门禁证据同步到本地`analysis_next_round/v37_six_scene_gate`，传输压缩包随后删除。`tools/build_current_code_snapshot.ps1`另以标签`v37_domain_parity_20260927`封存数值修复后的当前源码和全部执行脚本；归档不含数据、点云、训练结果，校验值见`version_snapshots/SHA256SUMS.txt`。不因封存源码而把V37支撑候选说成六场景达标模型。

## V38–V39：图像驱动诊断、可见轮廓约束与原始支撑恢复（2026-09-28）

本节延续上文相对于原始IMLS的全部累计差异；原法线监督、可选深度与反射网络保留。V38/V39均不引入测试图像/测试法线到训练。方法、引用和实验结论见方法记录第21–22节。

|文件|函数/修改内容|
|---|---|
|`analyze_surface_error_regions.py`（新增）|`audit`使用原始法线/alpha/camera-Z与RGB/GT划分缺失、多余、边界和三档内部法线误差；`visualize/panel`生成六格诊断；`main`等间隔取20视角、保存所有逐视角分区统计，按RGB误差选两张代表图。`--iteration/--suffix`支持不同受控实验。|
|`utils/visible_contour.py`（新增）|`visible_contour_loss`按光栅化可见三角形ID和轮廓像素选网格顶点，投影到同一NDC约定，与训练mask轮廓双向最近距离；`symmetric_contour_distance`以Smooth-L1聚合。可见性/采样选择不反传，投影坐标保留梯度。|
|`arguments/__init__.py`|`OptimizationParams`增加`lambda_visible_contour=0`与`visible_contour_samples=2048`；零权重保持所有历史路径。|
|`train.py::training`|仅新权重>0且存在训练alpha时计算可见轮廓损失，加到既有几何优化中；打印样本数和损失。没有删除或改变法线/深度项。|
|`test_visible_contour.py`（新增）|四项CPU测试：分离轮廓有正确梯度、重合轮廓零损失、未见三角形顶点不获梯度、空光栅安全返回。与V37/V33合计10项通过。|
|`run_refnerf_v38_visible_contour.sh`（新增）|相同V32检查点及自动域、200次几何更新，仅新损失权重0/0.5不同；直接在训练域导出20视角，规避重载导致提取域变化。计算RGB、原始法线/轮廓、标准LPIPS。|
|`summarize_v38_contour.py`（新增）|`main`检查配对20视角指标完整；`compare`保存逐视角MSE变化、固定视角和最好/最差变化视角的RGB及法线比较，绿/红分别表示平方误差改善/恶化。|
|`restore_original_surface_support.py`（新增）|`main`读取原始op2与当前V32点集，筛出缺失支撑并以训练投影/GT法线/第一表面证据验证；新点使用原始3DGS位置、可靠GT法线投票、8近邻插值`feature_*`，尺度沿用最近原点；复制外观检查点并保存筛选证据、范围变化与同域网格参数。|
|`run_refnerf_v39_source_restore.sh`（新增）|先从V32导出全部训练几何缓冲，执行源点恢复，然后把原点集/恢复点集放到相同提取网格零更新导出20测试视角；全部场景遵守同一规则。|

2026-09-29按用户要求：未完成六场景统一验证的局部结果不正式记录或归档。V39的模型、图像、指标、实验结果叙述与独立源码归档已撤回清理；保留工作树代码及必要的代码修复说明，不代表采纳该方法。

### V39训练导出帧身份修复与历史勘误

发现`Scene.__init__`先shuffle训练相机，而`utils/camera_utils.py::loadCam`将运行id作为uid，原始索引在colmap_id。保留帧身份修复及历史勘误，不保留V39局部实验数据。

|新增/修改文件|函数和内容|
|---|---|
|`utils/export_views.py`|`ordered_export_cameras`按source_id、名称和运行uid稳定排序；`write_view_manifest`写编号/原始帧名/source_id/runtime_uid；`validate_refnerf_manifest`逐帧验证与transforms_train一致，缺失或错配拒绝离线几何监督。|
|`train.py`|`training`使用原始帧顺序导出并写`views.json`；`_save_geometry_raw`增加image_name/source_id。原训练相机采样顺序不变；测试CameraInfo原本未shuffle，测试顺序保持原始数据顺序。|
|`restore_original_surface_support.py`|筛选之前强制调用manifest检查，不能以旧无身份缓冲作几何证据。|
|`augment_visual_hull_points_refnerf.py`|新增`parse_args`包装器，在读取缺失mask前验证帧身份；保留ball独立alpha审计。|
|`filter_visibility_support_points_aligned.py`（新增）|包装旧工具的`parse_args`，检查manifest后调用已有筛点算法；原工具保留用于历史追溯。|
|`run_refnerf_v36_surface_support_probe.sh`|筛点调用改为有帧身份检查的包装器，未来重跑必须重导出训练缓冲。|
|`test_export_views.py`（新增）|验证乱序运行uid恢复原始顺序、缺失manifest拒绝、错误源帧拒绝；本地3项通过，服务器与V38/V37/V33合计13项通过。|
|`retire_v39_unaligned_probe.py`（新增）|仅在确定的服务器根目录执行；确认旧输出没有manifest后，保存错配审计和0接纳点元数据，定向删除三个本轮无效car输出目录。|
|`summarize_v38_contour.py`|增加version/candidate参数供V39同域对照复用，并保存代表视角的原分辨率RGB/GT/法线组合以便清理后继续检查。|

工具目录中的历史筛点/剔除实现本轮未修改；应使用新的有身份检查的入口。之前直接按编号配对训练文件的旧脚本只能作为历史版本复现，不能在新实验中绕过manifest验证。

`compact_v38_contour_probe.py::main`与`retire_v39_unaligned_probe.py`是历史定向清理入口；V39实验输出及其审计文件已按用户要求清除。

后续配对入口已准备：`run_refnerf_v39_geometry_pair.sh`从同域原点/恢复点组各做500次仅法线/轮廓/切平面几何更新；`run_refnerf_v39_appearance_pair.sh`冻结位置、尺度和法线，各做1000次反射外观适配，可通过`SOURCE_VERSION/SOURCE_ITERATION`明确选择统一几何阶段。新增`OptimizationParams.lpips_standard_input=False`，`train.py::training`仅在显式开启时把LPIPS输入由[0,1]映射到[-1,1]；V39外观配对两组均开启该项，历史默认行为保留。该变化纠正训练感知损失口径，而非新反射结构；是否采用两阶段结果必须以实际完成的配对指标与图像为准。

`summarize_v38_contour.py::compare`核对源帧manifest和GT一致，并支持固定控制组像素区域的SSE统计。V39输出已删除，不代表当前有可用实验结果。

`compact_v39_source_maps.py::main`用于验证manifest和阶段结果后清理训练缓存；本轮依用户要求进一步删除全部V39结果与证据，不再保留恢复源点或阶段模型。

|后续新增/补充文件|函数和用途|
|---|---|
|`analyze_surface_error_regions.py`|`save_low_normal_error_hotspots`在法线角误差<5°的共享内部选择高RGB残差连通域，保存最多3个等比例放大ROI及坐标/SSE；此条件只用于诊断，不能证明残差完全来自反射，也不向训练反馈测试图。|
|`audit_v39_frozen_geometry.py`|`main`逐项核对外观适配前后的PLY点身份、位置、法线、尺度和置信度，最大允许浮点重归一化差2e-6；单独记录特征改变，证明外观收益不是再次改变几何所得。|
|`run_refnerf_v39_full_validation.sh`|从同一外观阶段候选做零更新、SSAA2、全部训练/测试视角导出；启用预测深度/Depth Anything及法线/GT组合，不存在参考图时由既有导出逻辑跳过；计算完整测试指标，不能把前20视角配对与全量表混用。|
|`compact_v39_completed_probes.py`|默认只打印清理计划；`--execute`需要三阶段完整20视角指标、全分辨率代表图和几何冻结审计；保存配置、执行日志及待删PLY校验值后，仅删除具名零更新重复模型和控制组，保留选中几何/外观阶段。|
|`summarize_v39_full_validation.py`|`main`只接受两场景各200测试视角的完整指标，逐像素检查与V32的GT相同，记录全量RGB、标准LPIPS、原始法线/轮廓及最终train/test各类导出数量；明确这是对上一个完整检查点的整体改动对比，不是等预算消融。|






2026-09-29按用户要求：未完成六场景统一验证的局部结果不正式记录或归档。V39的模型、图像、指标、实验结果叙述与独立源码归档已撤回清理；保留工作树代码及必要的代码修复说明，不代表采纳该方法。

全量导出入口保留SOURCE_PLY_TEMPLATE与OUTPUT_SUFFIX参数；当前不存在V39训练输出或归档。源码和函数记录仅用于继续开发，不保留局部性能结论。

## 2026-11-04/05：RefReal编排、内存适配与边界方案记录

本节累积在此前相对原始IMLS的全部代码记录之后；未改写历史源码快照或覆盖旧正式结果。用户当前要求仅关闭前置额外opacity监督，保留后续法线模块。

新增`tools/refreal_session_bridge.py`（内存中交互认证）、`prepare_refreal_20261104.py`（二进制相机/最小轴法线/隔离数据镜像）、`run_refreal_20261104.py`（锁、method-major队列、hash与产物验证、小时快照、失败证据、已完成结果复用和限定次数OOM重试）、`export_evaluate_refreal_20261104.py`（相机/GT身份、实际grid恢复、完整测试评价）。完整文件/函数与适配条件见`analysis_next_round/20261104_boundary/CODE_AND_TEST_PLAN_ZH.md`。

新增`tools/refreal_memory_shading.py::install`将独立像素着色分块并用非重入checkpoint重算激活；`install_optimizer_freezing`冻结永久零学习率参数组；`skip_frozen_zero_residuals`仅跳过已冻结且末层精确为零的三个残差MLP。只安装到本轮隔离的OP/V4/V6/V32副本，不修改受保护项目和原归档。新增`check_refreal_memory_shading.py`验证输出、参数/输入梯度、空块、训练/推理及恒零分支等价；GPU小输入检查通过，最大参数梯度绝对差约1.91e-6。全1600宽、256网格短程前后向和保存/导出亦已通过，不代表长训练矩阵已完成。

边界新算法仍停留调研和接口设计，尚未改动主训练器/渲染器来实现该方案；来源与待改接口写入上述两份新文档。状态必须以本轮服务器日志/完整评价为准，不能将旧小时监测或脚本存在当作实验成功。

## 2026-11-05 16:30之后：边界实现及验证入口

RefReal完成记录、逐场景结果和独立小时审计已同步至`analysis_next_round/20261104_boundary/`。已在`utils/opaque_boundary.py`新增`prepare_target/visible_segments/point_segment_distance/boundary_terms`；在`train.py::training`加入可选目标缓存、边界损失、保留训练视角、独立梯度核对和固定网格位移上限，`arguments/__init__.py`新增默认关闭的参数。仅将GT法线搬到预测设备，保留其损失和原有参数；修正使用CPU保存相机图像时的设备错误。

`test_opaque_boundary.py`四项在AutoDL通过。`tools/evaluate_opaque_boundary.py`计算同源视角的边界F1/F2/F4、ASSD、HD95、缺失/多余覆盖、共支持raw法线及全图PSNR/SSIM/标准LPIPS。`tools/run_opaque_boundary_20261105.py`锁定相同输入和配置跑分项探针，并显式检查RefReal完成门槛；首轮仅car/coffee训练保留视角，未触碰正式测试数据作选择。运行目录`/root/autodl-tmp/opaque_boundary_20261105`；完整科学结论以受控结果为准。

`tools/select_opaque_boundary_probe.py`从逐图支持缓存重建共支持法线对照并执行预先固定的选择门槛。`tools/audit_opaque_boundary_support.py`只读审计缺失覆盖、真实视差与原始点支撑，不产生训练点。`--phase domain-audit`以同一零更新检查点单独核对等体素间距的扩域效应。`tools/finish_opaque_boundary_probe.sh`等已运行的12项探针通过产物核对后再选方法及运行扩域诊断；只有选中的方法才会启动六场景正式测试。
