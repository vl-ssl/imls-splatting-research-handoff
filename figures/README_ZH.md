# Ref-NeRF V32 对比

目录结构沿用`comparison_refnerf_v4`与`comparison_refnerf_v6`，包含六场景的 RGB 联系图、V32/GT 法线对比、V32 相机 Z 深度预览，以及`metrics.csv`和`metrics.json`。

RGB 联系图使用每场景测试视角`00000`，列顺序为原始 IMLS、OP、V32、GT。六场景的200视角 SSIM、PSNR 和旧版 LPIPS-VGG 数值都来自各自归档的`results.json`。因此LPIPS列沿用V4/V6比较文件的历史口径，没有与重新计算的标准[-1,1] LPIPS混在一起。

法线图保留V32导出的预测法线与GT法线并排图。`*_camera_z.png`将预测相机Z取倒数后，在物体前景内按1%至99%分位数独立归一化，用于查看深度形状。V32导出没有相同格式的Depth Anything比较图，因此此图只显示预测深度。

`_source/`保留生成联系图所需的V32首视角RGB、法线图、raw几何数组和指标小文件。重建本目录时运行项目工具`tools/build_refnerf_v32_comparison.py`。
