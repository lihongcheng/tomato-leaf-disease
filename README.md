# 番茄叶病害识别（Tomato Leaf Disease Recognition）— 代码与资源总览

本仓库汇总了番茄叶病害识别项目的代码、模型、实验产出、移动端应用与论文相关资产，便于一键复现与二次开发。本文档按照“功能分类 + 目录与文件说明”的方式，逐项说明各文件的作用。

项目要点
- 统一训练与评估协议：固定随机种子（SEED=42）、统一数据划分与训练配置，评估报告与日志可溯源。
- 完整可视化与复现实验链路：训练日志、曲线图生成脚本、TFLite 推理评估脚本与移动端应用工程齐备。
- 可发布建议：对外发布的 GitHub 仓库可剔除论文草稿与 SVG 源文件，仅保留脚本可重现图表与结果。

一、根目录文件说明
- README.md：本说明文档。
- LICENSE：开源许可（MIT）。
- current_requirements.txt：当前 Python 依赖冻结列表（pip freeze 导出），用于快速复现实验环境。
- env.txt：训练/推理环境关键信息（Python/框架/硬件版本等），辅助环境对齐与复现。
- master.csv：数据索引或主列表（如样本路径、标签等，用于数据处理/统计）。
- model_tflite_comparison_results.csv：TFLite 推理比较的汇总结果（准确率/时延等）。
- MobileNetV1_checkpoint.pth、MobileNetV3_checkpoint.pth、efficientnetb0_checkpoint.pth、shufflenetV2_checkpoint.pth：PyTorch 训练权重（可按需移至 Git LFS 或 Releases）。
- MobileNetV3_small.tflite：导出/量化后的 TFLite 模型（移动端推理用）。
- evaluate_all_models.output、evaluate_models_subprocess.output：评估脚本运行的标准输出日志。
- train_mobilenetv1.log、train_mobilenetv3_with_attention.log：训练日志（含训练/验证指标等）。
- train_efficientnetb0.output、train_mobilenetv3.output、train_ShuffleNetV2.output：训练过程标准输出日志。

二、核心脚本（根目录，按功能分组）
数据与预处理
- data_process.py：数据清洗/均衡与划分脚本；固定随机种子（SEED=42）确保复现。
- generate_augmentation_examples.py、generate_augmentation_examples_final.py：生成数据增强示例图片，辅助可视化与说明。

模型与训练
- attention_module.py：注意力模块实现（如 CoordAtt/改进模块）。
- train_mobilenetv1.py、train_mobilenetv3.py、train_mobilenetv3_with_attention.py、train_efficientnetb0.py、train_ShuffleNetV2.py、train_ablation_study.py：模型训练脚本；统一记录训练/验证损失与准确率，输出 training_log.csv 与训练曲线。

评估与部署
- evaluate_all_models.py、evaluate_models_safe.py、evaluate_models_subprocess.py：批量评估各模型精度/效率，输出统一日志与结果文件。
- evaluate_models_for_tflite.py、evaluate_tflite_models.py、run_tflite_evaluation.py：TFLite 推理评估与对比，支持不同量化/部署工况。

可视化与图表生成
- draw_training_loss_curve.py：解析多模型日志，绘制训练损失收敛曲线。
- parse_and_plot.py：通用日志解析与曲线绘制（如生成 Fig5-4_training_loss_convergence*.svg）。
- generate_grad_cam.py：生成 Grad-CAM 可视化，分析模型关注区域。

三、实验产出
- results/：实验结果汇总目录
  - baseline/、ablation_no_optimizer/、ablation_no_scheduler/、ablation_no_labelsmooth/、ablation_no_sampler/：各实验设定的输出目录，均包含：
    - training_log.csv：按 epoch 记录训练/验证损失与准确率。
    - training_curve.png：训练/验证曲线图。

四、移动端应用（Android）
- TomatoLeafDiseaseApp/：完整 Android 应用工程，用于在移动端进行番茄叶病害识别。
  - app/：应用模块（Java/Kotlin 源码、资源等）。
  - gradle/、gradlew、settings.gradle、build.gradle、gradle.properties：Gradle 构建配置与脚本。
  - .gradle/、.idea/、app/build/、local.properties：本地缓存/构建产物/IDE 配置（建议在仓库中忽略或不提交）。


五、快速复现指南
- 环境准备：
  - 建议使用 Python 虚拟环境（venv/conda）。
  - pip install -r current_requirements.txt 安装依赖；必要时参考 env.txt 对齐版本。
- 数据准备：
  - 根据 master.csv 与 data_process.py 的约定组织数据目录；如需演示增强示例，可运行 generate_augmentation_examples*.py。
- 训练：
  - 以 MobileNetV3 为例：python train_mobilenetv3.py（其他模型脚本同理）。
  - 训练产出位于 results/<实验名>/（training_log.csv 与 training_curve.png）。
- 评估：
  - 统一评估：python evaluate_all_models.py（或使用 evaluate_models_subprocess.py 进行隔离评估）。
  - TFLite 评估：python run_tflite_evaluation.py 或 evaluate_tflite_models.py，并对比 model_tflite_comparison_results.csv。
- 可视化：
  - 日志曲线：python parse_and_plot.py 或 draw_training_loss_curve.py 生成训练/验证曲线。
  - Grad-CAM：python generate_grad_cam.py 输出可视化结果。
- 移动端：
  - 使用 Android Studio 打开 TomatoLeafDiseaseApp/，配置本地 SDK 与设备后即可运行。


六、许可
- 本项目采用 MIT 许可证，详见 LICENSE。