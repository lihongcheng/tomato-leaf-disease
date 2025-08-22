# Tomato Leaf Disease Recognition — Code and Resources Overview

This repository consolidates code, models, experiment outputs, the Android mobile app, and paper-related assets for the tomato leaf disease recognition project. It is organized for one-click reproducibility and easy extension. This document explains the repository by function area and describes the purpose of each directory and major file.

Project Highlights
- Unified training and evaluation protocol: fixed random seed (SEED=42), unified data split and training configs, traceable evaluation reports and logs.
- Complete visualization and reproduction pipeline: training logs, curve plotting scripts, TFLite inference evaluation scripts, and the Android mobile app project.
- Publication suggestion: for the public GitHub release, you may drop paper drafts and raw SVGs; retain scripts to regenerate figures and results when needed.

1) Root-level Files
- README.md: This documentation (Chinese).
- README_EN.md: This English documentation.
- LICENSE: Open-source license (MIT).
- current_requirements.txt: Frozen Python dependencies (pip freeze) for environment reproduction.
- env.txt: Key environment info (Python/framework/hardware versions) to align environments.
- master.csv: Data index or master list (e.g., sample paths and labels) for data processing/statistics.
- model_tflite_comparison_results.csv: Summary of TFLite inference comparisons (accuracy/latency, etc.).
- MobileNetV1_checkpoint.pth, MobileNetV3_checkpoint.pth, efficientnetb0_checkpoint.pth, shufflenetV2_checkpoint.pth: PyTorch model weights (consider moving to Git LFS or Releases).
- MobileNetV3_small.tflite: Exported/quantized TFLite model for on-device inference.
- evaluate_all_models.output, evaluate_models_subprocess.output: Stdout logs from evaluation scripts.
- train_mobilenetv1.log, train_mobilenetv3_with_attention.log: Training logs (training/validation metrics).
- train_efficientnetb0.output, train_mobilenetv3.output, train_ShuffleNetV2.output: Stdout logs from training processes.
- Figures (Fig*.svg, *.png): Paper/report figures. To slim the repo, figures can be regenerated via scripts and raw SVGs can be excluded in the public release.

2) Core Scripts (root, grouped by function)
Data & Preprocessing
- data_process.py: Data cleaning/balancing and split; fixed random seed (SEED=42) for reproducibility.
- generate_augmentation_examples.py, generate_augmentation_examples_final.py: Generate data augmentation examples for visualization.

Modeling & Training
- attention_module.py: Attention module implementation (e.g., CoordAtt / improved variants).
- train_mobilenetv1.py, train_mobilenetv3.py, train_mobilenetv3_with_attention.py, train_efficientnetb0.py, train_ShuffleNetV2.py, train_ablation_study.py: Training scripts that log training/validation losses and accuracy, producing training_log.csv and training curves.

Evaluation & Deployment
- evaluate_all_models.py, evaluate_models_safe.py, evaluate_models_subprocess.py: Batch evaluation of model accuracy/efficiency with unified logs and result files.
- evaluate_models_for_tflite.py, evaluate_tflite_models.py, run_tflite_evaluation.py: TFLite inference evaluation and comparisons under various quantization/deployment settings.

Visualization & Figure Generation
- draw_training_loss_curve.py: Parse multi-model logs and plot training loss convergence curves.
- parse_and_plot.py: Generic log parsing and curve plotting (e.g., generate Fig5-4_training_loss_convergence*.svg).
- generate_grad_cam.py: Produce Grad-CAM visualizations for model interpretability.

3) Experiment Outputs
- results/: Aggregated experiment outputs
  - baseline/, ablation_no_optimizer/, ablation_no_scheduler/, ablation_no_labelsmooth/, ablation_no_sampler/: Output folders for each experimental setting, each containing:
    - training_log.csv: Epoch-wise training/validation loss and accuracy.
    - training_curve.png: Training/validation curves.

4) Mobile App (Android)
- TomatoLeafDiseaseApp/: Full Android project for on-device tomato leaf disease recognition.
  - app/: Application module (Java/Kotlin sources, resources, etc.).
  - gradle/, gradlew, settings.gradle, build.gradle, gradle.properties: Gradle build configuration and scripts.
  - Local caches/build artifacts/IDE configs (.gradle/, .idea/, app/build/, local.properties) are ignored via TomatoLeafDiseaseApp/.gitignore and should not be committed.

5) Paper & Figure Assets (optional to keep in public release)
- chapters/: Paper chapter drafts (Markdown).
- 论文_面向移动端的番茄叶病害识别研究.md, 论文设计.md, 附录A_资源清单.md: Paper draft, design notes, and resource list documents.
- Fig*.svg, *.png: Figure assets (system architecture, evaluation pipeline, training curves, confusion matrices, UI mockups, etc.).

6) Quick Reproduction Guide
- Environment:
  - Use a Python virtual environment (venv/conda) when possible.
  - pip install -r current_requirements.txt to install dependencies; use env.txt to align versions when needed.
- Data Preparation:
  - Organize data as expected by master.csv and data_process.py; run generate_augmentation_examples*.py to produce augmentation examples if needed.
- Training:
  - Example (MobileNetV3): python train_mobilenetv3.py (other models analogous).
  - Outputs are saved under results/<exp>/ (training_log.csv and training_curve.png).
- Evaluation:
  - Unified evaluation: python evaluate_all_models.py (or evaluate_models_subprocess.py for isolated runs).
  - TFLite evaluation: python run_tflite_evaluation.py or evaluate_tflite_models.py and compare with model_tflite_comparison_results.csv.
- Visualization:
  - Curves: python parse_and_plot.py or draw_training_loss_curve.py to generate training/validation curves.
  - Grad-CAM: python generate_grad_cam.py for interpretability visualizations.
- Mobile App:
  - Open TomatoLeafDiseaseApp/ in Android Studio, set up local SDK/device, and run the app.

7) Release & Compliance Tips
- Size optimization: Move large weights (*.pth, *.tflite) to GitHub Releases or Git LFS and link them in the README.
- Asset slimming: Exclude chapters/ and raw *.svg files in the public repo; keep scripts to regenerate figures.
- Privacy & safety: Avoid committing local.properties, .idea/, .gradle/, app/build/ and other local/build artifacts; TomatoLeafDiseaseApp/.gitignore covers these.

8) License
- MIT License. See LICENSE for details.