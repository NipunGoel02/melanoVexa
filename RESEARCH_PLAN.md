# Research protocol: SCC-focused lesion classification

## Proposed question

On a fixed three-class ISIC 2019 subset, how does focal loss change the trade-off between macro-F1 and SCC sensitivity compared with cross-entropy when both use the same EfficientNet-B0 feature-attention head and weighted sampler?

This is a controlled empirical study, not a claim that focal loss or feature attention is a new algorithm. Prior ISIC work already studies class imbalance, loss balancing, and attention. The project contribution should be the controlled comparison, SCC-centered analysis, and reproducible reporting.

## Why this experiment

- The prepared multiclass split contains 5,082 training images and 1,270 held-out tuning images. SCC has 376 training and 94 tuning images, about 7.4% of each split.
- The existing training scripts use inverse-frequency weighted sampling, but save the “best” checkpoint according to training loss. This research runner instead selects checkpoints using held-out macro-F1.
- The existing feature-attention head learns one softmax-normalized feature vector shared across images. It is feature-wise gating; it is not a spatial attention map.
- The focal-loss arm retains the same weighted sampler and changes the loss relative to the attention + cross-entropy control. It deliberately has no second class-weight multiplier, avoiding sampler-plus-alpha double reweighting.

## Controlled arms

| Arm | Backbone/head | Training objective | Sampler |
| --- | --- | --- | --- |
| `plain_ce` | EfficientNet-B0 + original custom classifier | Cross-entropy | Inverse-frequency weighted sampling |
| `attention_ce` | EfficientNet-B0 + feature-attention head | Cross-entropy | Same weighted sampling |
| `attention_focal` | Same feature-attention head | Focal loss, gamma 2 | Same weighted sampling |

Run all arms with the same seed, train/tuning split, batch size, optimizer, learning-rate schedule, and ImageNet initialization. The direct focal-loss comparison is `attention_ce` vs `attention_focal`; the plain arm measures the feature-head ablation.

## Outcomes

- **Checkpoint-selection metric:** validation macro-F1.
- **Key SCC outcome:** SCC recall/sensitivity at the macro-F1-selected checkpoint; also report the complete SCC precision/recall trade-off rather than interpreting recall alone.
- **Other outcomes:** per-class precision/recall/F1, accuracy, Cohen’s kappa, confusion matrix, one-vs-rest ROC-AUC and SCC PR-AUC.
- Report results across multiple seeds with uncertainty intervals. Tune thresholds and choose epochs on tuning data only.
- Preserve per-image predictions for paired comparisons and failure analysis.

## Pilot run (one seed, internal tuning split)

| Arm | Best epoch | Accuracy | Macro-F1 | SCC recall | SCC F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain EfficientNet-B0 + CE | 9 | 0.9173 | 0.8599 | 0.6702 | 0.7200 |
| Feature attention + CE | 4 | 0.8441 | 0.5804 | 0.0000 | 0.0000 |
| Feature attention + focal loss | 4 | 0.8441 | 0.5815 | 0.0000 | 0.0000 |

These preliminary results do **not** show that focal loss improves SCC detection. At the macro-F1-selected checkpoints, both attention variants predicted zero SCC validation images as SCC. Focal changed macro-F1 by only 0.0011 over the attention + CE control, while the plain head performed much better overall. This suggests a feature-attention/training-recipe failure mode that needs investigation; it is not evidence of clinical performance or generalization. The epoch-wise recall also fluctuated substantially, reinforcing the need for multiple seeds and an untouched test set.

## Important boundary

`data_mult/val` has already been used for previous model comparisons. Treat it as a tuning/internal-testing split, not as an untouched final test. For a paper’s final estimate, evaluate selected models once on a separate, previously unused test set; audit duplicate/related lesions across partitions first. If no such test set is available, describe results as internal and exploratory, and state that limitation.

## Run the ablation

From the repository root, with the project environment activated:

```powershell
python classifiers/multiple_classifier/fit_research.py --experiment all --epochs 10 --batch_size 8 --seed 42
```

Runs are stored separately under `out/research_ablation/<arm>/`: checkpoint, `manifest.json`, and per-image tuning predictions. Repeat with other seeds into separate output folders before combining results; do not overwrite prior runs.

## Reporting

Follow the [CLAIM 2024 reporting checklist](https://pubs.rsna.org/doi/10.1148/ryai.240300). Report the image-level split method, class counts, data source, reference standard, image preprocessing, initialization, sampler, loss, optimizer/schedule, seeds, checkpoint-selection rule, hardware/software, and all class-wise outcomes. Cite ISIC 2019 and its component datasets per the [ISIC attribution instructions](https://challenge.isic-archive.com/data/).
