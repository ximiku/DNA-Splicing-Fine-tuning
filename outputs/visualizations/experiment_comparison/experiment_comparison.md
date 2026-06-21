| experiment | split_or_eval | training_mode | accuracy | macro_f1 | mean_auroc | mean_auprc | motif_hard_fpr | train_runtime_sec | peak_gpu_reserved_gib | trainable_parameters | trainable_fraction | final_model_size_gib | checkpoint_size_gib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_baseline | random full test | full | 0.966906 | 0.966328 | 0.996329 | 0.993541 | 0.061291 | 7367.853300 |  | 117070851 | 1.000000 | 0.436497 | 3.926667 |
| chrom_holdout_full | chr9/chr10 test | full | 0.967469 | 0.966799 | 0.996493 | 0.993841 | 0.059621 | 6709.747000 |  | 117070851 | 1.000000 | 0.436498 | 3.926690 |
| random_only_ablation | original full test | full | 0.967033 | 0.966504 | 0.996541 | 0.993890 | 0.071004 | 7345.236100 |  | 117070851 | 1.000000 | 0.436498 | 3.926668 |
| linear_probe | random full test | linear_probe | 0.631480 | 0.562948 | 0.777888 | 0.623310 | 0.127121 | 2892.696200 | 16.580078 | 592899 | 0.005064 | 0.436497 | 1.322716 |
| lora | random full test | lora | 0.949439 | 0.948605 | 0.992616 | 0.987111 | 0.073592 | 6964.450000 | 24.974609 | 887811 | 0.007526 | 0.003675 | 0.031039 |
