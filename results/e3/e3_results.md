E3 config (frozen from validation): {'tau': 0.7, 'alpha': 2, 'fallback': 'top1', 'split_to_inconclusive': False, 'val_macro_f1': 0.6567794651318679, 'val_macro_f1_e2_top1': 0.6508281609764395, 'grid_size': 72, 'ablation_split_on': {'tau': 0.7, 'alpha': 2, 'fallback': 'top1', 'val_macro_f1': 0.6385365691974467}}

| system | subset | n | macro F1 | acc |
|---|---|---|---|---|
| e2_top1 | all | 355 | 0.554 | 0.608 |
| e2_top1 | unseen | 257 | 0.546 | 0.568 |
| e3 | all | 355 | 0.565 | 0.617 |
| e3 | unseen | 257 | 0.554 | 0.576 |

bootstrap_all: E3 CI [0.5120752729826229, 0.6150683102315899], E2 CI [0.503967046240338, 0.6050638260695014], diff (E3-E2) mean 0.011, 95% CI [-0.012000806183185175, 0.03535732574482713]

bootstrap_unseen: E3 CI [0.496979764497288, 0.612318566377571], E2 CI [0.4897460911876019, 0.603207705275649], diff (E3-E2) mean 0.008, 95% CI [-0.015869363492453836, 0.033903837662956605]

E3 test confusion (rows=true, cols=pred, order ['Inconclusive', 'Evidence', 'NEI']):
[[25, 21, 8], [31, 68, 12], [47, 17, 126]]