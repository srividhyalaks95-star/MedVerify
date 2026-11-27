E4 frozen config: T=1.55, lam=0.5, threshold=0.261 (target selective acc 0.75 on validation)

- test claims: 355; accuracy without abstention: 0.617; macro F1: 0.565
- calibration (test): ECE 0.122 -> 0.103; NLL 1.042 -> 0.934
- coverage: 0.772 (274 answered, 81 abstained), 95% CI [0.7267605633802817, 0.8169014084507042]
- accuracy on answered claims: 0.679, 95% CI [0.625467824310521, 0.7328251164328823]
- macro F1 on answered: 0.6009350194160755
- errors abstained: 48 of 136; correct answers abstained: 33
- unseen: coverage 0.739, accuracy on answered 0.621
- abstained by true class: {'Inconclusive': 21, 'Evidence': 37, 'NEI': 23} (of {'Inconclusive': 54, 'Evidence': 111, 'NEI': 190})
- abstained by predicted class: {'Inconclusive': 28, 'Evidence': 47, 'NEI': 6}
- accuracy at coverage (ranked by score): {'1.0': 0.6169014084507042, '0.8': 0.676056338028169, '0.6': 0.7183098591549296, '0.4': 0.8591549295774648}
- AURC (lower is better): {'chosen_score': 0.2073911254619206, 'confidence_only': 0.20189364545578603, 'random_abstention': 0.3830985915492958}
- ablation, confidence only (no disagreement): {'threshold': 0.5577408908889117, 'coverage': 0.6873239436619718, 'sel_acc': 0.7377049180327869, 'aurc': 0.20189364545578603}

Answered-claims confusion (rows=true, cols=pred, order ['Inconclusive', 'Evidence', 'NEI']):
[[18, 8, 7], [21, 44, 9], [36, 7, 124]]