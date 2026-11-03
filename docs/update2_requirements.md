Checklist for Success:
1) Dual-Head Architecture (2 points): Present your model diagram showing the integration of Switch and Duration heads into the shared transformer backbone.
2) Loss Function Optimization (3 points): Discuss your multitask loss weighting (Ltotal​=λ1​Lsw​+λ2​Ldur​) and your implementation of ignore_index=-1 for the duration head.
3) Experiment A: Comparative Study (2 points): Compare two different backbones (e.g., mBERT vs. XLM-R). Which architecture shows higher predictive accuracy at this stage?
4) Initial Universality Audit (2 points): Use a bar chart to show Anticipatory F1 across your different language pairs. Is the model’s performance consistent?
5) LMS Preparation (1 point): Report on training convergence for both tasks (Switch vs. Duration).