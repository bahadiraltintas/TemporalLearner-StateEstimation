# Independent synthetic potential-outcome benchmark

This benchmark is independent of the frozen `expected_learning_gain` target. It defines a known potential-outcome surface for each learner state and each of the five eligible activities, then adds Gaussian noise (SD 0.35) only to the factual observed action outcome used for training.

It is included to test the action-conditioned model's ability to recover a known synthetic value surface and best-action ranking. It is **not** evidence of causal treatment effects, pedagogical validity, or real learning effectiveness.
