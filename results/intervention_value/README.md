# State-dependent intervention value

An action-conditioned Ridge model uses explicit activity × learner-state interactions
to estimate `G_hat(S_t, a)` for five candidate activities. Guided Review is excluded
because it has only three observations.

**Critical limitation:** the frozen dataset contains only one observed activity and one
observed gain per student. There are no counterfactual outcomes for unchosen activities.
Therefore these are conditional predictive estimates, not causal treatment effects.
