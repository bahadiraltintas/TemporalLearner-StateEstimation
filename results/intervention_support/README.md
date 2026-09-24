# Intervention-value support diagnostics

The support analysis evaluates candidate state-action pairs on the fixed 200-student test partition.

- **Min-max support:** a candidate is outside support if any numeric learner-state variable lies outside the training range for that activity.
- **NN95 support:** a candidate is outside support if its nearest standardized training state for that activity is farther than the 95th percentile of leave-one-out nearest-neighbour distances among training states for that activity.

These are operational overlap diagnostics, not formal causal positivity tests.
