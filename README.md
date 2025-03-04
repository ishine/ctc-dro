# CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition

We provide a minimal implementation of CTC-DRO in `dro_ctc.py` and the batch sampler enabling length-matched group losses in `duration_language_batch_sampler.py`. These files are intended to be used inside the [ESPNet](https://github.com/espnet/espnet) framework. However, they can be used with other codebases as well, as long as inputs are passed to them in compatible formats.

To reproduce the results reported in the [CTC-DRO paper](https://arxiv.org/abs/2502.01777), please refer to our codebase inside the ESPNet framework, which can be found [here](https://github.com/Bartelds/espnet/tree/master/egs2/asr_dro/asr1). 

## Batching

The `DurationBatchSampler` inside `duration_language_batch_sampler.py` returns a list of batches such that each batch contains data points from some group, with durations summing up to a provided `duration_batch_length`. It further shuffles these batches such that batches from different groups are distributed uniformly in the list. It requires

1. `shape_files` containing the duration of each audio data point with rows of `<data_point_id> <duration>`.
2. `utt2category_file` with rows of `<data_point_id> <group_id>` containing the mapping between groups and data points.

## CTC-DRO Implementation

`dro_ctc.py` contains a standalone implementation of the CTC-DRO loss function. It requires:

1. `category2numbatches` files with rows of `<group_id> <num_batches>` containing the number of batches for each group.
2. `utt2category` files with rows of `<data_point_id> <group_id>` containing the mapping between groups and data points.

It can be used with any codebase as follows:

1. Copy `dro_ctc.py` to your codebase.
2. Import the DROCTCLoss and initialize it:

```[python]
from dro_ctc import DROCTCLoss

loss_fn = DROCTCLoss(blank, zero_infinity, dro_group_count,dro_step_size, dro_q_epsilon, smoothing)
loss_fn.init_weights(train_dir, valid_dir)
```

3. Compute the CTC-DRO on batches sampled from `duration_language_batch_sampler`

```[python]
loss = loss_fn(log_probs, targets, input_lengths, target_lengths, utt_id)
```

Arguments:
```
    log_probs: Log-probs from the model for the current batch
    targets: Transcript tokens for each example in the current batch
    input_lengths: Length of input audio for each example in the current batch
    target_lengths: Length of transcript for each example in the current batch
    utt_id: data_point_id for each data point in the current batch (for mapping to groups)
```