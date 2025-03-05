# CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition
Code associated with the paper: CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition.

**Abstract:** Modern deep learning models often achieve high overall performance, but consistently fail on specific subgroups. Group distributionally robust optimization (group DRO) addresses this problem by minimizing the worst-group loss, but it fails when group losses misrepresent performance differences between groups. This is common in domains like speech, where the widely used connectionist temporal classification (CTC) loss scales with input length and varies with linguistic and acoustic properties, leading to spurious differences between group losses. We present CTC-DRO, which addresses the shortcomings of the group DRO objective by smoothing the group weight update to prevent overemphasis on consistently high-loss groups, while using input length-matched batching to mitigate CTC's scaling issues. We evaluate CTC-DRO on the task of multilingual automatic speech recognition (ASR) across five language sets from the ML-SUPERB 2.0 benchmark. CTC-DRO consistently outperforms group DRO and CTC-based baseline models, reducing the worst-language error by up to 47.1% and the average error by up to 32.9%. CTC-DRO can be applied to ASR with minimal computational costs, and offers the potential for reducing group disparities in other domains with similar challenges.

This repository provides a minimal, standalone implementation of CTC-DRO. It includes two main components:

- **CTC-DRO loss function:** Implemented in [`ctc_dro.py`](./ctc_dro.py), this file contains a standalone implementation of the CTC-DRO loss function.
- **Length-matched batch sampler:** Implemented in [`duration_batch_sampler.py`](./duration_batch_sampler.py), this batch sampler returns batches of data points from the same group whose durations sum to a specified target length.

These modules are designed for seamless integration with the [ESPNet](https://github.com/espnet/espnet) framework but can also be incorporated into other codebases as long as the input formats are compatible.

To reproduce the results reported in the [CTC-DRO paper](https://arxiv.org/abs/2502.01777), please refer to our complete codebase within the ESPNet framework, available [here](https://github.com/Bartelds/espnet/tree/master/egs2/asr_dro/asr1).

---

## Citation

```bibtex
@misc{bartelds2025ctcdrorobustoptimizationreducing,
      title={CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition}, 
      author={Martijn Bartelds and Ananjan Nandi and Moussa Koulako Bala Doumbouya and Dan Jurafsky and Tatsunori Hashimoto and Karen Livescu},
      year={2025},
      eprint={2502.01777},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01777}, 
}
```

---

## Batching

The file [`duration_batch_sampler.py`](./duration_batch_sampler.py) implements the `DurationBatchSampler`, which ensures that:
- Each batch contains only examples from one language or category.
- The total duration of audio in each batch is approximately equal to a specified target (`duration_batch_length`).
- Batches are uniformly shuffled so that examples from different groups are well distributed over training iterations.

**Requirements:**
1. **Shape files:**  
   A set of files containing the duration of each audio file. Each file should be formatted with rows of:  
   `<data_point_id> <duration>`  
   (The loader uses a CSV integer format.)
2. **utt2category file:**  
   A file mapping each data point to its group. Each line should be formatted as:  
   `<data_point_id> <group_id>`

The sampler uses these files to verify that all keys match, sort utterances by duration (largest first), and apply a greedy bin-packing algorithm to generate duration-equalized batches.

---

## CTC-DRO implementation

The file [`ctc_dro.py`](./ctc_dro.py) contains the implementation of the CTC-DRO loss function.

**Requirements:**
1. **category2numbatches file:**  
   A file with rows formatted as:  
   `<group_id> <num_batches>`  
   indicating the number of batches for each group.
2. **utt2category file:**  
   A file with rows formatted as:  
   `<data_point_id> <group_id>`  
   mapping each data point to its group.

The `init_weights` method of `DROCTCLoss` loads these files from a specified training directory and initializes the internal state required for group-wise loss aggregation.

---

## Usage example

To integrate the CTC-DRO loss into your codebase, follow these steps:

1. **Copy the file:**  
   Copy [`ctc_dro.py`](./ctc_dro.py) into your project directory.

2. **Import and initialize:**  
   Import the `DROCTCLoss` class and initialize it with the required hyperparameters:
   
   ```python
   from dro_ctc import DROCTCLoss

   # Initialize the loss function with your desired settings:
   # - blank: the blank token for CTC loss
   # - zero_infinity: whether to zero out infinite losses
   # - dro_group_count: total number of groups
   # - dro_step_size: step size for updating group weights
   # - dro_q_epsilon: small constant to prevent group weights from reaching zero
   # - smoothing: smoothing parameter for the weight update (set >0 to use smoothing)
   loss_fn = DROCTCLoss(blank=0, zero_infinity=True, dro_group_count=6, dro_step_size=0.0001, dro_q_epsilon=1e-10, smoothing=0.1)

   # Initialize weights using your training and validation directories.
   # These directories should contain 'category2numbatches' and 'utt2category' files.
   loss_fn.init_weights(train_file="path/to/train_dir", valid_file="path/to/valid_dir")

3. **Compute the loss:**

    When processing a batch, compute the loss as follows:

    ```python
    # log_probs: Log probabilities from your model (Tensor)
    # targets: Target transcript tokens (Tensor)
    # input_lengths: Lengths of input audio for each example (Tensor)
    # target_lengths: Lengths of target transcripts for each example (Tensor)
    # utt_id: List of data point IDs for each example (used to map examples to groups)

    loss = loss_fn(log_probs, targets, input_lengths, target_lengths, utt_id)
    ```

    The function returns the CTC-DRO scaled loss for training purposes, and it returns the standard CTC loss during validation.

---

## License

This repository is distributed under the terms of the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
