import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor
import pdb 

class DROCTCLoss(torch.nn.Module):
    def __init__(self, blank=0, zero_infinity=False, dro_group_count=0, dro_step_size=0.01, dro_q_epsilon=1e-10, smoothing=0):
        '''
        Wrapper to compute the CTC-DRO loss for a given batch

        Arguments:
            blank: Blank (epsilon) token for CTC loss
            zero_infinity: Whether to zero infinite losses
            dro_group_count: Total number of groups
            dro_step_size: Step size for CTC-DRO group weight updates
            dro_q_epsilon: Small constant added to group weights to ensure they never reach zero for any group.
            smoothing: Alpha for CTC-DRO group weight updates
        '''
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size

        # Uniform initialization of DRO group weights
        self.dro_q = torch.ones(self.dro_group_count) * 1.0/self.dro_group_count
        self.dro_q_epsilon = dro_q_epsilon
        self.group_id_to_ix = {}
        self.smoothing = smoothing

    def init_weights(self, train_file, valid_file):
        group_sizes = {}

        # Load number of batches for each group
        # category2numbatches has format: <group_id> <num_batches>
        with open(str(train_file) + '/category2numbatches', 'r') as f:
            for line in f:
                line = line.strip().split()
                group_sizes[line[0]] = int(line[1])
        
        # Load mapping from data points to group
        # utt2category has format: <data_point_id> <group_id>
        self.utt2category = {}
        with open(str(train_file) + '/utt2category', 'r') as f:
            for line in f:
                line = line.strip().split()
                self.utt2category[line[0]] = line[1]

        with open(str(valid_file) + '/utt2category', 'r') as f:
            for line in f:
                line = line.strip().split()
                self.utt2category[line[0]] = line[1]
        
        # losses for batches from each group encountered during training
        self.group_losses = {}
        for _ in range(len(group_sizes)):
            self.group_losses[_] = []

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, utt_id: List[str], valid: bool = True) -> Tensor:
        '''
        Compute the CTC-DRO loss for a batch

        Arguments:
            log_probs: Log-probs from the model for the current batch
            targets: Transcript tokens for each example in the current batch
            input_lengths: Length of input audio for each example in the current batch
            target_lengths: Length of transcript for each example in the current batch
            utt_id: data_point_id for each data point in the current batch (for mapping to groups)
            valid: Set to true if being run during validation
        '''
        
        log_probs = torch.transpose(log_probs, 0, 1)

        batch_lang_ids = [self.utt2category[_] for _ in utt_id]

        batch_lang_q_indices = []
        for lang_id in batch_lang_ids:
            if lang_id not in self.group_id_to_ix:
                self.group_id_to_ix[lang_id] = len(self.group_id_to_ix)
            batch_lang_q_indices.append(self.group_id_to_ix[lang_id])

        losses = F.ctc_loss(
            log_probs, 
            targets, input_lengths, target_lengths, 
            self.blank, reduction='none',
            zero_infinity=self.zero_infinity
        )

        step_size = self.dro_step_size

        for q_ix in set(batch_lang_q_indices):
            # calculate losses for each group
            group_losses = torch.tensor([
                losses[i]
                for i in range(losses.shape[0])
                if batch_lang_q_indices[i] == q_ix
            ])

            group_loss = torch.sum(group_losses)
            self.group_losses[q_ix].append(group_loss)

        # Check to see if each group has been encountered at least once
        check = True
        for _ in self.group_losses:
            if len(self.group_losses[_]) == 0:
                check = False
                break 

        if check:
            # Perform the CTC-DRO group weight update
            for _ in self.group_losses:
                update_term = sum(self.group_losses[_])/len(self.group_losses[_])
                if self.smoothing > 0:
                    # Use smoothing
                    self.dro_q[_] *= torch.exp((update_term * step_size)/(self.dro_q[_] + self.smoothing))
                    print("Update Magnitude", torch.exp((update_term * step_size)/(self.dro_q[_] + self.smoothing)))
                else:
                    # Original group DRO update
                    self.dro_q[_] *= torch.exp(update_term * step_size)
                    print("Update Magnitude", torch.exp(update_term * step_size))

            self.normalize_dro_q()
            for _ in self.group_losses:
                self.group_losses[_] = []
        
        dro_losses = torch.stack([
            losses[ix] * self.dro_q[batch_lang_q_indices[ix]] 
            * self.dro_group_count
            for ix in range(losses.shape[0])
        ])

        # Return CTC-DRO losses for gradient descent on model
        if not valid:
            return dro_losses
        else:
            return losses

    def normalize_dro_q(self):
        # normalize the group weights
        self.dro_q += self.dro_q_epsilon
        self.dro_q = self.dro_q / self.dro_q.sum()
