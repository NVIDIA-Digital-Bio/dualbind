# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

class MTLLoss(torch.nn.Module):
    """Args:
            losses: a list of task specific loss terms
            num_tasks: number of tasks
    """

    def __init__(self, num_tasks=2):
        super(MTLLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_sigma = torch.nn.Parameter(torch.zeros((num_tasks)))

    def get_precisions(self):
        return 0.5 * torch.exp(- 2.0 * self.log_sigma)

    def forward(self, *loss_terms):
        assert len(loss_terms) == self.num_tasks

        total_loss = 0
        precisions = self.get_precisions()

        for task in range(self.num_tasks):
            total_loss += precisions[task] * loss_terms[task] + self.log_sigma[task]

        return total_loss
