import torch

import torch.nn as nn
import torch.nn.functional as F


class BinarizedF():
  def forward(input, threshold):
    a = torch.ones_like(input).cuda()
    b = torch.zeros_like(input).cuda()
    output = torch.where(input>=threshold,a,b)
    return output

  


