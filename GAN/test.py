import torch
import torch.nn as nn

a = torch.rand(3,3,3)
print(a.shape)
norm =  nn.BatchNorm1d(3)
out = norm(a)

out = out.permute(1, 0, 2)
out = out.reshape(3, -1)
print(out.shape)
print(torch.mean(out,dim=1))
