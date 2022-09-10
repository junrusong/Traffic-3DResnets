import numpy as np
import torch
import torch.nn as nn


batch_size =2

my_cosine = torch.nn.CosineSimilarity(dim=2)
x = torch.rand(batch_size, 5, 2)
print(x)
# indices = torch.tensor([-1])
# x_lasttimestep = torch.index_select(x, 1, indices)
x_last_step = x[:, -1, :]
x_last_step = torch.unsqueeze(x_last_step, 1)
# print(x_last_step)
# print(x_last_step.size())
x_rest_step = x[:, :-1, :]
# print(x_rest_step.size())
repeat_time = x_rest_step.size()[1]
# print(repeat_time)
x_lastrepeat = x_last_step.repeat(1, repeat_time, 1)
# print(x_lastrepeat.size())
logits = my_cosine(x_lastrepeat, x_rest_step)
print(logits)
logits2 = torch.softmax(logits, dim=1)
print(logits2)





_, indices = torch.sort(logits2, descending=True)
print(indices)




k = 2
selected_indices = indices[:,:k]
print(selected_indices)
# final_indices, _ = torch.sort(selected_indices, descending=False)
# print(final_indices)

mask = torch.zeros(2, 4)
print(mask)
mask.scatter_(1, selected_indices, 1.)
mask = torch.ge(mask, 0.5)

print(mask)



logits3 = torch.masked_select(logits2, mask).view(batch_size,k)
print(logits3)
