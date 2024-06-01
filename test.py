import torch
import numpy as np
 # sample
before_state = torch.random.get_rng_state()
np_seed = np.random.seed()
a = torch.randn(1) 

print('torch.backends.cudnn.deterministic',torch.backends.cudnn.deterministic)

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
b = torch.randn(1)  # sample again

torch.set_rng_state(before_state)
np.random.seed(np_seed)
torch.backends.cudnn.deterministic = False
c = torch.randn(1)  # sample

after_state = torch.random.get_rng_state()

print("RNG state progressed:", not torch.allclose(before_state, after_state))  # expected: True, actual: False
print(a)
print(b)
print(c)