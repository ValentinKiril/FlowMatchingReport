import time
import torch
import math
import numpy as np

from torch import nn, Tensor

import pandas as pd

# Data
Data = pd.read_csv("C:/Users/amval/Pendulum/Pendulum_Data.csv")

Data_Theta1 = Data['Theta1'].values.tolist()
Data_Theta2 = Data['Theta2'].values.tolist()
Data_Theta1 = torch.Tensor(Data_Theta1)
Data_Theta2 = torch.Tensor(Data_Theta2)
Data_Theta1 = Data_Theta1 % 2*np.pi
Data_Theta2 = Data_Theta2 % 2*np.pi


# flow_matching
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver, RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import FlatTorus, Manifold

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

torch.manual_seed(42)

def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = Data_Theta1
    #x2_ = (torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size, ), device=device) * 2)
    x2 = Data_Theta2

    data = torch.cat([x1[:, None], x2[:, None]], dim=1)

    return data.float()

def wrap(manifold, samples):
    center = torch.zeros_like(samples)

    return manifold.expmap(center, samples)

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Sequential(
                FourierFeatures(1),
                nn.Linear((input_dim + time_dim) * 2, hidden_dim),
            )

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        h = self.input_layer(h)
        output = self.main(h)

        return output.reshape(*sz)


class FourierFeatures(nn.Module):
    """Assumes input is in [0, 2pi]."""

    def __init__(self, n_fourier_features: int):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x: Tensor) -> Tensor:
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield: nn.Module, manifold: Manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.manifold.projx(x)
        v = self.vecfield(x, t)
        v = self.manifold.proju(x, v)
        return v

# training arguments
lr = 0.001
batch_size = 4096
iterations = 5001
print_every = 1000
manifold = FlatTorus()
dim = 2
hidden_dim = 512

# velocity field model init
vf = ProjectToTangent(  # Ensures we can just use Euclidean divergence.
    MLP(  # Vector field in the ambient space.
        input_dim=dim,
        hidden_dim=hidden_dim,
    ),
    manifold=manifold,
)
vf.to(device)

# instantiate an affine path object
path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)

# init optimizer
optim = torch.optim.Adam(vf.parameters(), lr=lr) 

# train
start_time = time.time()
for i in range(iterations):
    optim.zero_grad() 

    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
    x_1 = inf_train_gen(batch_size=batch_size, device=device) # sample data
    x_0 = torch.randn_like(x_1).to(device)

    x_1 = wrap(manifold, x_1)
    x_0 = wrap(manifold, x_0)

    # sample time (user's responsibility)
    t = torch.rand(x_1.shape[0]).to(device) 

    # sample probability path
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # flow matching l2 loss
    loss = torch.pow( vf(path_sample.x_t,path_sample.t) - path_sample.dx_t, 2).mean()

    # optimizer step
    loss.backward() # backward
    optim.step() # update
    
    # log loss
    if (i+1) % print_every == 0:
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
              .format(i+1, elapsed*1000/print_every, loss.item())) 
        start_time = time.time()

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x=x, t=t)

wrapped_vf = WrappedModel(vf)

# step size for ode solver
step_size = 0.01
N = 6

norm = cm.colors.Normalize(vmax=50, vmin=0)

batch_size = 2000  # batch size  # UPDATE THIS
eps_time = 1e-2
T = torch.linspace(0, 1, N)  # sample times
T = T.to(device=device)

x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
x_init = wrap(manifold, x_init)

solver = RiemannianODESolver(velocity_model=wrapped_vf, manifold=manifold)  # create an ODESolver class
sol = solver.sample(
    x_init=x_init,
    step_size=step_size,
    method="midpoint",
    return_intermediates=True,
    time_grid=T,
    verbose=True,
)

sol = sol.cpu()
T = T.cpu()

gt_samples = inf_train_gen(batch_size=50000)  # sample data
gt_samples = wrap(manifold, gt_samples)

samples = torch.cat([sol, gt_samples[None]], dim=0).numpy()

_, axs = plt.subplots(1, N + 1, figsize=(20, 3.2))
for i in range(N + 1):
    H = axs[i].hist2d(
        samples[i, :, 0],
        samples[i, :, 1],
        300,
        range=((0, 2 * math.pi), (0, 2 * math.pi)),
    )
    cmin = 0.0
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    _ = axs[i].hist2d(
        samples[i, :, 0],
        samples[i, :, 1],
        300,
        range=((0, 2 * math.pi), (0, 2 * math.pi)),
        norm=norm,
    )
    axs[i].set_aspect("equal")
    axs[i].set_xlim([0, 2 * math.pi])
    axs[i].set_ylim([0, 2 * math.pi])
    axs[i].axis("off")

    if i < N:
        axs[i].set_title("t= %.2f" % (T[i]))
    else:
        axs[i].set_title("data")

plt.tight_layout()
plt.show()