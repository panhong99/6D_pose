"""Policy networks for the basket insertion (2026-10-13).

    BC (train.py bc):      obs --Normalizer--> MLP --> [6 TCP-step means, 1 gripper-open logit]
    RL (train.py rl):      action = BC(obs) (frozen) + residual_scale * ResidualActor(obs)    critic: MLP(obs) -> V
The MLP is the policy function itself; BC fits it to expert (obs, action) pairs, PPO trains the residual on top.
Actions are in the slot frame and scaled to step_max (5 cm/s, 10 deg/s): a tanh head can never exceed the speed caps.
"""
import torch
from torch import nn


def mlp(sizes, act='elu', out_act=None):
    acts = dict(elu=nn.ELU, relu=nn.ReLU, tanh=nn.Tanh)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(acts[act]())
    if out_act:
        layers.append(acts[out_act]())
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """Fixed obs standardisation from the training data (stored with the weights, so eval / the robot use the same)."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))

    def fit(self, x):
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, x):
        return (x - self.mean) / self.std


class BCPolicy(nn.Module):
    """MLP policy: obs (17) -> 6 continuous TCP steps in [-step_max, step_max] + gripper-open logit."""

    def __init__(self, obs_dim, step_max, hidden=(512, 256, 128), act='elu'):
        super().__init__()
        self.norm = Normalizer(obs_dim)
        self.body = mlp([obs_dim, *hidden, len(step_max) + 1], act)
        self.register_buffer('step_max', torch.as_tensor(step_max, dtype=torch.float32))

    def forward(self, obs):
        """-> (continuous action in step units [-1, 1], gripper logit)"""
        out = self.body(self.norm(obs))
        return torch.tanh(out[..., :-1]), out[..., -1]

    @torch.no_grad()
    def act(self, obs):
        """Executed action: metres / radians per step + open flag."""
        cont, logit = self(obs)
        return cont * self.step_max, (logit > 0).float()


class ResidualActorCritic(nn.Module):
    """PPO model on a frozen BC: mean action = BC + scale * residual (both in step units), Gaussian with learned std."""

    def __init__(self, bc: BCPolicy, scale=0.2, hidden=(512, 256, 128), act='elu', init_std=0.3):
        super().__init__()
        self.bc = bc.eval()
        for p in self.bc.parameters():
            p.requires_grad_(False)
        n_in, n_out = bc.norm.mean.numel(), bc.step_max.numel() + 1
        self.actor = mlp([n_in, *hidden, n_out], act)
        self.critic = mlp([n_in, *hidden, 1], act)
        nn.init.zeros_(self.actor[-1].weight)          # residual starts at exactly 0: the policy IS the BC at iteration 0
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((n_out,), float(torch.log(torch.tensor(init_std)))))
        self.scale = scale

    def mean(self, obs):
        cont, logit = self.bc(obs)
        r = self.actor(self.bc.norm(obs))
        return torch.cat([cont, logit[..., None]], -1) + self.scale * r

    def value(self, obs):
        return self.critic(self.bc.norm(obs)).squeeze(-1)

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())


def save(path, model, meta):
    torch.save(dict(state=model.state_dict(), meta=meta), path)


def load_bc(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    m = ck['meta']
    model = BCPolicy(m['obs_dim'], m['step_max'], m['hidden'], m['activation']).to(device)
    model.load_state_dict(ck['state'])
    return model.eval(), m


class InsertPolicy(nn.Module):
    """(10/15) insert-correction policy (one decision per insertion, PPO as a contextual bandit): obs -> Gaussian over
    the pre-tanh correction (expert.INSERT_SCALE).  The mean head starts at 0 = the hand-tuned settings, so iteration 0
    IS the current method; a separate value head is the baseline."""

    def __init__(self, obs_dim, act_dim, hidden=(128, 128), init_std=0.5):
        super().__init__()
        self.mu = mlp([obs_dim, *hidden, act_dim], 'elu')
        self.v = mlp([obs_dim, *hidden, 1], 'elu')
        nn.init.zeros_(self.mu[-1].weight)
        nn.init.zeros_(self.mu[-1].bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))

    def forward(self, obs):
        return self.mu(obs), self.log_std.exp().expand(obs.shape[0], -1)

    def value(self, obs):
        return self.v(obs)[..., 0]
