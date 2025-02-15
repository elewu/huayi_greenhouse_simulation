
import torch



def get_advantage_and_return(value, reward, done, γ, λ):
    """
        reward: torch.Size([batch, num_trajs, num_steps])
    """
    lastgaelam = 0
    advantage_reversed = []

    num_steps = reward.shape[-1]

    for t in reversed(range(num_steps)):
        next_non_terminal = 1.0 - done[..., t]
        nextvalues = value[..., t + 1] if t < num_steps - 1 else 0.0
        delta = reward[..., t] + γ * next_non_terminal * nextvalues - value[..., t]
        lastgaelam = delta + γ * λ * next_non_terminal * lastgaelam
        advantage_reversed.append(lastgaelam)
    advantage = torch.stack(advantage_reversed[::-1], dim=-1)
    returns = advantage + value
    # advantage = whiten(advantage)
    return advantage, returns




