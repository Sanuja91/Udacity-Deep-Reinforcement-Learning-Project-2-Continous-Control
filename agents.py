import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class A2C_ACKTR():
    def __init__(self, actor_critic, value_loss_coef, entropy_coef, lr = None, eps = None, alpha = None, max_grad_norm = None, acktr = False):
        """Intializes A2C agent with option of converting to ACKTR"""
        self.actor_critic = actor_critic
        self.device = self.actor_critic.device
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        ## ENABLE ONCE ACKTR IS CONFIGURED PROPERLY
        # if acktr:
        #     self.optimizer = KFACOptimizer(actor_critic)
        # else:
        #     self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps = eps, alpha = alpha)
        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps = eps, alpha = alpha)


    def act(self, states):
        """Agent acts using the actor critic"""
        mu, values = self.actor_critic(states)

        std = self.actor_critic.log_std.exp().expand_as(mu)

        dist = Normal(mu, std)
        actions = dist.sample()

        # print("ACTIONS | STATES", states.shape, "ACTIONS", actions.shape)

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        
        return values, actions, action_log_probs, dist_entropy 


    def evaluate_actions(self, states, actions):
        """Evaluates the previous actions against the selected actions"""
        mu, values = self.actor_critic(states)

        std = self.actor_critic.log_std.exp().expand_as(mu)
        
        dist = Normal(mu, std)

        # print("EVALUATIONS | STATES", states.shape, "ACTIONS", actions.shape)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        return values, action_log_probs, dist_entropy


    def update(self, rollouts):
        """Learn from experience in rollouts"""
        state_size = rollouts.states.size()[2:]
        action_size = rollouts.actions.size()[-1]

        num_steps, num_processes, _ = rollouts.rewards.size()

        # values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
        #     rollouts.states[:-1].view(-1, *state_size),
        #     rollouts.masks[:-1].view(-1, 1),
        #     rollouts.actions.view(-1, action_size))

        values, action_log_probs, dist_entropy = self.evaluate_actions(
            # rollouts.states[:-1].view(-1, *state_size),
            rollouts.states.view(-1, *state_size),
            # rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_size))

        values = values.view(num_steps, num_processes, 1)
        # print("EVALUATION VALUES |", values.shape)
        action_log_probs = action_log_probs.view(num_steps, num_processes, action_size)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        # print("VALUE LOSS", value_loss)
        action_loss = -(advantages.detach() * action_log_probs).mean()
        # print("ACTION LOSS", action_loss)
        

        ## ENABLE ONCE ACKTR IS CONFIGURED PROPERLY
        # if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
        #     # Sampled fisher, see Martens 2014
        #     self.actor_critic.zero_grad()
        #     pg_fisher_loss = -action_log_probs.mean()

        #     value_noise = torch.randn(values.size())
        #     if values.is_cuda:
        #         value_noise = value_noise.cuda()

        #     sample_values = values + value_noise
        #     vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

        #     fisher_loss = pg_fisher_loss + vf_fisher_loss
        #     self.optimizer.acc_stats = True
        #     fisher_loss.backward(retain_graph=True)
        #     self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef 
        loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def save_agent(self, fileName, average_reward, last_timestep):
        """Save the checkpoint"""
        checkpoint = {'state_dict': self.actor_critic.state_dict(), 'average_reward': average_reward, 'last_timestep': last_timestep}
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints") 
        
        filePath = 'checkpoints\\' + fileName + '.pth'
        # print("\nSaving checkpoint\n")
        torch.save(checkpoint, filePath)

    def load_agent(self, fileName):
        """Load the checkpoint"""
        # print("\nLoading checkpoint\n")
        filePath = 'checkpoints\\' + fileName + '.pth'

        if os.path.exists(filePath):
            checkpoint = torch.load(filePath, map_location = lambda storage, loc: storage)
            self.actor_critic.load_state_dict(checkpoint['state_dict'])
            average_reward = checkpoint['average_reward']
            last_timestep = checkpoint['last_timestep']
            

            print("Loading checkpoint - Last Best Reward {} (%) at Timestep {}".format(average_reward, last_timestep))
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(fileName))        


## ENABLE ONCE ACKTR IS CONFIGURED PROPERLY
# class KFACOptimizer(optim.Optimizer):
#     def __init__(self, model, lr = 0.25, momentum = 0.9, stat_decay = 0.99, kl_clip = 0.001, damping = 1e-2, weight_decay = 0, fast_cnn = False, Ts = 1, Tf = 10):
#         defaults = dict()

#         def split_bias(module):
#             for mname, child in module.named_children():
#                 if hasattr(child, 'bias') and child.bias is not None:
#                     module._modules[mname] = SplitBias(child)
#                 else:
#                     split_bias(child)

#         split_bias(model)

#         super(KFACOptimizer, self).__init__(model.parameters(), defaults)

#         self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

#         self.modules = []
#         self.grad_outputs = {}

#         self.model = model
#         self._prepare_model()

#         self.steps = 0

#         self.m_aa, self.m_gg = {}, {}
#         self.Q_a, self.Q_g = {}, {}
#         self.d_a, self.d_g = {}, {}

#         self.momentum = momentum
#         self.stat_decay = stat_decay

#         self.lr = lr
#         self.kl_clip = kl_clip
#         self.damping = damping
#         self.weight_decay = weight_decay

#         self.fast_cnn = fast_cnn

#         self.Ts = Ts
#         self.Tf = Tf

#         self.optim = optim.SGD(
#             model.parameters(),
#             lr=self.lr * (1 - self.momentum),
#             momentum=self.momentum)

#     def _save_input(self, module, input):
#         if torch.is_grad_enabled() and self.steps % self.Ts == 0:
#             classname = module.__class__.__name__
#             layer_info = None
#             if classname == 'Conv2d':
#                 layer_info = (module.kernel_size, module.stride,
#                               module.padding)

#             aa = compute_cov_a(input[0].data, classname, layer_info,
#                                self.fast_cnn)

#             # Initialize buffers
#             if self.steps == 0:
#                 self.m_aa[module] = aa.clone()

#             update_running_stat(aa, self.m_aa[module], self.stat_decay)

#     def _save_grad_output(self, module, grad_input, grad_output):
#         # Accumulate statistics for Fisher matrices
#         if self.acc_stats:
#             classname = module.__class__.__name__
#             layer_info = None
#             if classname == 'Conv2d':
#                 layer_info = (module.kernel_size, module.stride,
#                               module.padding)

#             gg = compute_cov_g(grad_output[0].data, classname, layer_info,
#                                self.fast_cnn)

#             # Initialize buffers
#             if self.steps == 0:
#                 self.m_gg[module] = gg.clone()

#             update_running_stat(gg, self.m_gg[module], self.stat_decay)

#     def _prepare_model(self):
#         for module in self.model.modules():
#             classname = module.__class__.__name__
#             if classname in self.known_modules:
#                 assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
#                                     "You must have a bias as a separate layer"

#                 self.modules.append(module)
#                 module.register_forward_pre_hook(self._save_input)
#                 module.register_backward_hook(self._save_grad_output)

#     def step(self):
#         # Add weight decay
#         if self.weight_decay > 0:
#             for p in self.model.parameters():
#                 p.grad.data.add_(self.weight_decay, p.data)

#         updates = {}
#         for i, m in enumerate(self.modules):
#             assert len(list(m.parameters())
#                        ) == 1, "Can handle only one parameter at the moment"
#             classname = m.__class__.__name__
#             p = next(m.parameters())

#             la = self.damping + self.weight_decay

#             if self.steps % self.Tf == 0:
#                 # My asynchronous implementation exists, I will add it later.
#                 # Experimenting with different ways to this in PyTorch.
#                 self.d_a[m], self.Q_a[m] = torch.symeig(
#                     self.m_aa[m], eigenvectors=True)
#                 self.d_g[m], self.Q_g[m] = torch.symeig(
#                     self.m_gg[m], eigenvectors=True)

#                 self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
#                 self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

#             if classname == 'Conv2d':
#                 p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
#             else:
#                 p_grad_mat = p.grad.data

#             v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
#             v2 = v1 / (
#                 self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
#             v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

#             v = v.view(p.grad.data.size())
#             updates[p] = v

#         vg_sum = 0
#         for p in self.model.parameters():
#             v = updates[p]
#             vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

#         nu = min(1, math.sqrt(self.kl_clip / vg_sum))

#         for p in self.model.parameters():
#             v = updates[p]
#             p.grad.data.copy_(v)
#             p.grad.data.mul_(nu)

#         self.optim.step()
#         self.steps += 1