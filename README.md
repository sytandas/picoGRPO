## picoGRPO

GRPO (Group Relative Policy Optimization) by DeepSeek for training LLM with RL.
This project is target small gpu, eg. 12GB rtx3060. 
And build it from scratch and optimized and experiment how to make it efficient on these GPUs if possible or not. 

## Algorithm 

Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. The idea is simple: for each question, we randomly sample multiple answers. The advantage of an answer is then defined as the normalized reward. This gets rid of the value estimation network. In particular, we implement the following algorithm:

1. For each training step, randomly sample $N$ questions $q_1, q_2, \cdots, q_N$.
2. For each question $q_i$, sample $M$ answers $a_{i,1}, a_{i,2}, \cdots, a_{i,M}$.
3. Compute the reward $r_{i,j}$ for each answer $a_{i,j}$.
4. Compute the mean and std of the rewards for each question $q_i$.

$$
\begin{aligned}
\mu_i &\leftarrow \text{mean}(r_{i,1}, r_{i,2}, \cdots, r_{i,M}) \\
\sigma_i &\leftarrow \text{std}(r_{i,1}, r_{i,2}, \cdots, r_{i,M})
\end{aligned}
$$

5. For each token $t$ in the answer $a_{i,j}$, compute the advantage as

$$A_{i,j}[t] \leftarrow \frac{r_{i,j} - \mu_i}{\sigma_i}$$

6. Compute policy gradient using PPO surrogate objective. For simplicity, we will only do one policy update per iteration, in which the gradient of the PPO objective is equivalent to following vanilla policy gradient estimation (per token).

$$
\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]
$$

7. Update the policy network $\pi(\theta)$ using the gradient. Go back to step 1.