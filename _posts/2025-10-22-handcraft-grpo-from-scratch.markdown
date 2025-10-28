---
layout: post
title:  "Handcrafting GRPO from Scratch"
date:   2025-10-22 22:58:09 +0800
categories: jekyll update
usemathjax: true
---

# Introduction

GRPO (Group Relative Policy Optimization) is the reinforcement learning (RL) method to fine-tune a large language model (LLM) by comparing different actions and making small, controlled updates using a group of observations. It’s like a smart way to learn from experience without making drastic changes that could mess things up.

Handcrafting GRPO is especially useful for projects with strict constraints where off-the-shelf solutions cannot express required regularizers or safety checks, or the research explorations that require new reward functions or update rules. And we are also free to adapt and optimize it according to our own needs. At the same time, it is quite useful for those who wish to dive deeply into the underlying mechanisms of GRPO.

This post demonstrates how to handcraft GRPO from scratch rather than adapting an existing library or recipe. That means, we will code the dataset pipeline, loss functions, KL/regularization, clipping, etc. If you are also interested in looking inside GRPO, let's dive in.

#### Modelling LLM Fine-tuning as Reinforcement Learning

Both PPO and GRPO are reinforcement learning approaches, and a common question about this approach is how we can link LLM fine-tuning with reinforcement learning.

Briefly speaking, the generation of each token can be viewed as an *action* performed by the LLM. And when the language model completes its generation (i.e. a sentence), a reward model or a set of rules is used to evaluate whether the generated sentence is good or not, and assign a *reward* according to the evaluation result. At the end of each step, the model is updated according to the reward received.

By introducing the *action* and *reward*, the LLM fine-tuning process can be modelled as the reinforcement learning training process, and employ the reinforcement learning algorithms to solve it.

# Procedure of LLM Fine-Tuning with GRPO

When fine-tuning with GRPO, there are three models:

* Policy model (or new model), denoted as `policy_model` in the program code, is the trainable model. It is the one we are fine-tuning with GRPO updates, and represents the current policy $\pi_{\theta}$.
* Old model, denoted as `old_model`, is a frozen copy of the policy from the previous update step $\pi_{\theta_{\text{old}}}$. It’s used to compute the probability ratio.
* Reference model, generally denoted as `ref_model`, is the original pretrained model (before any fine-tuning).
It provides a stable distribution $\pi_{\text{ref}}$ against which we measure KL divergence.

Since these models are very large, we often try to reduce the number of active copies. In my program code below, I reuse a single model to compute both ‘policy’ and ‘old’ log-probs by snapshotting the old logits before inner updates.

The GRPO fine-tuning process is composed of the following steps:

1. First, we sample serveral questions (with the number of questions denoted as `B`). And for each question, we also use the policy model to generate several answers (denoted as `G`). The final sampling results are the `input_ids` tensor with shape `(B*G, L+prompt_len)`, in which `L` represents the maximum length of the generated answering tokens, and `prompt_len` represents the maximum length of the question tokens.

2. Run a forward step with the policy model and `input_ids` to compute the probability that each token in `input_ids` will be generated.

3. Calculate the loss function using the probability values for the tokens in `input_ids`, not only from the policy model in Step 2, but also from the reference model and the old model.

4. Update the policy model parameter with `loss.backward()` and `optimizer.step()`.

5. For each sampled `input_ids`, repeat Steps 2 ~ 4 for several times, as specified by `mu`.

6. Repeat the whole process (Step 1 ~ 5) for a few epochs (typically 1 or 2 epochs). One epoch means the entire dataset is seen once.

Based on the steps explained above, we can see that the GRPO training process is primarily composed of two parts:

* Data curation, or sampling different answers from the question. In our code, it is the `GRPODataSet` class.
* Training flow, which includes the implementation of the loss function. In the code, that's the `GRPOTrainer` class.

#### Inner and Outer Loop during Fine-Tuning Procedure

Note that there is an inner loop (Step 5) and an outer loop (Step 6) during the GRPO fine-tuning. The inner training loop uses the same question and answer samples, and is repeated by several times controlled by `mu`. The outer training loop, on the other hand, resamples the question and answer tokens every time.

Now, let's go through them one by one.

# How the Training Set Is Created for GRPO?

We can build the training set for GRPO from a public dataset like [GSM8K](https://huggingface.co/datasets/openai/gsm8k), which contains grade school math problems. GSM8K itself is composed of a training set and a testing set, and has at least two columns containing questions and answers as the ground-truth answers for each question.

Briefly speaking, there are 2 primary steps in creating the training set for GRPO:

1. First of all, sample a batch of records (4 or 8, as denoted by `B`) from the train set like GSM8K. Each record is generally composed of a question field and an answer field.

2. Then use the policy model to generate a batch of answers (4, 8, or 16, as denoted by `G`) for each question. In GRPO, we call the answers generated from the same question as a group of training data.

Note that at the end of the batch sampling, we will get a tensor of token IDs (`input_ids`) with the shape of `(B*G, L+prompt_len)`, which is `B` groups of training data.

# Loss Function

Below is the reward function to optimize for GRPO, and the loss function is simply the minus of it. It may seem intimidating at the first sight, but we will see  that most of the effort in handcrafting GRPO is actually implementing the loss function.

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = 
\mathbb{E}\!\left[q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O|q)\right]
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
J_{i,t}(\theta)
$$

where

$$
J_{i,t}(\theta) = \min \left[
\frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}
\hat{A}_{i,t},\;
\mathrm{clip}\!\left(
\frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})},
1 - \varepsilon,\;
1 + \varepsilon
\right)
\hat{A}_{i,t}
\right]
- \beta\, D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right]
$$

The notion $J_{i,t}(\theta)$ represents the per-token loss of the *t*-th generated token of the *i*-th answering sequence, while $\mathcal{J}_{\mathrm{GRPO}}(\theta)$ is the expected value of the averaged per-token loss.

To simplify the implementation, I am going to split $J_{i,t}(\theta)$ into the following 3 parts, implement them separately, then piece them together.

- Probability Ratio: $\frac{\pi_{\theta}(o_{i,t} \| q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \| q, o_{i,<t})}$
- Advantages: $\hat{A}_{i,t}$
- KL Penalty: $D_{\mathrm{KL}}\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right]$

The loss function is implemented in the `GRPOTrainer` class. And below is the initialization of `GRPOTrainer`.

```
class GRPOTrainer:    
    def __init__(self, config, dataset, device=torch.device('cuda:0')):
        self.config = config
        self.dataset = dataset
        self.device = device
```

## Probability Ratio

At the heart of GRPO is the probability ratio between the new policy and the old policy. In the following probability ratio formula below, $i$ represents the record number in a certain sampling group, and $t$ represents the step number (or the token index) during the generation, so that $o_{i,t}$ means that the specific token (denoted as $o_{i,t}$) is observed at the *t*-th step of the *i*-th record in a certain group, while $\pi_{\theta}$ and $\pi_{\theta_{\text{old}}}$ represents the policy model and the old model, respectively.

$$
r_{i,t} = \frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}
$$

Thus, the whole formula is just the probability ratio for generating the token $o_{i,t}$ between the policy model and the old model.

As this is the most important formula in handcrafted GRPO training, let's gain a better understanding with an example. Suppose the following is the 3rd answer ($i=3$) sampled from the question (*q*) in a given sampling group:

* Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
* Answer ($o_3$): Natalia sold 48/2 = 24 clips in May ...

Thus, we can observe the tokens of $o_{3,t}$ at different generation steps (*t*) like the following:

* $o_{3,0}\Rightarrow$ *Natalia*
* $o_{3,1}\Rightarrow$ *sold*
* $o_{3,2}\Rightarrow$ *48*
* $o_{3,3}\Rightarrow$ */*
* $o_{3,4}\Rightarrow$ *2*
* $o_{3,5}\Rightarrow$ *=*
* $o_{3,6}\Rightarrow$ *24*
* $o_{3,7}\Rightarrow$ *clips*
* $o_{3,8}\Rightarrow$ *in*
* $o_{3,9}\Rightarrow$ *May*

Suppose the policy model generates *May* with a probability of 0.8, while the old model assigns 0.4. Then we will have $\pi_{\theta}(o_{3,9}=May)$ = 0.8 and $\pi_{\theta_{\text{old}}}(o_{3,9}=May)$ = 0.4. And thus $r_{3,9}$ = 0.8 / 0.4 = 2.

We can see from the narration above that the probability ratio $r_{i,t}$ is computed on a per-token basis. So we often call it per-token probability ratio.

The following Pytorch code can be used to compute the probability ratio, in which `input_ids` represents the token IDs of the questions as well as the generated answers, while `L` represents the net answer token length, excluding the question tokens.

```
def _compute_log_probs(model, input_ids, L):
    logits = model(input_ids).logits[:, -L-1:-1, :]
    per_token_log_prob = logits.log_softmax(dim=-1)
    per_token_log_prob = per_token_log_prob.gather(dim=-1, index=input_ids[:, -L:, None]).squeeze(-1)
    return per_token_log_prob
```

There are also 2 points to note in the above code:

* During the tensor computation, some probability value will overflow when too large. The common trick is to compute *log* of the probabilities, instead of the probabilities themselves.

* `logits.log_softmax()` returns the probability for each token in the vocabulary, so that its returning tensor shape is like (B*G, L, vocab_size). However, as we only care about the token appeared in the answer sequence, we can collect only the log probability for the tokens in the sequence using the
`gather()` function.

Generally speaking, we perform 3 ~ 6 rounds (specified by the parameter `mu`) of gradient decent iterations for each sampling group. In each interation, we compute a new `policy_log_prob` value. Before the iterations for the sampling group start, `old_log_prob` is computed with the same policy model, and it will be kept still during the whole iterations. By this means, we avoid introducing a 3rd model for computing `old_log_prob`.

The probability ratio is computed in each iteration. When computing ratio, we convert the log probability back to probability at the same time.
```
ratio = torch.exp(policy_log_probs - old_log_probs)
```

## Rewards and Advantages

The advantages in GRPO are based on the rewards received from the environment. For GSM8K, an easy approach is to set up some ground rules to evaluate the sentence generated by the model, and assign the reward according to the rules. The following is an example of the rules for GSM8K.

* If the generated text follows the correct format like `<think>...</think><answer>...</answer>`, award with 0.5 point.
* If the answer (the text between `<answer>...</answer>`) is an integer, further award with another 0.5 point.
* If the answer is correct, award with 1 more point.

After the rules are drafted, we can simply use Python to hard-code them. And during the training, we only need to invoke the Python code to evaluate the reward of each sentence generated by the policy model.

Rewards for a batch are stored in the tensor of `rewards`, whose shape is like (B, G). Each row of the tensor is a *group*, representing the rewards of different sampling answers generated by a question. That is to say, there are `B` (B = 4, 8, ..) questions selected for the batch, and for each question, `G` (G = 4, 8, 16, ..) answers are generated by the policy model. The tensor is shaped in this way for calculating the group-wise *mean* and *std*.

Then we can calculate the advantages of the generated answer with the formula below. We can see that it is just the relative reward ratio of each answer in its group, and how the name *Group Relative* comes from.

$$
A_i = \frac{r_i - \mu_G}{\sigma_G}
$$

Below is the Pytorch code to compute the advantages out of the rewards. As the tensor shape like (B, G) is only required when computing group-wise `mean` and `std`, the tensor is reshaped back to 1-D after the computation.

```
def _compute_rewards_and_advantages(self, txt_o, answer):
    B, G = self.config.B, self.config.G
    rewards = self._compute_rewards(txt_o, answer).view(B, G)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    adv = (rewards - mean) / std  # (B, G)
    
    return rewards, adv.view(B*G)
```

As for the advantages of each generated token in the answering text, GRPO simply sets all token-level advantages equal to their sequence-level advantage $A_{i}$, i.e.
$$
A_{i,t}=A_{i}
$$

## KL Penalty

The KL Penalty is used to prevent the fine-tuned policy model from going too far away from the original reference model. The problem is, how we can measure the KL Divergence between two models, as the model itself is a collection of parameters instead of the probability distributions.

The approach employed by GRPO is to compute the distribution divergence for the tokens being generated. Instead of using the KL Divergence directly, GRPO's implementation actually uses [the approximation of Reverse KL](http://joschu.net/blog/kl-approx.html) to estimate the KL divergence with the following formula.

$$
D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right] = x - \log x
- 1
$$

where

$$
x = \frac{\pi_{\mathrm{ref}}(o_{i,t} \mid q, o_{i,<t})}
{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}
$$

We can use the following Pytorch code to implement it. And the shape of `ref_log_probs` and `policy_log_probs` are both like (B*G, L).

```
def _compute_per_token_kl(ref_log_probs, policy_log_probs):
    x = ref_log_probs - policy_log_probs
    per_token_kl = torch.exp(x) - (x) - 1
    per_token_kl = torch.clip(per_token_kl, 0, 3)
    return per_token_kl
```

#### KL Approximation and Clipping

Instead of computing the full KL divergence across the vocabulary, GRPO approximates the reverse KL term for each generated token, as shown in the Pytorch code above. As our purpose is only to prevent the trained policy from deviating too far from the reference model, such approximation is sufficiently accurate.

However, during the training, we see some spikes for the $D_{\mathrm{KL}}$ values, which further lead to spikes in the loss function and potential reward collapse. A common approach to deal with the spikes in the loss function is to clip the gradient with the following Pytorch function.

```
torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
```

## Put Everything Together

Having all of the code snippets above, we can now define the function to compute the loss as follows.

```
def _compute_loss(self, input_ids, L, old_log_probs, ref_log_probs, advantages, completion_mask):
    policy_log_probs = GRPOTrainer._compute_log_probs(policy_model, input_ids, L)
    input_ids = input_ids[:, -L:]

    clip_epsilon = self.config.clip_epsilon
    beta = self.config.beta
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)

    per_token_kl = GRPOTrainer._compute_per_token_kl(ref_log_probs, policy_log_probs)
    per_token_adv = torch.min(ratio * advantages[:, None], clipped_ratio * advantages[:, None]) - beta * per_token_kl
    total_adv = ((per_token_adv * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return -total_adv
```

The code is straightforward. And the only confusing thing might be the `completion_mask` tensor, which is a boolean tensor used to mask out the tailing `<|eot|>` tokens.

# Training Flow

We have presented the high-level procedure of GRPO-based model fine-tuning in the section of *Procedure of LLM Fine-Tuning with GRPO*, which is quite similar to the general Pytorch model training approach. Now let's implement it in Pytorch.

First of all, define an optimizer like the following:

```
optim = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)
```

Next, we start the outer training loop. In each iteration of the outer loop, we sample `B` records (questions) from the train set, and use the policy model to generate `G` answers for each question. The sampled questions and answers are stored in the tensor of `input_ids`, whose shape is like `(B*G, L+prompt_len)`. And since we only care about the tokens before the end-of-token (`<|eot_id|>`), `first_eots` is used to record the first occurance of `<|eot_id|>`.

```
input_ids, ans, prompt_len = dataset.samples()
L = input_ids.size(1) - prompt_len
first_eots = dataset.get_first_eots(input_ids, prompt_len)
```

We can then convert the tokens of the answering texts back to their text representations, and use `_compute_rewards_and_advantages()` to calculate the rewards and advantages.

```
input_txt = [tokenizer.decode(out_tok[:first_eots[i]])
             for i, out_tok in enumerate(input_ids[:, prompt_len:])]
rewards, advantages = self._compute_rewards_and_advantages(input_txt, ans)
completion_mask = torch.arange(L, device=self.device).expand(input_ids.size(0), L) <= first_eots[:, None]  # (B*G, L)
```

And, of course, the probabilities that the sampling tokens are generated in the old model, i.e. $\pi_{\theta_{\text{old}}}(o_{i,t})$, and the reference model, i.e. $\pi_{ref}(o_{i,t})$. Be cautious that no gradients are required here, as we are not going to fine-tune the old model and the reference model.

```
with torch.no_grad():
    old_log_probs = GRPOTrainer._compute_log_probs(policy_model, input_ids, L)
    ref_log_probs = GRPOTrainer._compute_log_probs(ref_model, input_ids, L)
```

Note that the computation of `old_log_prob` and `ref_log_prob` are during the outer loop. That's true. Because they are used as the baseline to compute the probability ratio and the KL Penalties with `policy_log_prob`.

Finally, we enter the inner loop, and optimize the policy model. This loop is quite standard for Pytorch, but make sure to put `torch.nn.utils.clip_grad_norm_()` into the loop, especially we are using $D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right]$ as a KL approximation.

```
policy_model.train()
for grpo_iter in range(config.mu):
    optim.zero_grad()

    loss = self._compute_loss(
        input_ids, L,
        old_log_probs, ref_log_probs,
        advantages, completion_mask
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        policy_model.parameters(),
        config.max_grad_norm
    )
    optim.step()
```

However, the training code above is the most basic version, and doesn't include any optimizations. For example, you might feel that your GPU memory is not enough to compute the batch as large as `B*G`. If so, just add gradient accumulation into the code.

Here are the diagrams on the rewards, loss, and KL divergence during one of my GRPO training processes using GSM8K.

![Handcraft GRPO](/assets/handcraft-grpo.jpg)