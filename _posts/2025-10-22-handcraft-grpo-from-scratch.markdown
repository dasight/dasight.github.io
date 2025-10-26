---
layout: post
title:  "Handcrafting GRPO from Scratch"
date:   2025-10-22 22:58:09 +0800
categories: jekyll update
usemathjax: true
---
GRPO (Group Relative Policy Optimization) is a method used in reinforcement learning (RL)
to help a model learn better by comparing different actions and making small, controlled 
updates using a group of observations. It’s like a smart way to learn from experience without 
making drastic changes that could mess things up.

Handcrafting GRPO is especially useful to the projects with strict constraints where off-the-shelf 
solutions cannot express required regularizers or safety checks, or the research explorations that 
require new reward functions or update rules. It is also useful for those who wish to dive deeply
into the underlying mechanisms of GRPO.

We are going to handcraft GRPO from scratch rather than adapting an existing library or recipe.
That means, we will code the dataset pipeline, loss functions, KL/regularization, clipping, etc.
So you are interested in looking into the inside of GRPO, this blog is right for you.

## Modelling LLM Fine-tuning as Reinforcement Learning

Both PPO and GRPO are reinforcement learning approaches. A common question about this approach is how we can link LLM fine-tuning with reinforcement learning.

For this question, we can think that the generation of a token can be regarded as an action performed by LLM. And when the language model completes its generation (i.e. a sentense), the reward model, or some rules will be used to evaluate whether the generated sentence is good or not, and grant the reward according to the evaluation result. At the end of the step, the model is updated according to the reward received.

# Procedure of LLM Fine-Tuning with GRPO

When fine-tuning with GRPO, there are 3 models:

* Policy model (or new model), denoted as `policy_model` in the program code, is the trainable model. It is the one we are fine-tuning with GRPO updates, and represents the current policy $\pi_{\theta}$.
* Old model, denoted as `old_model`, is a frozen copy of the policy from the previous update step $\pi_{\theta_{\text{old}}}$. It’s used to compute the probability ratio.
* Reference model, generally denoted as `ref_model`, is the original pretrained model (before any fine-tuning).
It provides a stable distribution $\pi_{\text{ref}}$ against which we measure KL divergence.

As these are all very large models, we often try to reduce the number of the models. In my program code below, I combine the policy model and old model into one, and sampling as policy model or old model at different times.

The fine-tuning process of GRPO is composed of the following steps:

1. First of all, sample a batch of records (4 or 8, as denoted by `B`) from the train set like [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Each record is generally composed of a question field and an answer field.

2. Then use the policy model to generate a batch of answers (4, 8, or 16, as denoted by `G`) for each question. At this time, we get a tensor of token IDs (denoted as `input_ids`) with the shape of `(B*G, L+prompt_len)`, in which `L` is the maximum length the answer sequences, and prompt_len is the maximum length of the question sequences.

3. Calculate the loss function using `input_ids`.

4. Update the policy model parameter with `loss.backward()` and `optimizer.step()`.

5. For each sampled `input_ids`, repeat Step 2 ~ 4 for several times, as specified by `mu`.

6. Repeat the whole process (Step 1 ~ 5) for a few epoches (typically 1 or 2 epoches). One epoch means all of the records in the dataset are passwd once.

Based the steps explained above, we can see that the GRPO training process is primary composed of 2 parts:

* Data curation, or sampling different answers from the question. In our code, it is the `GRPODataSet` class.
* Training flow, which includes the implementation of the loss function. And in our code, it is the `GRPOTrain` class.

## Inner and Outer Loop during Fine-Tuning Procedure

You may have noted that there are an inner loop (Step 5) and an outer loop (Step 6) during the GRPO fine-tuning. The inner training loop uses the same question and answer samples, and repeated by several times controlled by `mu`. The outer training loop, in the other hand, needs to resample the question and answer tokens every time.

Now, let's go through them one by one.

# How the Training Data are Created?

We can build the training data for GRPO from a public data set like [GSM8K](https://huggingface.co/datasets/openai/gsm8k).

## Should we use the ground truth during the training?



# Loss Function

Below is the loss function to optimize for GRPO, which looks to be a little scary. As a matter of fact, most of the efforts in handcrafting GRPO are to implement the loss function. As for the detailed explanation of the loss function, we will address in the respective sections that follow.

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = 
\mathbb{E}\!\left[q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O|q)\right]
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} 
\left\{
\min \left[
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
\right\}
$$

To make life easier, I am going to split the function into 3 parts, implement them separately, then piece them together.

The 3 parts are:

- Probability Ratio: $\frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}$
- Advantages: $\hat{A}_{i,t}$
- KL Penalty: $D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right]$

The loss function is implemented in the `GRPOTrain` class. And below is the initialization of `GRPOTrain`.

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

As this is the most important formula for handcrafting the GRPO training program. Let's get better understanding with an example. Suppose the following is the 3rd answer ($i=3$) sampled from the question (*q*) in the certain sampling group:

* Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
* Answer ($o_3$): Natalia sold 48/2 = 24 clips in May ...

Thus, we can obverve the tokens of $o_{3,t}$ for different *t* like the following:

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

Suppose the policy model generates *May* with the propability of 0.8, while the old model generates with 0.4, we will have $\pi_{\theta}(o_{3,9}=May)$ = 0.8 and $\pi_{\theta_{\text{old}}}(o_{3,9}=May)$ = 0.4. And thus $r_{3,9}$ = 0.8 / 0.4 = 2.

We can see the narratation above that the probability ratio $r_{i,t}$ is computed on per-token basis. And we often call it per-token probability ratio.

The following Pytorch code can be used to compute the probability ratio, in which `input_ids` represents the token IDs of the questions as well as the generated answers, while `L` represents the net answer token length, excluding the question tokens.

```
def _compute_log_probs(model, input_ids, L)
    logits = policy_model(input_ids).logits[:, -L-1:-1, :]
    policy_log_prob = logits.log_softmax(dim=-1)
    policy_log_prob = policy_log_prob.gather(dim=-1, index=input_ids[:, -L:, None]).squeeze(-1)
    return policy_log_prob
```

There are also 2 points to note in the above code:

* During the tensor computation, some probability value will overflow when too large. The common trick here is to compute *log* of the probabilities, instead of the probabilities themselves.

* `logits.log_softmax()` will return the probability for each token in the vocabulary, so that its returning tensor shape will be (B*G, L, vocab_size). However, as we only care about the token appeared in the answer sequence, we can collect only the log probability for the tokens in the sequence using the
`gather()` function.

Generally speaking, we will perform 3 ~ 6 rounds (specified by the parameter `mu`) of gradient decent iterations for each sampling group. In each interation, we will compute a new `policy_log_prob` value. Before the iterations for the sampling group start, `old_log_prob` is computed with the same policy model, and it will be kept still during the whole iterations. By this means, we avoid introducing a 3rd model for computing `old_log_prob`.

The probability ratio is computed in each iteration. When computing ratio, we convert the log probability back to probability at the same time.
```
ratio = torch.exp(policy_log_probs - old_log_probs)
```

## Rewards and Advantages

The Advantages in GRPO is based on the rewards received from the environment. For GSM8K, an easy approach is to set up some ground rules to evaluate the sentence generated by the model, and grant the reward according to the rules. The following is an example of the rules for GSM8K.

* If the generated text follows the correct format like `<think>...</think><answer>...</answer>`, reward with 0.5 point.
* If the answer (the text between `<answer>...</answer>`) is an integer, further reward with another 0.5 point.
* If the answer is correct, reward with 1 more point.

After the rules are drafted, we can simply use Python to hardcode them. And during the training, we only need to invoke the Python code to evaluate the reward of each sentence generated by the policy model.

The rewards for a batch of train set will be finally stored in the tensor of `rewards`, whose shape is like (B, G). Each row of the tensor is a *group*, representing the rewards of different sampling answers generated by a question. That is to say, there are `B` (B = 4, 8, ..) questions picked up for the batch, and for each question, `G` (G = 4, 8, 16, ..) answers are generted by the policy model. The tensor is shaped in this way for calculating the group-wise *mean* and *std*.

Then we can calculate the Advantages of the generated answer with the formula below. We can see that it is just the relative reward ratio of each answer in its group, and how the name *Group Relative* comes from.

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

As for the advantages of each generated token in the answering text, GRPO simply makes them all equals to $A_{i}$, i.e.
$$
A_{i,t}=A_{i}
$$

## KL Penalty

The KL Penalty is used to prevent the fine-tuned policy model from going too far away from the original reference model. The problem is, how we can measure the KL Divergence between two models, as the model itself is a collection parameters instead of the probability distributions.

The approach employed by GRPO is to compute the distribution divergence for the tokens to generate. Instead of using the KL Divergence directly, GRPO's implementation actually uses [the approximation of Reverse KL](http://joschu.net/blog/kl-approx.html) to estimate the KL divergence with the following formula.

$$
D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right] = x - \log x
- 1
$$
In which
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

### KL Approximation and Clipping

You may notice that when computing $D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right]$, we are simply using the quotient between the probability of the generated token $o_{i,t}$ in the above Pytorch code, instead of the probability distribution of the LLM's vocabulary at Step *t*. As our purpose is only to prevent the trained policy train from going too away from the reference model, such approximation is fairly enough.

However, during the training, we do see some spikes for the $D_{\mathrm{KL}}$ values, which further lead to the spikes of the loss function and reward collapse. A common approach to deal with the spikes in the loss function is to clip the gradient with the following Pytorch function.

```
torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
```

## Put Everything Together

Having all of the code snippets above, we can now get the function to compute the loss like the following.

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

The code should be largely straightforward for you. And the only confusing thing might be the `completion_mask` tensor, which is a boolean tensor used to mask out the tailing `eot` tokens.

# Training Flow

We have presented the high-level procedure of GRPO-based model fine-tuning in the section of *Procedure of LLM Fine-Tuning with GRPO*, which is quite similar to the general Pytorch model training approach. Now let's implement it in Pytorch.

First of all, let's define an optimizer like the following:

```
optim = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)
```

Next, we start the outer training loop. In each iteration of the outer loop, we sample `B` records (questions) from the train set, and use the policy model to generate `G` answers for each question. The sampled questions and answers are stored in the tensor of `input_ids`, whose shape is like `(B*G, L+prompt_len)`. And as we only care about the tokens before end-of-token (`eot`), `first_eots` is used to keep record of the first occurance of of `eot`.

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

And, of course, the probabilities that the sampling tokens are generated in the old model, i.e. $\pi_{\theta_{\text{old}}}(o_{i,t})$, and the reference model, i.e. $\pi_{ref}(o_{i,t})$. Be cautious that no gradients are required here, as we are not going to fine tune the old model and the reference model.

```
with torch.no_grad():
    old_log_probs = GRPOTrainer._compute_log_probs(policy_model, input_ids, L)
    ref_log_probs = GRPOTrainer._compute_log_probs(ref_model, input_ids, L)
```

You may have already noticed that the computation of `old_log_prob` and `ref_log_prob` are during the outer loop. That's true. Because they are used as the baseline to compute the probability ratio and the KL Penalties with `policy_log_prob`.

At last, we will enter the inner loop, and optimize the policy model. This loop is quite standard for Pytorch, but make sure to put `torch.nn.utils.clip_grad_norm_()` into the loop, especially we are using $D_{\mathrm{KL}}\!\left[\pi_{\theta} \,\|\, \pi_{\mathrm{ref}}\right]$ as the approximation as the KL Divergence.

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

However, the training code above is most basic one, and doesn't include any optimizations. For example, you might feel that your GPU memory is not enough to compute the batch as large as `B*G`. If so, just add the gradient accumulation into the code.