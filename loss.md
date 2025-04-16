# 大模型训练过程中的loss

## 1.pretrain阶段的loss


$$
Input = <bos>, w_1, w_2...w_T 
$$

$$
Label=w_1,w_2...w_T,<eos>
$$

$$
Loss_{pretrain} = -\sum_{k=1}^{T+1} \log{P(w_t|<bos>, w_1, w_2...w_{t-1})}
$$

## 2.sft阶段的loss

$$
Input = <bos>, p_1, p_2...p_{L_p},r_1,r2...r_{L_t}
$$

$$
Label=\underbrace{[-100, -100...-100]}_{L_p+1个ignoreIndex},r_1,r2...r_{L_t}
$$

$$
Loss_{sft}=-\sum_{t=L_p+2}^{L_p+L_r+1}\log{P(w_t|Input_{<t})}
$$

## 3.rlfh阶段的loss

$$
Loss_{R}(R_{\phi})=-E_{(x,y_{win},y_{lose})\in{D}}[\log{{\sigma}((r_{\phi}(x, y_{win}) - r_{\phi}(x, y_{lose})))}]
$$

$$
Loss(\pi_{\theta})=-E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}[r_{\phi}(x,y)] + \beta D_{KL}[\pi_{\theta}(y|x) || \pi_{ref}(y|x)]
$$

## 4.dpo的loss

$$
Loss=\max_\limits{\pi_{\theta}}\{{E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}[r_{\phi}(x,y)] - \beta D_{KL}[\pi_{\theta}(y|x) || \pi_{ref}(y|x)]}\}
$$

$$
=\max_\limits{\pi_{\theta}}E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}[r_{\phi}(x,y) - \beta\log\frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)}]
$$

$$
=\min_\limits{\pi_{\theta}}E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}[\log\frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta}r_{\phi}(x,y)]
$$

$$
=\min_\limits{\pi_{\theta}}E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}\log\frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)e^{r_{\phi}(x,y)/\beta}}
$$

考虑归一化上式右边的分母:
$$
Z(x)=\sum_{y}\pi_{ref}(y|x)e^{r_{\phi}(x,y)/\beta}
$$
可以构造如下概率分布：
$$
\pi^{*}(y|x)=\pi_{ref}(y|x)e^{r_{\phi}(x,y)/\beta}/Z(x)
$$
于是：
$$
Loss=\min_\limits{\pi_{\theta}}E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}[\log\frac{\pi_{\theta}(y|x)}{\pi^{*}(y|x)}-logZ(x)]
$$

$$
=\min_\limits{\pi_{\theta}}E_{x\verb|~|D,y\verb|~|\pi_{\theta}(y|x)}\log\frac{\pi_{\theta}(y|x)}{\pi^{*}(y|x)}
$$

$$
=\min_\limits{\pi_{\theta}}E_{x\verb|~|D}D_{KL}\pi_{\theta}(y|x)||(\pi^{*}(y|x))
$$

KL散度在两个分布相等于最小