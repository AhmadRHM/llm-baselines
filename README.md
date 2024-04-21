# Vita team submission

The code base containing our changes and added modules for the hackathon on LLM pretraining.

We tried several ideas, like mixture of experts, label smoothing for cross entropy, using different optimizers, regularizers, using QR decomposition, and playing with hyperparameters.

We will first share the setup for our best results, and then share the ideas we tried and what we learned from them.

## Our final results
The highest accuracy we could achieve on the validation set after 3 hours of training is ***0.4361***. Here is the command to train this model:
```
python3 ./src/main.py --config_format base --wandb --wandb_project llm-hack --model llama2 --moe_num_experts 4 --iterations 11000 --moe_num_experts_per_tok 2
```
This should work with the docker image provided for the hackathon, but in case there are some packages missing which are needed in the code, you could also use the image at `registry.rcp.epfl.ch/vita/lauzhack_llm`.

Unfortunately, we don't have access to the checkpoints, as they were overrided with later submissions with same setting! (We turned off the checkpoint existence check, as it was annoying for debug :D)

We would be very thankful if you could re-train the model for evaluation. You can access the [wandb report](https://wandb.ai/socialcausality/llm-hack/workspace?nw=nwuserahmadrahimi) to check our different experiments (the final submission's name is `llama2-moe-e4-ept2`).

Minor point, added after the deadline: We were also able to get higher accuracy with a similar model, which uses smoothing. It acheives ***0.4372***, but does so after 3 hours and 1 minute (!). So we didn't announce it as our final result, but if it's interesting, you could train it by adding `--label_smoothing 0.1` to the above command. The training logs of this run is also available at the wandb report under the name `llama2-moe-e4-ept2-smooth0.1`. 

## Ideas tried and lessons learned
We found it very hard to beat the gpt2 baseline provided with the original code, and we think the baseline is very strong (maybe a bit too strong to beat in the scope of a hackathon :D). 
We started by playing with some hyperparameters like the learning rate, size of the model (making it both smaller and larger), the warmup, and the batch size, but all of them led to worse results!
We had several ideas that we also shared in the presentation on Saturday. Here we will go over each of them, explaining them briefly, and sharing our take home message.

1. ***MoE:*** One of the ideas that we tried and were able to improve the baseline is mixture of experts. 
The idea is that in the feed forward part of the model, we use several copies of the MLP, each of them being an expert in certain areas.
We then use a router which chooses which experts to use for each token. There are several auxiliary loss functions, ensuring load balance for different experts.
Our idea was that using this architecture, we could increase the number of parameters of the model by adding more experts, hopefully leading to better modeling of more complex texts and patterns, without computational overhead.
However, after implementing this idea in the code, we found out that the base model's implementation is very optimized, unlike our MoE implementation.
Additionally, updating multiple experts at training time multiplies the training time. 
These two were the drawbacks of this approach which we think kept us from achieving substantial improvements. We also tried other off the shelf MoE models like ST-MoE, but they didn't turn out to work well either.
Nevertheless, our final approach uses MoE which shows some improvement over the baseline.
2. ***Label smoothing:*** The idea is that instead of having one hot target for training, we use a smoother one in the cross entropy. 
For example, if the smoothing value is 0.1, it means that the target probability for the cross entropy loss is 0.9 for the correct token (instead of 1), and the 0.1 is distributed uniformly for all other tokens. Label smoothing emerged as a key regularization technique in our work. Commonly utilized in deep learning models, it addresses the tendency of models trained with hard labels to exhibit overconfidence. By introducing noise through a slight reduction in the probability of the correct class and redistributing it across alternative classes, label smoothing fosters a more robust and generalizable model. 
We found that it is a bit helpful to have better results.

3. ***Shuffle Regularizer*** We delved into a new regularization approach for the attention mechanism by introducing noise to the query values. This method involved randomly shuffling a percentage of rows in the query matrix, with initial trials employing a 10% shuffle for both baseline and Llama-2 models. While this tactic occasionally yielded training accuracy boosts, the final performance after 3 hours was inferior to models without shuffling. Despite not integrating this method into our final submission, exploring varied shuffle percentages or extended training durations could offer promising avenues for further inquiry.

4. ***Different Attention Mechanisms*** Another point of our investigation was the exploration of alternative attention mechanisms. A local attention mechanism is implemented to assess its potential superiority over the standard approach, as well as experimented with combining global and local attention. These variations led to slower training compared to the standard mechanism. Additionally, we contemplated leveraging orthonormal matrices for query, key, and value to potentially extract richer features. However, the computational overhead proved prohibitive, prompting us to discontinue this line of exploration.

5. ***LAMP Optimizer*** Furthermore, the effect of different optimizers is investigated. Initially, the LAMP optimizer was applied, tailored specifically for Large Language Models (LLMs). However, LAMP's performance suffered with our batch size of 32, indicating its preference for larger batch sizes.
