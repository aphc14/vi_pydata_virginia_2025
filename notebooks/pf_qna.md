**Q: What is $log p$ or $\log p(\theta)$?** \
It is the log of a tractable log density function. We use the joint log density, $\log p(y | \theta) + \log p(\theta)$ of the model as the tractable log density function.

**Q: Can this be scalable using distributed computing?** \
Great question! The answer is yes! Because the Multi-path Pathfinder algorithm can run the Single-path Pathfinder independently on different parts of the parameter space, it is possible to parallelize the algorithm across multiple cores or machines.

Please do reach out once you have tried this using cloud distributed computing to perform large scale inference. We are curious to know how it goes!


**Q: What are the recommended settings for...** \
Pathfinder can also be sensitive to the algorithm settings. This could be a pro instead of con though, because it can provide much more degree of control over the posterior approximation compared to ADVI. After your model and posterior evaluation, and wanting to improve the model fit (assuming you have a well parameterised model), you can rerun Pathfinder inference with different settings:

- `jitter`, `num_paths` and `maxcor` (highly sensitive to change)
- `ftol`, `gtol`, `num_elbo_draws` (less sensitive to change)

**Q: What are the recommended settings for... (continued)** \

**- Q: the number of paths (`num_paths`)?** \
  A: It depends on the complexity of the posterior space and the jitter scale (`jitter`). If the posterior space is complex, multimodal, and has several local modes which causes the optimization trajectory to be stuck in a local mode, then the jitter scale and number of paths should be increased to enable the posterior space to be explored more.

**- Q: jitter scale (`jitter`)? Q: initial values (influenced by `jitter`)?** \
  See the number of paths (`num_paths`) section. It is recommended that the initial points are NOT concentrated in a small region of the parameter space and NOT at the tail of the posterior distribution. 
  
  A rough idea of the size of the parameter space should be considered when setting the jitter scale, which is dependent on the model, data, and prior.

  Initial points should be at the tail of the posterior distribution as this would allow for the L-BFGS optimization trajectory to properly explore the posterior shape and remain in a concentrated region of the parameter space.
  
  And if the posterior distribution is multimodal, then the initial points should be at the tail of the local modes. 
  
  If the initial points are at the extreme tail of the posterior distribution, then it is possible for L-BFGS to not converge (unlikely if `maxcor` is set to a large number) or provide bad samples from that particular single-path Pathfinder.
  
  In practice, knowing the tail regions or good tail regions can be difficult. So, testing out different settings for jitter and number of paths may be necessary. Importance sampling is there to exclude bad samples from the single-path Pathfinder. And by not using resampling approach through `importance_sampling="psis"`, it reduces the risk of having over-saturated samples that are concentrated at the peak of the posterior distribution.




**- Q: history size (`maxcor`)?** \
  Generally, the higher the history size, the more improvement would be seen in the model fit. However, the improvement can be diminishing as you increase the history size. It's also possible in some scenarios where a pathfinder might struggle to adapt to the local geometry/curvature with high history sizes.

  From the paper:
  > We encourage a larger J [history size] for approximation when the target distribution is expected to have high dependencies among a large number of parameters.

  So, the history size can be increased if needed if the model parameters are highly correlated.
  
  The computational time goes up significantly as you increase history size though. To balance out between performance and compute time, we have chosen to use the default `maxcor` to be `ceil(3 * log(N))` or `5`, whichever is greater. 

**- Q: the number of samples per path (`num_draws_per_path`)?** \

**- Q: the number of Monte Carlo samples (`num_elbo_draws`)?** \

**- Q: the number of samples overall (`num_draws`)?** \

**- Q: importance sampling (`importance_sampling`)?** \

**- Q: maximum L-BFGS iterations (`maxiter`)?** \

**- Q: maximum number of function evaluations (`maxls`)?** \


**Q: Which Pathfinder settings affects the computational time the most?** \
The history size (`maxcor`), number of paths (`num_paths`), and the number of samples per path (`num_draws_per_path`). 


