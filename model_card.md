# Model Card- Adaptive Bayesian Black-Box Optimisation (BBO)

## Model Description

Input
The model takes as input a continuous vector of decision variables, where dimensionality depends on the function being optimised:
•	Function 1–2: 2D inputs
•	Function 3: 3D inputs
•	Function 4–5: 4D inputs
•	Function 6: 5D inputs (simplex constraint)
•	Function 7: 6D inputs
•	Function 8: 8D inputs
Each input represents a candidate solution evaluated by a black-box simulator. These inputs are generated sequentially based on previously observed data.

Output
The model outputs a single scalar value (objective score) representing the performance of each input configuration.
The goal of the model is to:
•	Maximise this output value
•	Identify the best-performing input across all evaluations
•	Provide a sequence of improvements over time (convergence)

Model Architecture
The model is not a single algorithm but an adaptive optimisation framework combining several components:
•	Surrogate models:
o	Gaussian Processes (Matérn / RBF) → for smooth problems
o	Random Forest → for irregular or non-smooth functions

•	Acquisition functions:
o	Expected Improvement (EI) → exploitation
o	Upper Confidence Bound (UCB) → exploration
o	Thompson Sampling (TS) → stochastic exploration

•	Search strategies:
o	Global Bayesian Optimisation (early stage)
o	Local optimisation (forced EI / local sampling)
o	Trust-region methods (TuRBO)

The architecture follows a phase-based logic:
1.	Exploration → understand the space
2.	Discovery → identify promising regions
3.	Refinement → local optimisation
4.	Validation → confirm convergence
This adaptive behaviour is the key strength of the model. 


## Performance

Evaluation metrics
Performance was evaluated using:
•	Best observed value (best-y)
•	Convergence behaviour over time
•	Stability of the optimum across iterations
•	Qualitative diagnostics (e.g. clustering, local behaviour)
 
Summary of results
Function	Best Week	Key insight
F1	Week 3	Narrow peak → local search critical
F2	Week 9	Irregular → model switch (RF) needed
F3	Week 0	Optimum already in initial data
F4	Week 9	Multimodal → TuRBO breakthrough
F5	Week 4	Smooth → rapid convergence
F6	Week 1	Early optimum, later degradation
F7	Week 8	Local optimisation required
F8	Week 13	Late discovery + refinement

Key performance insights
•	Improvements occur in discrete jumps, not continuously
•	Local optimisation dominates global exploration in most functions
•	Best results often occur early or after structural strategy changes
•	Performance depends more on strategy adaptation than model complexity

## Limitations

The model has several important limitations:
•	Dependence on early data:
Early samples strongly influence later decisions; misleading initial data can bias results

•	Limited global exploration in later stages:
Once local optimisation dominates, unexplored regions may be ignored

•	Sensitivity to function type:
A mismatch between model assumptions and function behaviour (e.g. GP in non-smooth functions) can delay convergence

•	Sequential evaluation constraint:
Only one query per iteration limits learning speed

•	Requires human interpretation:
Strategy adaptation is not fully automated and depends on manual decisions

## Trade-offs

The model involves several key trade-offs:

1. Exploration vs Exploitation
•	Too much exploration → slow convergence / noise
•	Too much exploitation → risk of missing global optima

Example:
•	Function 5 → over-exploration caused performance collapse
•	Function 4 → lack of exploration delayed discovery

2. Model complexity vs robustness
•	Complex models (RF + ensemble strategies) did not always improve results
•	Simpler approaches (GP + EI) often performed better

3. Global vs Local search
•	Global search → necessary early
•	Local search → dominant in later stages

Example:
•	Function 7 → global BO failed, TuRBO succeeded

4. Short-term vs long-term performance
•	Exploration may reduce short-term performance but enable later breakthroughs
•	Over-optimising locally may prevent discovering better regions
