# Adaptive Bayesian Black-Box Optimisation for Multi-Function
 
This project is about finding the best possible solutions when we don’t fully understand how a system works. Instead of having a clear formula, we can only test inputs and observe the results. I built a smart search strategy that learns from each step and adapts over time to make better decisions. The goal is to efficiently discover high-performing solutions while using very limited trials. This approach mirrors real-world problems like pricing, industrial optimisation, and machine learning tuning, where experiments are expensive, and decisions must be made with incomplete information.

## DATA

The dataset is generated through an iterative process where each week I submit a single input (query) for each of eight unknown functions and receive a corresponding output value. These functions vary in dimensionality and complexity, ranging from 2D to 8D input spaces. The data is stored as structured arrays (.npy files) and grows over time as new observations are added.
The project simulates real-world optimisation scenarios where:

•	The true function is unknown
•	Evaluations are expensive
•	Data is limited and sequential

Each function represents a different challenge, such as smooth behaviour (Function 5), irregular noisy patterns (Function 2), or highly localised optima (Function 1)

## MODEL 

The optimisation framework uses Bayesian Optimization with adaptive surrogate models. The main models used are:

•	Gaussian Processes (GPs): used for smooth, low-dimensional problems due to their strong uncertainty estimation
•	Random Forests: used when functions are non-smooth or highly irregular
•	TPE (Tree-structured Parzen Estimator): used for multimodal or noisy environments
I selected this hybrid modelling approach because no single model performs well across all problem types. For example:
•	GP performed well in Function 5 (smooth, unimodal)
•	Random Forest was critical for Function 2 (non-smooth)

The model choice was therefore adaptive, based on observed data behaviour rather than fixed assumptions.

## HYPERPARAMETER OPTIMSATION

Hyperparameters were not optimised through a fixed grid search, but instead adaptively tuned during the optimisation process, based on performance and feedback.

Key hyperparameters included:
•	Kernel parameters (length scale, noise) for GP models
•	Exploration parameters (ξ in EI, κ in UCB)
•	Trust-region size (for TuRBO/local search)
•	Candidate sampling density and radius

Optimisation strategy:
•	Early stage → higher exploration (large radius, higher κ)
•	Mid stage → balanced exploration and exploitation
•	Late stage → strong exploitation (small radius, low ξ)

For example:
•	In Function 1, reducing the search radius was critical to detect a narrow peak
•	In Function 4 and 7, adjusting trust-region size significantly improved results
•	In Function 5, collapsing EI to near zero indicated convergence and no further tuning was needed

This adaptive approach ensured efficient use of limited queries.


## RESULTS

The results demonstrate that different optimisation strategies are required depending on the structure of each function.

Key outcomes:

•	Function 1: optimum found early (Week 3) → highly local problem
•	Function 2: improved after switching model (Week 9) → irregular function
•	Function 3: best solution already in initial data → validation problem
•	Function 4 & 7: major improvements after using TuRBO (F2 week 9 & F7 week 8)→ multimodal functions
•	Function 5: fast convergence (Week 4) → smooth function with boundary optimum
•	Function 6: early optimum (Week 1), later degradation due to over-exploration
•	Function 8: late improvement (Week 13) → required extended refinement

Key insights:

•	Performance improvements occur in discrete jumps, not gradual trends
•	Local search dominates global search in most cases
•	Over-exploration after convergence degrades performance

Overall, the model demonstrates that adaptive strategies outperform static ones, and that understanding the problem structure is more important than algorithm complexity.
