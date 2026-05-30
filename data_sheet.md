# Datasheet Capstone Project

## Function overview

1. Which function does this datasheet describe?

Function X – Black-box optimisation task (dimensionality varies per case).

2. What real-world scenario does this function simulate?

This function simulates a real-world optimisation problem where the objective function is unknown, expensive to evaluate, and must be explored through sequential experimentation. Depending on the function, this can represent scenarios such as:
Chemical process optimisation (yield maximisation)
Industrial parameter tuning
Hyperparameter optimisation in machine learning

The goal is to efficiently discover the best-performing input configuration under uncertainty.

3. What is the dimensionality of the input?
Varies by function:

Function 1–2: 2D
Function 3: 3D
Function 4–5: 4D
Function 6: 5D (simplex constraint)
Function 7: 6D
Function 8: 8D

4. How many initial data points were provided?

A small set of initial points (Design of Experiments), typically limited and insufficient to fully describe the function.
Dataset size grows sequentially as one point is added per iteration.

5. What does the output represent?

A scalar performance metric, such as:

Yield
Quality score
Reward / objective value

The optimisation objective is to maximise this value.
 
## Nature of the data

1. Structure of the initial dataset
   
•	Input: matrix of shape(N, d)
•	Output: vector of shape(N, 1)
•	Typically small (few initial observations)
•	Stored in structured arrays (.npy, CSV)

2. Dataset evolution
   
•	One new data point added per week
•	Dataset grows sequentially (small → moderate size)
•	Early iterations: global coverage
•	Later iterations: highly concentrated local sampling

This leads to non-uniform coverage of the search space, intentionally focused on high-performing regions. 

3. Noise or randomness

Some functions exhibit noise or irregular behaviour, where similar inputs produce different outputs (e.g., Function 2, Function 4).
This affected the strategy by:
•	Requiring more robust models (e.g. Random Forest)
•	Increasing the need for controlled exploration

4. Function behaviour (observed)

Depending on the function, behaviour includes:

•	Unimodal, smooth (Function 5)
•	Highly localised peaks (Function 1, Function 6)
•	Multimodal with sparse islands (Function 4, Function 7, Function 8)
•	Irregular / non-smooth (Function 2)
These observations were based on:
•	Surrogate model behaviour
•	Lack or presence of improvement
•	Sensitivity to small input changes

## Your optimisation strategy

1. Optimisation methods used
   
•	Bayesian Optimisation 
  o	Gaussian Process + EI/UCB
  o	Random Forest + EI
  o	TPE (Tree Parzen Estimator)

•	Trust-region methods (TuRBO)
•	Local search (forced EI)

2. Why this method?

Different methods were chosen depending on function characteristics:

•	GP → smooth functions
•	RF → irregular/noisy functions
•	TuRBO → multimodal/high-dimensional problems

The strategy was adaptive based on empirical performance, not fixed assumptions.

3. Exploration vs exploitation balance

•	Early phase → exploration (UCB, global sampling)
•	Middle phase → mixed (EI + UCB / TS)
•	Late phase → exploitation (local EI, small radius)

4. Strategy evolution

•	From global BO → local optimisation
•	From GP-only → model switching (RF)
•	From exploration → validation

Changes were driven by:
•	Stagnation
•	Model mismatch
•	Discovery of high-performing regions

## Data handling and preprocessing

1. Input scaling
Inputs were normalised to ensure:
•	Stable surrogate training
•	Better numerical performance

2. Surrogate models
•	Gaussian Processes (Matérn kernel, ARD)
•	Random Forest (for irregular landscapes)

3. Preprocessing for surrogates 
•	Output normalisation
•	Kernel parameter tuning (lengthscale, noise)
•	Feature scaling

4. Outliers handling
•	No explicit removal
•	Instead: 
  o	Interpreted as signal (e.g. poor regions)
  o	Used to guide exploration/exploitation decisions

## Weekly iteration and learning

1. Learning over time
•	Early iterations → understanding global structure
•	Later iterations → refining local regions

2. Local optima detection
Detected through:
•	Repeated high values in same region
•	Lack of improvement despite exploration

3. Most informative inputs
•	Points near high-performing regions
•	Boundary points (Function 5)
•	High-uncertainty regions (early stages)

4. What I would do differently
•	Detect function type earlier (local vs global)
•	Switch strategies sooner
•	Automate model selection


## Performance and results

1. Best output achieved
Varies by function (examples):

•	Function 1 → Week 3
•	Function 2 → Week 9
•	Function 4 → Week 9
•	Function 5 → Week 4
•	Function 6 → Week 1
•	Function 7 → Week 8
•	Function 8 → Week 13

2. Best input vector
The specific input corresponding to best performance (stored in dataset).

3. Confidence in optimality
High confidence due to:
•	Stability across iterations
•	Lack of improvement after refinement
•	Low uncertainty in local regions

4. Alignment with expectations
Yes:
•	Smooth functions → early convergence
•	Complex functions → require exploration + local search


## Ethical, practical and general considerations

1. Real-world relevance

This task reflects real applications such as:
•	Industrial optimisation
•	Pricing strategies
•	Hyperparameter tuning

2. Limitations of synthetic setup
•	Simplified environment
•	Lower noise compared to real systems
•	Known evaluation budget

3. Scalability
Yes, but:
•	Requires automation (e.g. RL/meta-learning)
•	Needs parallel evaluation for real-world scale

4. Risks and pitfalls
•	Overfitting to local regions
•	Ignoring unexplored space
•	Misinterpreting stagnation
