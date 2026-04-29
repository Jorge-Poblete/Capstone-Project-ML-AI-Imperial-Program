# Capstone-Project-ML-AI-Imperial-Program
Section 1 — Project Overview

The core purpose of this project is to simulate a real-world optimization scenario where the underlying function is unknown, expensive to query, and highly constrained. Instead of having direct access to gradients or the true analytical form of the function, I must iteratively propose inputs (“queries”) and receive scalar outputs that represent performance.

The overall goal is to design a strategy that can efficiently search for the global optimum of several unknown functions while operating under strict query limits. This mirrors real machine learning and engineering applications such as chemical process optimization, hyperparameter tuning, and automated experimental design—domains where evaluations are costly and decisions must be guided by uncertainty rather than full information.

Working through this project strengthens my intuition for probabilistic modelling, experiment design, and decision making under uncertainty—skills that translate directly into my professional interests in pricing analytics, ML engineering, and Industrial optimization.

Section 2 — Inputs and Outputs

Each week, my model submits a single query for each black-box function. A query consists of a vector of continuous values—typically within the interval—whose dimensionality depends on the specific function:
•	Function 1–2: 2D continuous inputs
•	Function 3: 3D input vector
•	Function 4–5: 4D input vector
•	Function 6: 5D input vector constrained to a simplex (values sum to 1)
•	Function 7: 6D hyperparameter configuration
•	Function 8: 8D continuous vector

The output is a single scalar performance value (e.g., yield, score, reward). All data—initial and cumulative—are stored as .npy arrays and loaded weekly.

Section 3 — Challenge Objectives

My primary objective in the BBO capstone project is to maximize the unknown function values for all eight functions, while:

1.	Submitting only one query per function each week
2.	Working with no knowledge of the underlying function structure
3.	Adapting to noise, non-linearity, multimodality, and dimensionality differences
4.	Avoiding premature convergence and maintaining efficient exploration
5.	Leveraging the surrogate model to generalize from limited observations

Additional constraints/limitations:
•	Project: As this is my first project, it was challenging to evaluate the eight functions efficiently and perform the appropriate technical monitoring. I spent a lot of time understanding how to structure the project conceptually and automate it in python.
•	Response delays: updates occur weekly
•	Variability: some functions exhibit high noise or heavy-tailed outputs
•	Dimensionality challenges: especially in 5D–8D cases

My goal is not only to achieve good scores, but also to develop a principled and defensible optimization strategy that generalizes across all function types.
Section 4 — Technical Approach

Across the first three submissions, my approach evolved substantially as I observed more data and refined my understanding of each function’s behavior.

Surrogate Models Used
I relied on multiple surrogate modeling techniques depending on the function:
•	Gaussian Processes (GPs): Used for Functions 1, 2, 5, 6 — especially effective for smooth, lower dimensional landscapes.
•	ARD-enabled GPs: Essential for detecting relevant dimensions in 4D–6D tasks.
•	Tree based methods (Random Forests): Used when GPs struggled in high noise or high dimensional settings (Functions 4, 7, 8).
•	TPE (Tree structured Parzen Estimator): Applied for highly multimodal and noisy functions such as Function 4 and Function 7.

Acquisition Functions
To balance exploration and exploitation, I experimented with several AFs:
•	Expected Improvement (EI): strong for local exploitation
•	Upper Confidence Bound (UCB): useful for global exploration
•	Thompson Sampling (TS): adds stochastic exploration
•	Hybrid mixtures:
o	EI + UCB
o	EI + TS
o	TPE implicit EI
These mixtures helped stabilize behaviour when a single AF became unreliable.

Exploration vs. Exploitation Strategy
My strategy became more adaptive over time:
•	For smooth and unimodal functions (e.g., Function 5), I increased exploitation using EI and q EI.
•	For noisy or high dimensional functions (e.g., Function 7 or 8), I shifted toward UCB, TS, and TPE.
•	For plateau like functions (e.g., Function 1), I emphasized global exploration using high kappa UCB.
•	For functions with structural constraints (e.g., Function 6 simplex), I used distributions like Dirichlet to generate valid candidates.

Pipeline Architecture

My end-to-end pipeline is modular and extensible:
Load Data  → Normalize → Train Surrogate → Generate Candidates → Score Candidates → Select Best → Save Results → Visualize → Weekly Report → Export to Simulator
Core Modules
• load_data() — loads initial/cumulative .npy inputs/outputs
• normalize_outputs() — standardizes targets and returns mean/std for de-normalization
• build_gp_ard(), train_gp() — constructs and fits GP with Matérn + WhiteKernel (ARD-enabled as needed)
• lhs_samples(), sample_box() — global LHS/Sobol candidates and local Trust-Region boxes (R1, R2, R3)
• expected_improvement(), rank_normalize() — AF primitives
• select_one_point() — combines AFs (e.g., EI, STD, random) with rank-normalization to choose 1 point
• plot_observed_data(), plot_surrogate_slices() — 3D scatter, parallel coordinates, mean/STD/EI slices
• save_points_txt(), save_weekly_report() — simulator export and weekly summaries
• run_week() — orchestrates the full weekly cycle
 
Reporting & Simulator Integration
For each week and function, the pipeline writes a textual report with GP metrics (LML, MAE, RMSE), kernel parameters, candidate statistics (e.g., EI max, average STD), and the selected point.
The chosen input is exported for direct submission.
Why This Approach Is Thoughtful and Professional
The approach is modular, reproducible, and adaptive. It dynamically adjusts surrogates, kernels, AF weights, and candidate sources in response to empirical evidence (noise, uncertainty, dimensionality). It also balances global coverage with local refinement through multi-scale regions (global, meso, local, TR boxes), preventing early lock-in while enabling fine-grained exploitation when the signal is strong.

