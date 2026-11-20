# MORP-DRL: Multi-Objective Reliability-Based Portfolio Optimization

[![Stage](https://img.shields.io/badge/Stage-Paper%20Breakdown-blueviolet.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![RL](https://img.shields.io/badge/RL-PPO-orange.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2607.06610-b31b1b.svg)](https://arxiv.org/abs/2607.06610)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Current Project Stage:** Active research & theoretical deconstruction. Right now, I am diving deep into the mathematical foundations of the paper (risk measures, Student's t-copula, QMC scenario generation, and PPO formulation) before transitioning into full-scale Python implementation and backtesting.

A quantitative portfolio management framework combining **Deep Reinforcement Learning (PPO)**, **Reliability-Based Design Optimization (RBDO)**, and **tail-risk modeling** under realistic market frictions.

## Overview

Traditional mean-variance portfolio models rely on Gaussian return assumptions, linear correlations, and frictionless trading. During market stress, asset correlations spike and symmetric risk metrics (like variance) fail to protect against extreme drawdowns.

This repository is dedicated to studying, reproducing, and backtesting an adaptive portfolio rebalancing engine that jointly optimizes expected return and downside risk subject to probabilistic reliability constraints and proportional transaction costs.

### Core Concepts & Features

* **Market & Tail-Dependence Modeling:** Captures volatility clustering via univariate `GARCH(1,1)` processes, heavy tails via Extreme Value Theory (EVT), and nonlinear crisis co-movements using a **Student's t-copula**.
* **Quasi-Monte Carlo (QMC) Scenario Engine:** Generates low-discrepancy stress scenarios (using Sobol sequences) to evaluate probabilistic portfolio feasibility ($\Pr[\text{constraint}] \ge \beta$).
* **Three Complementary Risk Formulations:**
  * **Model A (Variance):** Classical symmetric risk baseline.
  * **Model B (CVaR):** Coherent downside risk targeting expected losses beyond the $\text{VaR}_\alpha$ threshold.
  * **Model C (EVaR):** Entropic Value-at-Risk providing a tighter exponential upper bound on extreme tail losses.
* **Adaptive Rebalancing Agent:** Trains a **Proximal Policy Optimization (PPO)** policy in a custom market environment accounting for proportional transaction costs and allocation bounds, benchmarked against `NSGA-II` and equal-weight ($1/N$) strategies.

## Mathematical Formulation

At each rebalancing step, the agent determines target portfolio weights $\mathbf{w} = (w_1, \dots, w_N)$ to solve the bi-objective problem:

$$
\max_{\mathbf{w}} \sum_{i=1}^{N} w_i \bar{r}_i \quad \text{and} \quad \min_{\mathbf{w}} \rho_\alpha(\mathbf{w})
$$

Subject to probabilistic reliability constraints and market frictions:

**1. Net Return Reliability Constraint** (accounting for proportional transaction costs $k_i$):

$$
\Pr\left( \sum_{i=1}^{N} w_i r_{i,t} - \sum_{i=1}^{N} k_i |w_i - w_i^0| \ge r_p^* \right) \ge \beta_1
$$

**2. Downside Risk Reliability Constraint:**

$$
\Pr\left( \rho_\alpha(\mathbf{w}) \le \rho^* \right) \ge \beta_2
$$

**3. Budget & Allocation Bounds:**

$$
\sum_{i=1}^{N} w_i = 1, \quad 0 \le w_{i,\min} \le w_i \le w_{i,\max} \le 1
$$

where:
* $\rho_\alpha(\cdot) \in \{\text{Variance}, \text{CVaR}_\alpha, \text{EVaR}_\alpha\}$ — selected risk measure,
* $w_i^0$ — initial portfolio weights before rebalancing,
* $\beta_1, \beta_2 \in (0, 1)$ — prescribed reliability confidence levels.



## Planned Architecture

```text
├── data/               # Market data loaders (yfinance) & log-return pipelines
├── scenarios/          # GARCH(1,1), EVT (GPD), t-copula fitting & QMC simulation
├── models/             # Risk evaluators (Variance, CVaR, EVaR) & RBDO constraints
├── env/                # Custom Gymnasium environment with transaction cost penalties
├── agents/             # PPO Actor-Critic agent & NSGA-II multi-objective baseline
├── backtest/           # Out-of-sample regime evaluation (Pre/During/Post-COVID)
└── notebooks/          # Mathematical notes, exploratory analysis & Pareto frontiers
```

## Study & Development Roadmap

### Phase 1: Paper Breakdown & Math Foundations *(Current Stage)*
* [x] Problem motivation & coherent risk measures ($\text{VaR}$, $\text{CVaR}$, $\text{EVaR}$)
* [x] Reliability-Based Design Optimization (RBDO) & Model A / Model B formulations
* [ ] Model C ($\text{EVaR}$) formulation & MDP / PPO reward function design
* [ ] Time-decomposed QMC + Student's t-copula scalability mechanics

### Phase 2: Implementation & Prototyping
* [ ] Historical data pipeline & `GARCH(1,1)` + EVT scenario generator
* [ ] Custom Gymnasium portfolio environment with proportional transaction costs
* [ ] PPO agent training loop & `NSGA-II` benchmark baseline

### Phase 3: Validation & Real-World Testing
* [ ] Cross-regime evaluation (Pre-COVID, COVID, Post-COVID) & Pareto frontier plots
* [ ] Backtesting on a custom real-world asset portfolio

## Reference

* **Paper:** *Deep Reinforcement Learning for Reliability Based Bi-Objective Portfolio Optimization*
* **Authors:** Sounaq Das (IIM Amritsar), Raghu Nandan Sengupta (IIT Kanpur), Tanmay Sen (ISI Kolkata), Aditya Gupta (McKinsey & Company)
* **Preprint:** [arXiv:2607.06610 [cs.LG]](https://arxiv.org/abs/2607.06610) (July 2026)
