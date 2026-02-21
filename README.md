# Disease Modeling Portfolio

A collection of computational epidemiology projects spanning compartmental ODE-based models, agent-based models (ABMs), and behavioral-economic frameworks. Each project was developed with an emphasis on scientific rigor, and reproducible methodology.

> **Individual project write-ups** — including motivation, methods, results, and interpretation — are available on my [portfolio website](https://vscerra.github.io). This repository houses the code, notebooks, and data underlying each project.

---

## Projects

### 1. HIV/AIDS Epidemic Modeling & Control in Cuba
**`notebooks/` | `src/` | `data/`**

A compartmental ODE model fit to 15 years of empirical HIV/AIDS surveillance data from Cuba (1986–2000), based on the non-linear contact-tracing framework of De Arazoza & Lounes (2002). The model captures HIV progression through diagnosed and undiagnosed compartments, AIDS development, and AIDS-related mortality, with differential evolution used for global parameter estimation.

Key contributions:
- Multi-stage compartmental model (Susceptible → HIV⁻/HIV⁺ undiagnosed → HIV⁺ diagnosed → AIDS → Death)
- Fit to three simultaneous time series (HIV cases, AIDS cases, AIDS deaths) via differential evolution
- Contact-tracing control mechanism modeled and compared against observed post-1997 data
- Sensitivity analysis on transmission rate β and tracing efficacy

**Techniques:** SciPy ODE integration, differential evolution, parameter uncertainty, epidemiological control analysis

---

### 2. Yellow Fever Outbreak Modeling — Senegal 2002
**`notebooks/` | `src/` | `data/`**

A multi-stage compartmental model of the 2002 Yellow Fever outbreak in Touba, Senegal (pop. ~800,000). A vaccination compartment was added to model the October 2002 emergency vaccination program.

Key contributions:
- Fit to WHO surveillance data (cases and deaths) via MLE optimization
- Counterfactual analysis estimating outbreak trajectory without the vaccination intervention
- R₀ estimation and endemic equilibrium analysis — assessing whether YF could persist in the urban cycle
- Quantification of averted cases and vaccine effectiveness under the observed campaign

**Techniques:** Multi-stage ODE systems, time-varying vaccination rate, MLE parameter estimation, counterfactual simulation, endemic stability analysis

---

### 3. School-Based Measles Outbreak — Agent-Based Model
**`notebooks/` | `src/` | `data/`**

A stochastic agent-based model of measles transmission within a structured school environment (classrooms → grades → school), designed to capture the spatial clustering and heterogeneous contact patterns that compartmental models cannot represent. Agents move through SEIR states with biologically calibrated parameters (R₀ = 12–18, latent period ~9 days, infectious period ~8 days).

Key contributions:
- Hierarchical contact network reflecting real-world school structure
- Stochastic fadeout vs. sustained outbreak dynamics as a function of vaccination coverage
- Demonstration of local vs. global herd immunity thresholds
- Animated outbreak visualizations at the classroom level
- Intervention scenarios: uniform vaccination, ring vaccination, school closure timing

**Techniques:** Mesa (Python ABM framework), NetworkX, stochastic simulation, Monte Carlo ensembles, spatial visualization

---

### 4. Vaccination Economics & Behavioral Decision-Making
**`notebooks/` | `src/` | `data/`**

An SEIR-V epidemic model with an integrated behavioral layer grounded in game theory and the WHO SAGE vaccine hesitancy framework (the "3 C's": Confidence, Complacency, Convenience). Individuals make vaccination decisions by comparing expected utilities — incorporating perceived infection risk, disease severity, vaccine costs, and social influence — creating a dynamic feedback loop between behavior and transmission.

Key contributions:
- Deterministic SEIR-V model with age structure and two-dose vaccination schedule
- Individual utility maximization framework (Bauch & Earn 2004; Galvani et al. 2007)
- Nash equilibrium vs. herd immunity threshold gap — quantifying the free-rider problem
- Cost-effectiveness analysis: cost per DALY averted across five policy intervention scenarios
- Parameter sweeps across the 3 C's space; sensitivity analysis on behavioral parameters

**Techniques:** Coupled ODE systems, game-theoretic decision modeling, DALY/ICER calculation, policy comparison, Latin hypercube sampling

---

## Repository Structure

```
disease_modeling_project/
│
├── hiv_cuba/                
│   ├── notebooks/          # analysis notebook
│   ├── src/                # modular source code for ODE system, parameter estimation, control mechanisms
│   └── data/                
│
├── measles_abm/                     
│   ├── notebooks/          # analysis notebooks  
│   ├── src/                # modular source code for agent classes, school network, simulation runner, visualizations
│   └── data/    
|
```

Each notebook is self-contained and calls modular functions from `src/`. You can run any notebook independently.

---

## Getting Started

### Requirements

```bash
pip install numpy scipy pandas matplotlib seaborn mesa networkx
```

### Running a Project

Clone the repo and launch Jupyter:

```bash
git clone https://github.com/vscerra/disease_modeling_project.git
cd disease_modeling_project
jupyter notebook
```

Open any notebook in `notebooks/` and run top to bottom.

---

## Key References

- See [References](https://vscerra.github.io/disease_modeling/references) 

---

## About

These projects were developed as part of a computational epidemiology portfolio, reflecting a transition from neuroscience research into disease modeling and public health data science. The modeling approaches span the main paradigms in mathematical epidemiology — compartmental ODEs, stochastic ABMs, and behaviorally-coupled dynamic models — and are intended to demonstrate both technical depth and scientific communication.

**Portfolio:** [vscerra.github.io](https://vscerra.github.io) | **GitHub:** [@vscerra](https://github.com/vscerra)
