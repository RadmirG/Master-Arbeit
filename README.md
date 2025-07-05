# Mastearbeit

The current repository consists the practical and theoretical parts of my master thesis, which wears the name of:

**Physics Informed Neural Networks (PINN) zur Lösung eines inversen Problems der
Wärmeleitungsgleichung** 

whts is nothing as

**Physics Informed Neural Networks (PINN) for solving an inverse heat conduction problem**.

This thesis investigates the inverse problem of heat conduction for determining the spatially 
dependent thermal conductivity. The theoretical foundation is provided by the partial differential 
equation of heat conduction. The objective is to reconstruct the local thermal conductivity from 
temperature measurements induced by a heat source. Since the determination of thermal conductivity 
using classical numerical methods is highly challenging, a modern approach from machine learning is 
pursued: Physics-Informed Neural Networks (PINNs). By incorporating physical laws into the training 
of the PINNs, it is demonstrated that this approach yields acceptable results. The overall outcome 
of this work is a software framework that can be easily applied to four variants of the problem. 
Both stationary and time-dependent inverse heat conduction problems are considered, up to the second 
spatial dimension.

The main part of practical work is placed in the folder 
[Master-Arbeit/code/InverseHeatSolver/](https://github.com/RadmirG/Master-Arbeit/tree/master/code/InverseHeatSolver). 
If You like to see the evaluation of this work plese see the MA_RadmirGesler.pdf.

---

### Problem Description

The center of view is the heat equation PDE defined on $`(\Omega \subset \mathbb{R}^n) \times (T \subset \mathbb{R})`$ 
space-time range in the following form:

$$ \frac{\partial u}{\partial t} - \nabla \cdot (a\nabla u) = f, $$

where $`u, f : \Omega \times T \rightarrow \mathbb{R}`$ and $`a:\mathbb{R} \rightarrow \mathbb{R}`$ are continues functions.

The goal is to infer the unknown thermal diffusivity $`a`$ from observed temperature data $`u`$ and the heat source $`f`$ .

---

## InverseHeatSolver

The [InverseHeatSolver](https://github.com/RadmirG/Master-Arbeit/tree/master/code/InverseHeatSolver/solver) 
caprtures the problem defined above throth the extension of a scintific deep-learning software library **DeepXDE**.

## Project Structure

```text
|-- FEniCS_scripts
    |-- direct_1D_fenics.eps
    |-- FEniCS_1D_td.py
    |-- FEniCS_1D_ti.py
    |-- FEniCS_2D_td.py
    |-- FEniCS_Dockerfile
    |-- ForwardHeatSolver.py
    |-- test_cases_FEniCS.py
|-- solver
    |-- History.py
    |-- Interpolator.py
    |-- InverseHeatSolver.py
    |-- ModelComposer.py
    |-- PdeMinimizer.py
    |-- PdeMinimizerDeepXde.py
    |-- Visualizer.py
|-- cases_and_plots.ipynb
|-- dde_forward_1D_ti.py
|-- dde_forward_2D_ti.py
|-- functions.py
|-- requirements.txt
|-- use_cases.py
|-- use_cases_deep_xde.py
