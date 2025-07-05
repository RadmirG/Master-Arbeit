# Master Thesis: Inverse Heat Conduction with PINNs

This repository contains the full implementation and theoretical background of my Master's thesis titled:

**Physics-Informed Neural Networks (PINNs) zur Lösung eines inversen Problems der Wärmeleitungsgleichung**  
or in English:  
**Physics-Informed Neural Networks (PINNs) for Solving an Inverse Heat Conduction Problem**

---

## Thesis Overview

This thesis addresses the inverse problem of heat conduction, aiming to reconstruct the **spatially dependent thermal conductivity** from temperature measurements influenced by a known heat source.

While classical numerical approaches struggle with the ill-posedness of the inverse problem, this work adopts a **modern scientific machine learning method**—**Physics-Informed Neural Networks (PINNs)**—which integrate physical laws directly into the learning process via the loss function. This approach demonstrates robust and interpretable results.

The main contribution is a modular and extensible **software framework** that supports four major variants of the problem:

- **1D and 2D domains**
- **Stationary and time-dependent** settings

The solver is built upon the [DeepXDE](https://github.com/lululxvi/deepxde) library and extended to handle inverse PDE problems for thermal diffusivity.

The implementation can be found in  
[`code/InverseHeatSolver/`](https://github.com/RadmirG/Master-Arbeit/tree/master/code/InverseHeatSolver)

For detailed evaluation and results, refer to the thesis document: `MA_RadmirGesler.pdf`.

---

### Problem Description

The center of view is the heat equation PDE defined on $`(\Omega \subset \mathbb{R}^n) \times (T \subset \mathbb{R})`$ 
space-time range in the following form:

$$ \frac{\partial u}{\partial t} - \nabla \cdot (a\nabla u) = f, $$

where $`u, f : \Omega \times T \rightarrow \mathbb{R}`$ and $`a:\mathbb{R} \rightarrow \mathbb{R}`$ are continues functions.

The goal is to infer the unknown thermal diffusivity $`a`$ from observed temperature data $`u`$ and the heat source $`f`$ .

---

## InverseHeatSolver Framework

The `InverseHeatSolver` implements the PINN-based solution for this inverse problem by extending and customizing the **DeepXDE** framework. It allows for:

- Modular configuration of PDE domains, boundary conditions, and data
- Support for regularization (e.g., $`\|\nabla a(x)\|^2`$)
- gPINNs (gradient-enhanced PINNs)
- Robust training from noisy temperature observations

---
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
