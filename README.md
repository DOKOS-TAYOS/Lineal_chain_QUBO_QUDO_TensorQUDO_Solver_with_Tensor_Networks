# README

## Polynomial-Time Tensor Network Solver for Tridiagonal QUBO, QUDO, and Tensor QUDO Problems

This repository provides the implementation of the algorithms described in the paper:

> **"Polynomial-time Solver of Tridiagonal QUBO, QUDO and Tensor QUDO Problems with Tensor Networks"**
> by Alejandro Mata Ali, Iñigo Perez Delgado, Marina Ristol Roura, and Aitor Moreno Fdez. de Leceta
> ([arXiv:2309.10509](https://arxiv.org/abs/2309.10509))

The code implements tensor network algorithms to solve tridiagonal QUBO (Quadratic Unconstrained Binary Optimization), QUDO (Quadratic Unconstrained Discrete Optimization), and Tensor QUDO problems with nearest-neighbor interactions in a 1D chain.

---

## 📘 Contents

* `Tensor_QUDO_Solver_tensor_networks.ipynb`: Jupyter notebook implementing the solver algorithms, demonstrations, and benchmark experiments.
* `Polynomial_time_Solver_of_Tridiagonal_QUBO__QUDO_and_Tensor_QUDO_problems_with_Tensor_Networks.pdf`: The full paper with theoretical background and complexity analysis.

---

## 🧠 Key Features

* **Quantum-Inspired Approach**: The algorithm simulates imaginary-time evolution of a quantum state using tensor networks, allowing for efficient extraction of the optimal solution via partial trace.
* **Support for Multiple Optimization Classes**:

  * **QUBO**: Binary variables, quadratic cost function, tridiagonal structure.
  * **QUDO**: Discrete variables of arbitrary dimension.
  * **Tensor QUDO**: Cost function defined via a position-dependent tensor.
* **Polynomial-Time Execution**:

  * QUBO: $O(N^2)$ complexity
  * QUDO and Tensor QUDO: $O(N^2 D^2)$ complexity
* **Degeneracy Handling**: The algorithm is robust to degenerate ground states, and multiple optimal solutions can be extracted.
* **Explicit Equation**: Provides a closed-form tensor network contraction equation for each solution component.
* **Efficient Contractions**: Optimized contraction using MPO-like structure, including the Humbucker method for phase-cancelled state filtering.

---

## 🔧 Dependencies

This notebook was developed using:

* Python 3.9+
* NumPy
* Decimal (for high-precision arithmetic)
* Matplotlib (optional, for plotting)
* tqdm
* ORTOOLS
* dimod

Install dependencies via:

```bash
pip install numpy matplotlib tqdm dimod ortools
```

---

## 🚀 How to Run

1. Open the `Tensor_QUDO_Solver_tensor_networks.ipynb` notebook.
2. Run all cells to load and initialize the solver.
3. Modify the problem instance in the cell titled `# Experiments` to define your own QUBO, QUDO, or Tensor QUDO problem.

---

## 📊 Experiments

The notebook includes empirical experiments to evaluate:

* Runtime scaling with number of variables $N$
* Runtime scaling with variable dimension $D$
* Performance against Google OR-Tools and D-Wave `dimod` solvers
* Dependence of solution quality on the decay parameter $\tau$

---

## 📄 Reference

If you use this code, please cite:

```
@article{mataali2023polynomial,
  title={Polynomial-time Solver of Tridiagonal QUBO, QUDO and Tensor QUDO Problems with Tensor Networks},
  author={Mata Ali, Alejandro and Perez Delgado, Iñigo and Ristol Roura, Marina and Moreno Fdez. de Leceta, Aitor},
  journal={arXiv preprint arXiv:2309.10509},
  year={2023}
}
```

---

## 📬 Contact

* Alejandro Mata Ali: [alejandro.mata.ali@gmail.com](mailto:alejandro.mata.ali@gmail.com)

---

## 🧩 Acknowledgments

This research was supported by the **Q4Real project (Quantum Computing for Real Industries)** under HAZITEK 2022, grant number ZE-2022/00033.
