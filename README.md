# Introduction to the Project

Scientific Machine Learning (SciML) is an innovative field that combines traditional numerical methods with modern machine learning. This approach offers fresh insights into solving Partial Differential Equations (PDEs) and Ordinary Differential Equations (ODEs), crucial for understanding complex systems in physics, chemistry, and biology. It excels in identifying patterns and improving simulations and predictions with high precision.

Traditionally, solving these equations relied on analytical methods suited for simpler problems, or numerical methods like finite difference and finite element techniques. However, these can fall short when dealing with complex, high-dimensional problems.

SciML stands out by leveraging machine learning algorithms, especially neural networks, to approximate solutions to differential equations more efficiently. These networks are trained to minimize the difference between their predictions and actual solutions, offering a powerful tool for tackling complex equations.

In this Julia notebook, SciML was applied to the Hénon-Heiles system, demonstrating how this method can illuminate and solve challenging mathematical problems.


## The Hénon-Heiles System

The Hénon-Heiles system is a notable example in the field of dynamical systems, originally introduced in 1964 by Michel Hénon and Carl Heiles as a simplified model to study the motion of stars within a galaxy. It has since become a classic problem used to explore chaos theory and nonlinear dynamics. The system was initially designed to approximate the third-order terms of the Taylor series expansion of the gravitational potential of a galaxy, providing insights into the complexity of stellar motion.

The Hénon-Heiles Hamiltonian, which represents the total energy of the system, is given by the following equation:

H(x, y, p_x, p_y) = 1/2(p_x^2 + p_y^2) + V(x, y)

where `p_x` and `p_y` are the momenta corresponding to the coordinates `x` and `y`, respectively. The potential `V(x, y)` is defined as:

V(x, y) = 1/2(x^2 + y^2) + x^2y - 1/3y^3


This potential leads to equations of motion that are nonlinear and, under certain energy conditions, can exhibit both regular (periodic) and chaotic behaviors. The Hénon-Heiles system is particularly famous for its role in the discovery and study of chaotic dynamics in conservative systems.

## Defining and Initializing the Hénon-Heiles System in Julia

In the Julia notebook, a function named `henonheiles` is implemented to calculate the derivatives of the system's state variables, crucial for solving the system's Ordinary Differential Equations (ODEs) through a numerical solver. Below is a breakdown of the function's components and their roles:

- `du`: A vector designated to hold the derivatives of the state vector `u`.
- `u`: Represents the state vector of the system, where `u[1]` and `u[2]` correspond to the spatial coordinates `x` and `y`, and `u[3]` and `u[4]` to their respective velocities or momenta `p_x` and `p_y`.
- `p`: A parameter vector for the system; it is not utilized in this specific context.
- `t`: Denotes the time variable.

The function's equations are aligned with the motion equations derived from the Hénon-Heiles Hamiltonian:

- `du[1] = u[3]` and `du[2] = u[4]` directly relate to the time derivatives of `x` and `y`, effectively the velocities `p_x` and `p_y`.
- `du[3] = -u[1] - 2*u[1]*u[2]` and `du[4] = -u[2] - u[1]^2 + u[2]^2` calculate the accelerations, based on the gradient of the potential `V(x, y)`.

### Initializing the ODE Problem

To analyze the Hénon-Heiles system's dynamics, initial conditions and a simulation time span are established:

- `u0` sets the system's initial state wi
- th positions `x=0.2` and `y=0.0`, along with velocities `p_x=0.4` and `p_y=0.0`.
- `tspan` delineates the simulation period, extending from `t=0` to `t=500`.
- `prob = ODEProblem(henonheiles,u0,tspan)` crafts an ODE problem instance, incorporating the `henonheiles` function, the initial conditions `u0`, and the defined time span `tspan`. This formulation is then ready for solution using Julia's suite of ODE solvers.

This setup allows for the exploration of the Hénon-Heiles system within a computational framework, facilitating the examination of its behavior over time through numerical integration.

## Numerical Solver Comparison in the Hénon-Heiles System Study

### Overview

In the study of differential equations, selecting an appropriate numerical solver is crucial, as each solver has its own strengths and weaknesses depending on the specific characteristics of the system being analyzed. For the Hénon-Heiles system, three different solvers were explored: `Tsit5()`, a stiff solver, and Julia's default solver, to understand their performance and suitability for our problem.

### Tsit5() Solver

`Tsit5()`, representing the Tsitouras 5th order Runge-Kutta method, is a highly efficient non-stiff solver known for its accuracy in solving ordinary differential equations (ODEs) where solutions exhibit gradual changes. For the Hénon-Heiles system:

- **Memory Estimate**: 248.78 KiB
- **Allocations Estimate**: 740

The low memory usage and minimal allocations highlight `Tsit5()`'s computational efficiency, indicating it as an excellent choice for this system under study.

This is the graph of the system which is solved with Tsit5( ) solver: 
![Ekran görüntüsü 2024-02-12 002359_kopya](https://github.com/tugcecalisir/Differental-Equations/assets/103861412/d87c303d-5296-45f8-9d53-5ce5b66f62d7)


### Stiff Solver

Stiff solvers are designed for ODE systems that experience rapid solution changes, requiring careful handling to ensure numerical stability. Our findings using a stiff solver on the Hénon-Heiles system revealed:

- **Memory Estimate**: 11.46 MiB
- **Allocations Estimate**: 29833

Compared to `Tsit5()`, the stiff solver requires significantly more memory and computational steps, indicative of its detailed and robust approach to handling stiff dynamics. However, for this particular problem, the increased resource demand suggests that a stiff solver might not be necessary, as evidenced by the system's behavior in the accompanying graph.

This is the graph of the system which is solved with Stiff Solver: 
![Ekran görüntüsü 2024-02-12 002430_kopya](https://github.com/tugcecalisir/Differental-Equations/assets/103861412/bdfe4735-3f08-466c-9384-5aea7d178d09)


### Default Solver

Julia's default solver is adaptively chosen based on the ODE problem's traits, aiming to provide a balanced approach to computational efficiency and accuracy. For our Hénon-Heiles system study:

- **Memory Estimate**: 272.62 KiB
- **Allocations Estimate**: 780

The performance metrics of the default solver closely align with those of the `Tsit5()` solver, suggesting that for this system, the default solver opts for a method similar to `Tsit5()`, favoring non-stiff solving strategies. This adaptability is reflected in the graph of the system solved by the default solver.

This is the graph of the system which solves with Default Solver:
![Ekran görüntüsü 2024-02-12 002447_kopya](https://github.com/tugcecalisir/Differental-Equations/assets/103861412/1c1610e7-1a87-426a-a574-57d4b8de48ae)


## Investigating the Potential Energy Function of the Hénon-Heiles System

In our study, the potential energy function `V(x, y)` characterizing the Hénon-Heiles system was searched in detail. To explore this, ranges for `x` and `y` to span the `x-y` plane were defined, each ranging from `-0.75` to `0.75` with a step size of `0.05`. This setup is pivotal for evaluating `V(x, y)` across a grid, facilitating a detailed visualization of the system's potential energy landscape.

The potential energy function, `V(x, y)`, is mathematically expressed as:

V(x, y) = 1/2(x^2 + y^2) + x^2y - 1/3y^3


This formulation includes several key components:

- **Harmonic Oscillator Term**: `1/2(x^2 + y^2)`, symbolizing the quadratic, isotropic potential that confines particles within a specific spatial region.
- **Coupling Term**: `x^2y`, introducing non-linear interactions between the `x` and `y` coordinates.
- **Cubic Term**: `-1/3y^3`, injecting asymmetry into the potential, thereby enriching the system with non-linear dynamics and the potential for chaotic behavior.

Through this exploration, the aim is to provide a comprehensive understanding of the forces at play within the Hénon-Heiles system and how they contribute to its fascinating dynamics.

Included are several graphical representations of the potential energy component of the Hénon-Heiles Hamiltonian:

![Ekran görüntüsü 2024-02-12 002503_kopya](https://github.com/tugcecalisir/Differental-Equations/assets/103861412/5b9c42a3-0ec0-4847-a4c6-81e72d622a01)

![Ekran görüntüsü 2024-02-12 002514_kopya](https://github.com/tugcecalisir/Differental-Equations/assets/103861412/b8358143-3cc6-4aae-82e7-a691263a2598)




