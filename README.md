Davidson Eigensolver
===================
This package contains a C++ implementation of the *Davidson diagonalization algorithms*.
Different schemas are available to compute the correction.

Available correction methods are:
 * **davidson**: Diagonal-Preconditioned-Residue
 * **jacobi**: Generalized Jacobi Davidson

 ### Note:
The Davidson method is suitable for **diagonal-dominant symmetric matrices**, that are quite common
in certain scientific problems like [electronic structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson method could be not practical
for other kind of symmetric matrices.i

Usage
-----
The following program calls the `solve` *subroutine* from the `davidson` module and computes
the lowest eigenvalue and corresponding eigenvector, using the *jacobi* correction method with a tolerance of `1e-8`.
```C++
eigenSolver<real> solver;
solver.solverOptions.numberOfEigenValues = numEig;
solver.solverOptions.tolerence = 1e-8;
solver.solverOptions.solver = "jacobi";
solver.solverOptions.sizeOfTheMatrix = A.Height();
solver.solve(A, grid);
```

The helper  `generateDDHermitianMatrix` function generates a diagonal dominant matrix.

**Variables**:
 * `A` (*in*) matrix to diagonalize
 * `solver.eigenValues` (*out*) resulting eigenvalues
 * `solver.eigenVectorsFull` (*out*) resulting eigenvectors
 * `solver.SolverOptions.solver`(*in*) Either "davidson" or "jacobi"
 * `solver.solverOptions.tolerance`(*in*) Numerical tolerance for convergence

### References:
 * [Davidson diagonalization method and its applications to electronic structure calculations](https://www.semanticscholar.org/paper/DAVIDSON-DIAGONALIZATION-METHOD-AND-ITS-APPLICATION-Liao/5811eaf768d1a006f505dfe24f329874a679ba59)
 * [Numerical Methods for Large Eigenvalue Problem](https://doi.org/10.1137/1.9781611970739)

Installation and Testing
------------------------

To compile execute:
```
cmake -H. -Bbuild && cmake --build build
```

To run the test:
```
cmake -H. -Bbuild -DENABLE_TEST=ON && cmake --build build
cd build && ctest -V
```

Dependencies
------------
This packages assumes that you have installed the following packages:
 * A C++ compiler >=  C++ 14
 * [CMake](https://cmake.org/)
 * [Elemental] (https://github.com/elemental/Elemental)
