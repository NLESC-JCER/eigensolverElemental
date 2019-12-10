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
the lowest 3 eigenvalues and corresponding eigenvectors, using the *jacobi* correction method with a tolerance of `1e-8`.
```C++
eigenSolver<real> solver;
solver.solverOptions.numberOfEigenValues = numEig;
solver.solverOptions.tolerence = 1e-8;
solver.solverOptions.solver = "jacobi";
solver.solverOptions.sizeOfTheMatrix = A.Height();
solver.solve(A, grid);
