Davidson and Jacobi Eigensolver
===================
This package contains a C++ implementation of the *Davidson and Jacobi diagonalization algorithms*.
Different schemas are available to compute the correction.

Available correction methods are:
 * **DPR**: Diagonal-Preconditioned-Residue
 * **GJD**: Generalized Jacobi Davidson

 ### Note:
The Davidson method is suitable for **diagonal-dominant symmetric matrices**, that are quite common
in certain scientific problems like [electronic structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson method could be not practical
for other kind of symmetric matrices.
