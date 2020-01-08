//
//  main.cpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#include "eigenValueSolver.hpp"
#include "utils.hpp"
#include <iomanip>
#include <limits>

using namespace eigenValueSolver;
using namespace El;

int main(int argc, char *argv[]) {
  // typedef to use either double or float
  typedef double real;

  // initialise elemental mpi
  El::Environment env(argc, argv);
  El::mpi::Comm comm = El::mpi::COMM_WORLD;
  const int commRank = El::mpi::Rank(comm);
  const int commSize = El::mpi::Size(comm);

  try {

    // Initalise all the MPI variables
    // Size of the block
    const El::Int blocksize =
        El::Input("--blocksize", "algorithmic blocksize", 128);

    // Height of the grid
    El::Int gridHeight = El::Input("--gridHeight", "grid height", 0);

    // Number of right hand sides
    const El::Int numRhs = El::Input("--numRhs", "# of right-hand sides", 1);

    // Error check
    const bool error = El::Input("--error", "test Elemental error?", true);

    // Print details
    const bool details = El::Input("--details", "print norm details?", true);

    // Size of the input matrix
    const El::Int matrixSize = El::Input("--size", "size of matrix", 10);

    // The number of eigenvalues to be calculated
    const El::Int numEig = El::Input("--numeig", "number of eigenvalues", 1);

    // Type of the solver used
    const std::string solverType =
        El::Input("--solver", "solver used", "davidson");

    // Set block size
    El::SetBlocksize(blocksize);

    // If the grid height wasn't specified, then we should attempt to build a
    // nearly-square process grid
    if (gridHeight == 0) gridHeight = El::Grid::DefaultHeight(commSize);
    El::Grid grid{comm, gridHeight};
    if (commRank == 0)
      El::Output("Grid is: ", grid.Height(), " x ", grid.Width());

    // The matrix A whose eigenvalues have to be evaluated
    El::DistMatrix<real> A(grid);

    // Initialize the matrix with zeros
    El::Zeros(A, matrixSize, matrixSize);

    // Generate the Diagonally dominant hermitian matrix
    generateDDHermitianMatrix<real>(A);

    // Create an instance of the eigenSolver class
    eigenSolver<real> solver;

    // Set solver options
    solver.solverOptions.numberOfEigenValues = numEig;
    solver.solverOptions.tolerence = 1e-8;
    solver.solverOptions.solver = solverType;
    solver.solverOptions.sizeOfTheMatrix = A.Height();

    // Solve function which calculates the eigenvalues and eigenvectors for
    // matrix A
    solver.solve(A, grid);

    // Print eigenvalues
    for (int i = 0; i < numEig; ++i) {
      std::cout << std::setprecision(
                       std::numeric_limits<long double>::digits10 + 1)
                << solver.eigenValues.GetLocal(i, 0) << std::endl;
    }
  } catch (std::exception &e) {
    El::ReportException(e);
  }
  return 0;
}
