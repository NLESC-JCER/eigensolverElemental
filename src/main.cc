//
//  main.cpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#include "eigenValueSolver.hpp"
#include "utils.hpp"
#include <limits>
#include <iomanip>

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

    // initalise all the MPI stuff
    const El::Int blocksize =
        El::Input("--blocksize", "algorithmic blocksize", 128);
    El::Int gridHeight = El::Input("--gridHeight", "grid height", 0);
    // const El::Int numEigVal =
    // El::Input("--numeigval", "size of matrix",
    //     10); // Number of eigenvalues to be evaluated
    const El::Int numRhs = El::Input("--numRhs", "# of right-hand sides", 1);
    const bool error = El::Input("--error", "test Elemental error?", true);
    const bool details = El::Input("--details", "print norm details?", true);
    const El::Int matrixSize = El::Input("--size", "size of matrix", 100);
    const El::Int numEig = El::Input("--numeig", "number of eigenvalues", 1);
    const std::string solverType =
        El::Input("--solver", "solver used", "davidson");

    // Set block size
    El::SetBlocksize(blocksize);

    // If the grid height wasn't specified, then we should attempt to build
    // a nearly-square process grid
    if (gridHeight == 0) gridHeight = El::Grid::DefaultHeight(commSize);
    El::Grid grid{comm, gridHeight};
    if (commRank == 0)
      El::Output("Grid is: ", grid.Height(), " x ", grid.Width());

    // Set up random A and B, then make the copies X := B
    El::Timer timer;

    // The matrix A whose eigenvalues have to be evaluated
    El::DistMatrix<real> A(grid);
    El::Zeros(A, matrixSize, matrixSize);
    // Generate the Diagonally dominant hermitian matrix
    generateDDHermitianMatrix<real>(A);

    eigenSolver<real> solver;
    solver.solverOptions.numberOfEigenValues = numEig;
    solver.solverOptions.tolerence = 1e-8;
    solver.solverOptions.solver = solverType;
    solver.solverOptions.sizeOfTheMatrix = A.Height();
    El::Output(A.Height());
    solver.solve(A, grid);
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) <<
     solver.eigenValues.GetLocal(0,0) << std::endl;
  } catch (std::exception &e) {
    El::ReportException(e);
  }
  return 0;
}
