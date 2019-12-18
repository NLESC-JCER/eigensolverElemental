#define BOOST_TEST_MODULE eigen_solver
#include "eigenValueSolver.hpp"
#include "utils.hpp"
#include <boost/test/included/unit_test.hpp>

using namespace eigenValueSolver;
using namespace El;

/**test the davidson implementation*/
BOOST_AUTO_TEST_CASE(davidson_solver) {
  // initialise elemental mpi
  El::Environment env;
  El::mpi::Comm comm = El::mpi::COMM_WORLD;
  const int commRank = El::mpi::Rank(comm);
  const int commSize = El::mpi::Size(comm);
  typedef double real;
  El::Int gridHeight = 0;
  gridHeight = El::Grid::DefaultHeight(commSize);
  El::Grid grid{comm, gridHeight};

  // The matrix A whose eigenvalues have to be evaluated
  real matrixSize = 100;
  El::DistMatrix<real> A(grid);
  El::DistMatrix<real> Ax(grid);
  El::DistMatrix<real> lambdax(grid);
  El::Zeros(A, matrixSize, matrixSize);
  El::Zeros(Ax, matrixSize, 1);
  El::Zeros(lambdax, matrixSize, 1);

  // Generate the Diagonally dominant hermitian matrix
  generateDDHermitianMatrix<real>(A);
  //El::Print(A);
  eigenSolver<real> solver;
  solver.solverOptions.numberOfEigenValues = 4;
  solver.solverOptions.tolerence = 1e-8;
  solver.solverOptions.solver = "davidson";
  solver.solverOptions.sizeOfTheMatrix = A.Height();

  solver.solve(A, grid);

  /**Check if Ax-lambda.x < 1e-6 holds!!
  */
  El::Range<int> beg(0, solver.solverOptions.sizeOfTheMatrix);
  El::Range<int> end(0, 1);

  double eVal = solver.eigenValues.GetLocal(0,0);
  El::DistMatrix<real> eVec = solver.ritzVectors(beg,end);

  El::Gemm(El::NORMAL, El::NORMAL, 1.0, A, solver.ritzVectors(beg,end), 0.0, Ax);
  El::Scale(eVal, eVec);


  El::DistMatrix<real> r = Ax;
  r -= eVec;
  double rNorm = El::Nrm2(r);

  BOOST_CHECK(rNorm < 1e-6);
}
