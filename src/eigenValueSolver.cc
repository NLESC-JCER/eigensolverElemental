//
//  eigenValueSolver.cpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#include "eigenValueSolver.hpp"
namespace eigenValueSolver {

template <typename real>
void eigenSolver<real>::initialise(El::Grid &grid) {

  // Set the grid (MPI) for all the member matrices/vectors
  V.SetGrid(grid);
  Vsub.SetGrid(grid);
  correctionVector.SetGrid(grid);
  eigenVectors.SetGrid(grid);
  ritzVectors.SetGrid(grid);
  eigenValues.SetGrid(grid);
  eigenValues_old.SetGrid(grid);
  AV.SetGrid(grid);
  residual.SetGrid(grid);
  ritzVectors.SetGrid(grid);

  begTheta = El::Range<int>{0, solverOptions.numberOfEigenValues};
  endTheta = El::Range<int>{0, 1};

  // Initialize the member matrices/vectors
  El::Identity(searchSpace, solverOptions.sizeOfTheMatrix,
               solverOptions.sizeOfTheMatrix);
  El::Identity(searchSpacesub, solverOptions.sizeOfTheMatrix,
               solverOptions.sizeOfTheMatrix);
  El::Identity(correctionVector, solverOptions.sizeOfTheMatrix,
               columnsOfSearchSpace);
  El::Identity(eigenVectors, solverOptions.sizeOfTheMatrix,
               solverOptions.sizeOfTheMatrix);
  //El::Identity(ritzVectors, solverOptions.sizeOfTheMatrix,
  //             solverOptions.sizeOfTheMatrix);
  El::Identity(eigenValues, solverOptions.sizeOfTheMatrix, 1);
  El::Identity(AV, solverOptions.sizeOfTheMatrix,
               solverOptions.sizeOfTheMatrix);
  El::Identity(residual, solverOptions.sizeOfTheMatrix, 1);
}

template <typename real>
void eigenSolver<real>::subspaceProblem(int iterations,
                                        const El::DistMatrix<real> &A,
                                        El::Grid &grid) {

  //Initialise the searchSpaceSub as a sub matrix of searchSpace                                        
  searchSpacesub = searchSpace;
  searchSpacesub.Resize(solverOptions.sizeOfTheMatrix, iterations + 1);

  //Initialise T = V^TAV
  El::DistMatrix<real> T(grid);
  El::Zeros(T, iterations + 1, iterations + 1);

  //Initialize the ritz vectors
  El::Zeros(ritzVectors, solverOptions.sizeOfTheMatrix, iterations + 1);

  //Parameters for GEMM
  real alpha = 1, beta = 0;

  //AV
  El::Gemm(El::NORMAL, El::NORMAL, alpha, A, searchSpacesub, beta, AV);
  //V^TAV
  El::Gemm(El::TRANSPOSE, El::NORMAL, alpha, searchSpacesub, AV, beta, T);

  //Get the eigen pairs for the reduced problem V^TAV
  El::HermitianEig(El::UPPER, T, eigenValues, eigenVectors);

  //Calculate the ritz vectors
  El::Gemm(El::NORMAL, El::NORMAL, alpha, searchSpacesub, eigenVectors, beta,
           ritzVectors);
}

template <typename real>
void eigenSolver<real>::expandSearchSpace(int iterations,
                                          const El::DistMatrix<real> &A,
                                          El::Grid &grid) {

                                        
  for (int j = 0; j < columnsOfSearchSpace; ++j) {
    El::Range<int> beg(0, iterations + 1);
    El::Range<int> end(j, j + 1);

    El::DistMatrix<real> residual(grid);  // residual Ay-thetay
    El::Zeros(residual, solverOptions.sizeOfTheMatrix, 1);

    // calculate the ritz vector Vs
    El::DistMatrix<real> Vs(grid);
    El::Zeros(Vs, solverOptions.sizeOfTheMatrix, 1);
    real alpha = 1, beta = 0;
    El::Gemv(El::NORMAL, alpha, searchSpacesub, eigenVectors(beg, end), beta,
             Vs);

    // Identitiy matrix
    El::DistMatrix<real> I(grid);
    El::Identity(
        I, solverOptions.sizeOfTheMatrix,
        solverOptions.sizeOfTheMatrix);  // Initialize as identity matrix
    I *= eigenValues.GetLocal(j, 0);
    El::DistMatrix<real> Atemp(AV);
    Atemp -= I;  // A-theta*I

    // Calculate the residual r=(A-theta*I)*Vs
    El::Gemv(El::NORMAL, alpha, Atemp, Vs, beta, residual);

    if (solverOptions.solver == "davidson") {
      real den = 1.0 / eigenValues.GetLocal(j, 0) - A.GetLocal(j, j);

      correctionVector = residual;  // new search direction
      correctionVector *= den;
    } else if (solverOptions.solver == "jacobi") {
      El::DistMatrix<real> proj(grid);  // projector matrix
      El::Zeros(proj, solverOptions.sizeOfTheMatrix,
                solverOptions.sizeOfTheMatrix);
      El::DistMatrix<real> Ip(grid);  // Identitiy matrix
      El::Identity(Ip, solverOptions.sizeOfTheMatrix,
                   solverOptions.sizeOfTheMatrix);

      El::Gemm(El::NORMAL, El::TRANSPOSE, alpha, Vs, Vs, beta, proj);
      proj -= I;
      proj *= -1.0;

      El::DistMatrix<real> projProd(grid);  // product of the projectors
      El::Zeros(projProd, solverOptions.sizeOfTheMatrix,
                solverOptions.sizeOfTheMatrix);

      El::DistMatrix<real> projTemp(grid);  // temp intermediate step
      El::Zeros(projTemp, solverOptions.sizeOfTheMatrix,
                solverOptions.sizeOfTheMatrix);

      El::Gemm(El::NORMAL, El::TRANSPOSE, alpha, proj, Atemp, beta, projTemp);
      El::Gemm(El::NORMAL, El::TRANSPOSE, alpha, projTemp, proj, beta,
               projProd);

      correctionVector = residual;
      correctionVector *= -1.0;

      El::LinearSolve(projProd, correctionVector);
    }

    for (int k = 0; k < solverOptions.sizeOfTheMatrix; ++k) {
      searchSpace.SetLocal(k, iterations + j + 1,
                           correctionVector.GetLocal(k, 0));
    }
    eigenValues_old -= eigenValues(begTheta, endTheta);

    if (El::Nrm2(eigenValues_old) < solverOptions.tolerence) {
      break;
    }
  }
}

template <typename real>
void eigenSolver<real>::solve(const El::DistMatrix<real> &A, El::Grid &grid) {

  eigenSolver<real>::initialise(grid);

  int maximumIterations = solverOptions.sizeOfTheMatrix / 2;
  double rNorm = 1;
  int iterations = columnsOfSearchSpace;
  // for (int iterations = columnsOfSearchSpace; iterations < maximumIterations;
  // iterations = iterations + columnsOfSearchSpace)
  while (rNorm > solverOptions.tolerence) {

    if (iterations <= columnsOfSearchSpace)  // If it is the first iteration
                                             // copy t to V
    {

      for (int i = 0; i < solverOptions.sizeOfTheMatrix; ++i) {
        for (int j = 0; j < columnsOfSearchSpace; ++j) {
          searchSpace.SetLocal(i, j, correctionVector.GetLocal(i, j));
        }
      }
      El::Ones(eigenValues_old, solverOptions.sizeOfTheMatrix,
               1);  // so this not to converge immediately
    } else  // if its not the first iteration then set old theta to the new one
    {
      eigenValues_old = eigenValues(begTheta, endTheta);
    }

    // Orthogonalize the searchSpace matrix using QR
    El::DistMatrix<real> R;  // R matrix for QR factorization
    El::Zeros(R, solverOptions.sizeOfTheMatrix, solverOptions.sizeOfTheMatrix);

    // QR factorization of V
    El::qr::Explicit(searchSpace, R, false);

    // solve the subspace problem VTAV
    subspaceProblem(iterations, A, grid);
    // expand the search space
    expandSearchSpace(iterations, A, grid);

    // TEST FOR CONVERGENCE
    El::DistMatrix<real> Ax(grid);
    El::DistMatrix<real> lambdax(grid);
    int matrixSize = solverOptions.sizeOfTheMatrix;
    El::Zeros(Ax, matrixSize, 1);
    El::Zeros(lambdax, matrixSize, 1);
    El::Range<int> beg(0, solverOptions.sizeOfTheMatrix);
    El::Range<int> end(0, 1);

    double eVal = eigenValues.GetLocal(0, 0);
    El::DistMatrix<real> eVec = eigenVectorsFull(beg, end);
    real alpha = 1, beta = 0;
    El::Gemm(El::NORMAL, El::NORMAL, alpha, A, eigenVectorsFull(beg, end), beta,
             Ax);
    El::Scale(eVal, eVec);

    El::DistMatrix<real> r = Ax;
    r -= eVec;
    rNorm = El::Nrm2(r);
    iterations = iterations + columnsOfSearchSpace;
    std::cout << "error norm = " << iterations << " " << rNorm << std::endl;
  }
  // El::Print(eigenVectors);
}

// explicit instantiations
template class eigenSolver<float>;
template class eigenSolver<double>;
}  // namespace eigenValueSolver
