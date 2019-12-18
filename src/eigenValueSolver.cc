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
  AV.SetGrid(grid);
  residual.SetGrid(grid);
  ritzVectors.SetGrid(grid);

  // Initialize eigenValue ranges
  begTheta = El::Range<int>{0, solverOptions.numberOfEigenValues};
  endTheta = El::Range<int>{0, 1};

  // Initialize the member matrices/vectors
  El::Identity(V, solverOptions.sizeOfTheMatrix, solverOptions.sizeOfTheMatrix);
  El::Identity(Vsub, solverOptions.sizeOfTheMatrix,
               solverOptions.sizeOfTheMatrix);
  El::Identity(eigenVectors, solverOptions.sizeOfTheMatrix,
               solverOptions.sizeOfTheMatrix);
  // El::Identity(ritzVectors, solverOptions.sizeOfTheMatrix,
  //             solverOptions.sizeOfTheMatrix);
  El::Identity(eigenValues, solverOptions.sizeOfTheMatrix, 1);
  El::Identity(residual, solverOptions.sizeOfTheMatrix, 1);
  // Initialize the correction vector
  El::Identity(correctionVector, solverOptions.sizeOfTheMatrix,
               sizeOfSearchSpace);
}

// Calculate the eigen pairs of the V^TAV problem
template <typename real>
void eigenSolver<real>::subspaceProblem(int iterations,
                                        const El::DistMatrix<real> &A,
                                        El::Grid &grid) {

  // Initialise the VSub as a sub matrix of V
  Vsub = V;
  Vsub.Resize(solverOptions.sizeOfTheMatrix, iterations + 1);

  // Initialise T = V^TAV
  El::DistMatrix<real> T(grid);
  El::Zeros(T, iterations + 1, iterations + 1);

  // Initialize AV
  El::Identity(AV, solverOptions.sizeOfTheMatrix, iterations + 1);

  // Initialize the ritz vectors
  El::Zeros(ritzVectors, solverOptions.sizeOfTheMatrix, iterations + 1);

  // Parameters for GEMM
  real alpha = 1, beta = 0;

  // Calculate AV
  El::Gemm(El::NORMAL, El::NORMAL, alpha, A, Vsub, beta, AV);

  // Calculate V^TAV
  El::Gemm(El::TRANSPOSE, El::NORMAL, alpha, Vsub, AV, beta, T);

  // Get the eigen pairs for the reduced problem V^TAV
  El::HermitianEig(El::UPPER, T, eigenValues, eigenVectors);
}

// Calculate the residual and the correction vector
template <typename real>
void eigenSolver<real>::expandSearchSpace(int iterations,
                                          const El::DistMatrix<real> &A,
                                          El::Grid &grid) {

  //
  // for (int j = 0; j < sizeOfSearchSpace; ++j) {
  for (int j = 0; j < iterations + 1; ++j) {

    // Set ranges to get jth eigen vectors
    El::Range<int> beg(0, iterations + 1);
    El::Range<int> end(j, j + 1);

    // Set ranges to get to get the relevent part of AV
    El::Range<int> begAV(0, iterations + 1);
    El::Range<int> endAV(0, iterations + 1);

    // Parameters for GEMM
    real alpha = 1, beta = 0;

    // Calculate the ritz vectors
    El::Gemm(El::NORMAL, El::NORMAL, alpha, Vsub, eigenVectors, beta,
             ritzVectors);

    // Matrix to store AV*s_j, where s is the eigenVector of the reduced problem
    // and Vs the ritz vectors
    El::DistMatrix<real> AVs(grid);
    El::Zeros(AVs, iterations + 1, 1);

    // Calculate AV*s_j
    El::Gemm(El::NORMAL, El::NORMAL, alpha, AV(begAV, endAV),
             eigenVectors(beg, end), beta, AVs);

    // Identitiy matrix
    El::DistMatrix<real> I(grid);
    El::Identity(I, iterations + 1, iterations + 1);

    // theta_j*I
    I *= eigenValues.GetLocal(j, 0);

    // Matrix to store theta*V*s_j
    El::DistMatrix<real> thetaVs(grid);
    El::Zeros(thetaVs, iterations + 1, 1);

    // Calculate theta_j*I*Vs_j
    El::Gemm(El::NORMAL, El::NORMAL, alpha, I, ritzVectors(beg, end), beta,
             thetaVs);

    // Calculate residual = AVs - thetaVs
    residual = AVs;
    residual -= thetaVs;

    if (solverOptions.solver == "davidson") {
      for (int k = 0; k < iterations + 1; ++k) {
        // Calculate the correction vector = r/(D-theta*I)
        real denominator = 1.0 / eigenValues.GetLocal(j, 0) - A.GetLocal(k, k);
        correctionVector.SetLocal(k, 0, residual.GetLocal(k, 0) * denominator);
        El::Print(correctionVector);
      }
    } /*else if (solverOptions.solver == "jacobi") {
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
    }*/

    // Append the correction vector to the search space
    for (int k = 0; k < solverOptions.sizeOfTheMatrix; ++k) {
      V.SetLocal(k, iterations + j + 1, correctionVector.GetLocal(k, 0));
    }

    // If residual is less than tolerence, then return
    if (El::Nrm2(residual) < solverOptions.tolerence) {
      return;
    }
  }
}

template <typename real>
void eigenSolver<real>::solve(const El::DistMatrix<real> &A, El::Grid &grid) {

  //Set grid sizes of the member matrices and initialize them
  eigenSolver<real>::initialise(grid);


  int iterations = sizeOfSearchSpace;
  // for (int iterations = columnsOfSearchSpace; iterations < maximumIterations;
  // iterations = iterations + columnsOfSearchSpace)
  while (rNorm > solverOptions.tolerence) {

    // Orthogonalize the searchSpace matrix using QR
    El::DistMatrix<real> R;  // R matrix for QR factorization
    El::Zeros(R, solverOptions.sizeOfTheMatrix, solverOptions.sizeOfTheMatrix);

    // QR factorization of V
    El::qr::Explicit(V, R, false);

    // solve the subspace problem VTAV
    subspaceProblem(iterations, A, grid);
    // expand the search space
    expandSearchSpace(iterations, A, grid);
    if (El::Nrm2(residual) < solverOptions.tolerence) {
      return;
    }
    iterations = iterations + sizeOfSearchSpace;
    
  }
}

// explicit instantiations
template class eigenSolver<float>;
template class eigenSolver<double>;
}  // namespace eigenValueSolver
