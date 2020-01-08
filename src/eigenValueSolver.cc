//
//  eigenValueSolver.cpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#include "eigenValueSolver.hpp"
#include <cmath>

namespace eigenValueSolver {

template <typename real>
void eigenSolver<real>::initialise(El::Grid &grid,
                                   const El::DistMatrix<real> &A) {
  // Intitialize the size of the search space
  sizeOfSearchSpace = solverOptions.numberOfEigenValues * 3;

  // Intialize the search space V as an identity matrix of size n x
  // sizeOfSearchSpace
  V.SetGrid(grid);
  El::Identity(V, solverOptions.sizeOfTheMatrix, sizeOfSearchSpace);

  // Initialize the Vsub which part of the search space relevent to the current
  // iteration of size n x numEigVal
  Vsub.SetGrid(grid);
  El::Identity(Vsub, solverOptions.sizeOfTheMatrix,
               solverOptions.numberOfEigenValues);

  // Initialize the eigen vectors
  eigenVectors.SetGrid(grid);
  El::Identity(eigenVectors, solverOptions.sizeOfTheMatrix,
               solverOptions.numberOfEigenValues);

  // Initalize the residuals for all the eigenvalues
  residual.SetGrid(grid);
  El::Identity(residual, solverOptions.sizeOfTheMatrix,
               solverOptions.numberOfEigenValues);

  // Set the eigen value vector to diagonal values of input matrix and calculate
  // respective residuals
  eigenValues.SetGrid(grid);
  El::Identity(eigenValues, solverOptions.numberOfEigenValues, 1);

  ritzVectors.SetGrid(grid);
  AV.SetGrid(grid);

}  // namespace eigenValueSolver

// Calculate the residual and the correction vector
template <typename real>
void eigenSolver<real>::calculateCorrectionVector(int iterations,
                                                  const El::DistMatrix<real> &A,
                                                  El::Grid &grid) {
  correctionVector.SetGrid(grid);
  El::Identity(correctionVector, solverOptions.sizeOfTheMatrix,
               solverOptions.numberOfEigenValues);
  for (int j = 0; j < solverOptions.numberOfEigenValues; ++j) {
    if (solverOptions.solver == "davidson") {
      for (int k = 0; k < solverOptions.sizeOfTheMatrix; ++k) {
        // Calculate the correction vector = r/(D-theta*I)
        real denominator =
            1.0 / (eigenValues.GetLocal(j, 0) - A.GetLocal(k, k));
        correctionVector.SetLocal(k, j, residual.GetLocal(k, j) * denominator);
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
  }
}

// Calculate the residual
template <typename real>
bool eigenSolver<real>::calculateResidual(int iterations, El::Grid &grid) {
  // Parameters for GEMM
  real alpha = 1, beta = 0;

  // Initialize the ritz vectors
  El::Zeros(ritzVectors, solverOptions.sizeOfTheMatrix,
            iterations * solverOptions.numberOfEigenValues);

  // Set ranges of V
  El::Range<int> begV(0, solverOptions.sizeOfTheMatrix);
  El::Range<int> endV(0, iterations * solverOptions.numberOfEigenValues);

  // Calculate the ritz vectors
  El::Gemm(El::NORMAL, El::NORMAL, alpha, V(begV, endV), eigenVectors, beta,
           ritzVectors);

  int residualCheck = 0;
  for (int j = 0; j < solverOptions.numberOfEigenValues; ++j) {

    // Set ranges to get jth eigen vectors
    El::Range<int> beg(0, iterations * solverOptions.numberOfEigenValues);
    El::Range<int> end(j, j + 1);

    // Set ranges to get jth ritz vectors
    El::Range<int> begR(0, solverOptions.sizeOfTheMatrix);
    El::Range<int> endR(j, j + 1);

    // Matrix to store AV*s_j, where s is the eigenVector of the reduced problem
    // and Vs the ritz vectors
    El::DistMatrix<real> AVs(grid);
    El::Zeros(AVs, solverOptions.sizeOfTheMatrix, 1);

    // Calculate AV*s_j
    El::Gemm(El::NORMAL, El::NORMAL, alpha, AV, eigenVectors(beg, end), beta,
             AVs);

    // Matrix to store theta*V*s_j
    El::DistMatrix<real> thetaVs(ritzVectors(begR, endR));

    // Calculate theta_j*I*Vs_j
    thetaVs *= eigenValues.GetLocal(j, 0);

    // Calculate residual = AVs - thetaVs
    AVs -= thetaVs;
    double normAVs = El::Nrm2(AVs);
    if (normAVs < solverOptions.tolerence) {
      ++residualCheck;
    }

    // Add the residual vector to the residual matrix
    for (int k = 0; k < solverOptions.sizeOfTheMatrix; ++k) {
      residual.SetLocal(k, j, AVs.GetLocal(k, 0));
    }
  }

  if (residualCheck == solverOptions.numberOfEigenValues)
    return true;
  else
    return false;
}

// Calculate the eigen pairs of the V^TAV problem
template <typename real>
void eigenSolver<real>::subspaceProblem(int iterations,
                                        const El::DistMatrix<real> &A,
                                        El::Grid &grid) {
  // Set ranges of V
  El::Range<int> begV(0, solverOptions.sizeOfTheMatrix);
  El::Range<int> endV(0, iterations * solverOptions.numberOfEigenValues);

  // Initialise T = V^TAV
  El::DistMatrix<real> T(grid);
  El::Zeros(T, iterations * solverOptions.numberOfEigenValues,
            iterations * solverOptions.numberOfEigenValues);

  // Initialize AV
  El::Identity(AV, solverOptions.sizeOfTheMatrix,
               iterations * solverOptions.numberOfEigenValues);

  // Parameters for GEMM
  real alpha = 1, beta = 0;

  // Calculate AV
  El::Gemm(El::NORMAL, El::NORMAL, alpha, A, V(begV, endV), beta, AV);

  // Calculate V^TAV
  El::Gemm(El::TRANSPOSE, El::NORMAL, alpha, V(begV, endV), AV, beta, T);

  // Get the eigen pairs for the reduced problem V^TAV
  El::HermitianEig(El::UPPER, T, eigenValues, eigenVectors);
}

// Expand the search space with the correction vector
template <typename real>
void eigenSolver<real>::expandSearchSpace(int &iterations,
                                          const El::DistMatrix<real> &A,
                                          El::Grid &grid) {
  //Implement Restart part                                          
  if ((iterations * solverOptions.numberOfEigenValues) >= sizeOfSearchSpace) {
    iterations = 0;
    eigenSolver<real>::initialise(grid, A);
    for (int j = 0; j < solverOptions.numberOfEigenValues; ++j) {
      // Append the correction vector to the search space
      for (int k = 0; k < solverOptions.sizeOfTheMatrix; ++k) {
        V.SetLocal(k, j, correctionVector.GetLocal(k, j));
      }
    }
  }
  //If no restart add the correction vector to V
  else {
    for (int j = 0; j < solverOptions.numberOfEigenValues; ++j) {
      // Append the correction vector to the search space
      for (int k = 0; k < solverOptions.sizeOfTheMatrix; ++k) {
        V.SetLocal(k, (iterations)*solverOptions.numberOfEigenValues + j,
                   correctionVector.GetLocal(k, j));
      }
    }
  }
}  // namespace eigenValueSolver

template <typename real>
void eigenSolver<real>::solve(const El::DistMatrix<real> &A, El::Grid &grid) {

  // Set grid sizes of the member matrices and initialize them
  eigenSolver<real>::initialise(grid, A);
  int count = 0;

  for (int iterations = 1; iterations < solverOptions.sizeOfTheMatrix / 2;
       ++iterations) {

    // Solve the subspace problem VTAV
    subspaceProblem(iterations, A, grid);

    // Calculate the residual
    bool tolerenceReached = calculateResidual(iterations, grid);
    if (tolerenceReached == true) return;

    // Calculate the correction vector based on the solver
    calculateCorrectionVector(iterations, A, grid);

    // calculate residual, correction vector and expand the search space
    expandSearchSpace(iterations, A, grid);

    // Set ranges of V
    El::Range<int> begV(0, solverOptions.sizeOfTheMatrix);
    El::Range<int> endV(0,
                        (iterations + 1) * solverOptions.numberOfEigenValues);

    // R matrix for QR factorization
    El::DistMatrix<real> R;
    El::Zeros(R, (iterations + 1) * solverOptions.numberOfEigenValues,
              solverOptions.sizeOfTheMatrix);

    El::DistMatrix<real> Q(V(begV, endV));
    // QR factorization of V
    El::qr::Explicit(Q, R);
    ++count;
  }
}

// explicit instantiations
template class eigenSolver<float>;
template class eigenSolver<double>;
}  // namespace eigenValueSolver
