//
//  eigenValueSolver.hpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#ifndef eigenValueSolver_h
#define eigenValueSolver_h

#include <El.hpp>
#include <El/blas_like/level3.hpp>
#include <El/lapack_like/euclidean_min.hpp>
#include <El/lapack_like/factor.hpp>
#include <El/lapack_like/spectral/HermitianEig.hpp>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <stdexcept>

namespace eigenValueSolver {

template <typename real>
class eigenSolver {

 public:
  /**
   * Groups all the options for the solvers
   */
  struct options {
    std::string solver = "davidson";  // either davidson or jacobi
    real tolerence = 1e-8;            /** tolerence for convergence*/
    int numberOfEigenValues = 1; /** number of eigen values to be calculated*/
    int sizeOfTheMatrix = 100;   /**Size of the input matrix*/
  };

  El::DistMatrix<real> eigenVectors; /**s stores the current eigenvector
                                        matrix*/
  El::DistMatrix<real> eigenVectorsFull; /**s stores the current eigenvector
                                                                              matrix*/
  El::DistMatrix<real, El::VR, El::STAR> eigenValues; /**vector of eigen
                                                         values*/

  options solverOptions;
  /**
   * This function computes the eigen value-vector pairs for the input matrix
   */
  void solve(const El::DistMatrix<real> &, El::Grid &);

 private:
  int columnsOfSearchSpace = solverOptions.numberOfEigenValues *
                             2; /**The columns of the search space*/
  /*************Initialise all the matrices necessary**************/
  El::DistMatrix<real> searchSpace;      /**Matrix that holds the search space*/
  El::DistMatrix<real> searchSpacesub;   /**Matrix that holds the search space*/
  El::DistMatrix<real> correctionVector; /**a matrix that holds the k guess
                                            eigen vectors each of size nx1*/
  El::DistMatrix<real, El::VR, El::STAR> eigenValues_old; /**vector of eigen
                                                             values*/

  // Range to loop over just the required number of eigenvalues in theta
  El::Range<int> begTheta;
  El::Range<int> endTheta;
  // El::Range<int> begTheta{0, solverOptions.numberOfEigenValues};
  // El::Range<int> endTheta{0, 1};

  /**
   * This function initialises all the required matrices
   */
  void initialise(El::Grid &);

  /**
   *Calculates the residual
   */
  void calculateResidual();

  /**
   *Calculates the correction vector
   */
  void calculateCorrectionVector();

  /**
   *Expands the search space with the correction vector
   */
  void expandSearchSpace(int, const El::DistMatrix<real> &, El::Grid &);

  /**
   * solve the subspace problem i.e. VTAV and eigenvalue/vectors
   */
  void subspaceProblem(int, const El::DistMatrix<real> &, El::Grid &);
};
}  // namespace eigenValueSolver

#endif /* eigenValueSolver_h */
