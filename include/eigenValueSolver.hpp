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
  //Groups all the options for the solvers
  struct options {
    // Type of solver used - either "davidson" or "jacobi"
    std::string solver = "davidson";  

    //Tolerence criterion for convergence
    real tolerence = 1e-8; 

    //The number of eigen values to be calculated         
    int numberOfEigenValues = 1; 

    //Size of the input matrix
    int sizeOfTheMatrix = 100;   
  };

  
  //MEMBER VARIABLES

  //Eigenvector matrix of the reduced problem
  El::DistMatrix<real> eigenVectors; 

  //ritzVectors i.e. eigen vectors of the full matrix
  El::DistMatrix<real> ritzVectors; 

  //Eigenvalues
  El::DistMatrix<real, El::VR, El::STAR> eigenValues; 

  //The guess eigenspace
  El::DistMatrix<real> V;  

  // A*V
  El::DistMatrix<real> AV;     

  //The guess eigenspace per iteration
  El::DistMatrix<real> Vsub;   

  //The correction vector
  El::DistMatrix<real> correctionVector; 

  //The residual
  El::DistMatrix<real> residual;

  //Instance of the options struct
  options solverOptions;

  //MEMBER FUNCTIONS
  //This function computes the eigen value-vector pairs for the input matrix
  void solve(const El::DistMatrix<real> &, El::Grid &);

 private:
  //The size of the search space
  real sizeOfSearchSpace; 
  
  // Range to loop over just the required number of eigenvalues in theta
  El::Range<int> begTheta;
  El::Range<int> endTheta;
  

  //This function initialises all the required matrices
   void initialise( El::Grid &, const El::DistMatrix<real> &);

  //Calculates the residual
  bool calculateResidual(int, El::Grid &);

  //Calculates the correction vector
  void calculateCorrectionVector(int, const El::DistMatrix<real> &, El::Grid &);

  //Expands the search space with the correction vector
  void expandSearchSpace(int, const El::DistMatrix<real> &, El::Grid &);

  //Solve the subspace problem i.e. VTAV and eigenvalue/vectors
  void subspaceProblem(int, const El::DistMatrix<real> &, El::Grid &);
};
}  // namespace eigenValueSolver

#endif /* eigenValueSolver_h */
