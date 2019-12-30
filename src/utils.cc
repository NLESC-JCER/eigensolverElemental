//
//  utils.cpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#include "utils.hpp"
#include "eigenValueSolver.hpp"

template <typename real>
void generateDDHermitianMatrix(El::DistMatrix<real> &A) {
  int matrixSize = A.Height();
  for (int i = 0; i < matrixSize; ++i) {
    for (int j = 0; j < matrixSize; ++j) {
      if (i == j)
        A.SetLocal(i, j, i + 21);
      else
        A.SetLocal(i, j, 0.1);
    }
  }
}

// explicit template function definitions
template void generateDDHermitianMatrix(El::DistMatrix<double> &A);
template void generateDDHermitianMatrix(El::DistMatrix<float> &A);
