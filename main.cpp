#include <iostream>
#include <functional>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <string>
#include <tr1/cmath>
#include "Dense"
#include "mkl.h"
#include "generate_mat.h"
#include "utility.h"
#include "factor_SVD.h"
#include "factor_ACA.h"
#include "common_base_blk_chol.h"
#include "simulations.h"
#include "simulations_HLIBpro.h"


using namespace std;
using namespace Eigen;


typedef std::chrono::time_point<std::chrono::steady_clock> TimeStamp;

int main()
{
	mkl_set_num_threads(8);
	for(int covType = 1; covType < 2; covType++)
		for(int domainType = 1; domainType < 2; domainType++)
		{
			string domainName, covName;
			if(domainType == 1)
				domainName = "fixed";
			else
				domainName = "expanding";
			if(covType == 1)
				covName = "exp";
			else
				covName = "whittle";
			string fileName = covName + "_" + domainName + ".txt";
			ofstream file(fileName);
			GRF_app(1024, covType, domainType, file);
			file.close();
		}
	return 0;
}

MatrixXd kron_prod(const MatrixXd& mat1 , const MatrixXd& mat2)
{
	int num_row = mat1.rows() * mat2.rows();
	int num_col = mat1.cols() * mat2.cols();
	MatrixXd kron_mat(num_row , num_col);
	for(int i = 0 ; i < mat1.rows() ; i++)
		for(int j = 0 ; j < mat1.cols() ; j++)
			kron_mat.block(i*mat2.rows(),j*mat2.cols(),mat2.rows(),
				mat2.cols()) = mat1(i,j) * mat2;
	return kron_mat;
}


