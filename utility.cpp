#include <cmath>
#include <functional>
#include <vector>
#include <cassert>
#include "Dense"

using namespace std;
using namespace Eigen;

#define ALMOSTZERO 1E-10

/* 
	compute all the int factors of @val
*/ 
vector<unsigned int> comp_factor(unsigned int val)
{
	vector<unsigned int> factor_vec(0);
	if(val > 0)
	{
		#pragma omp parallel for
		for(unsigned int i = 1 ; i <= val ; i++)
			if(val % i == 0)
				#pragma omp critical
				factor_vec.push_back(i);
	}
	return factor_vec;
}

/* return the exponential cov func */
function<double(double)> exp_cov_func(double beta)
{
	assert(beta > 0);
	return [beta](double h){return exp(- h / beta);};
}

/*
	Return non-stationary exponential correlation matrix
	Correlation kernel transformed based on the exponential kernel
	The kernel matrix at each location is a scalar matrix
		representing the range
	The scalar at each location equal to 0.1 + 0.2 * first coordinate value
	@geom should be in the unit square
	@idx match the order of @geom to the order of row/col indices
*/
Eigen::MatrixXd nc_exp_cov_mat(const MatrixXd &geom, const int *idx)
{
	int n = geom.rows();
	MatrixXd covM(n, n);
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
		{
			double sigma1 = geom.row(idx[i])(0) * 0.2 + 0.1;
			double sigma2 = geom.row(idx[j])(0) * 0.2 + 0.1;
			double sigmabar = (sigma1 + sigma2) / 2.0;
			double sigmabarSqr = sqrt(sigmabar);
			double q = (geom.row(idx[i]) - geom.row(idx[j])).norm() /
				sigmabarSqr;
			covM(i, j) = pow(sigma1, 0.25) * pow(sigma2, 0.25) / 
				sigmabarSqr * exp(-q);
		}
	return covM;
}







