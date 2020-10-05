#include <cmath>
#include <functional>
#include <vector>
#include <cassert>
#include "Dense"

using namespace std;
using namespace Eigen;

#define ALMOSTZERO 1E-10

unsigned int largest_factor_below_sqrt(unsigned int val)
{
	unsigned int val_sqrt = sqrt(val);
	for(int i = val_sqrt ; i > 0 ; i--)
		if(val % i == 0)
			return i;
	return 0;
}

/* 
	compute the int factors of val that are no bigger than sqrt(val) 
	factors are stored ascending
	2019/06/21
*/
vector<int> comp_factor_below_sqrt(unsigned int val)
{
	vector<int> factor_vec(0);
	int val_sqrt = sqrt(val);
	for(int i = 1 ; i <= val_sqrt ; i++)
		if(val % i == 0)
			factor_vec.push_back(i);
	return factor_vec;
}

/* 
	compute all the int factors of val
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

/* return the spherical cov func */
function<double(double)> sph_cov_func(double a)
{
	assert(a > 0);
	return [a](double h){double h_tilde = h / a;
		return h > a ? 0 : 
		1.0 - 1.5 * h_tilde + 0.5 * h_tilde*h_tilde*h_tilde;};
}

/* return the exponential cov func */
function<double(double)> exp_cov_func(double beta)
{
	assert(beta > 0);
	return [beta](double h){return exp(- h / beta);};
}

/* return the exponential cov func with nuggest */
function<double(double)> nugg_exp_cov_func(double beta, double nugget)
{
	assert(beta > 0);
	assert(nugget > 0);
	return [beta, nugget](double h){return h > ALMOSTZERO ? exp(-h/beta) : 1.0+
		nugget;};
}

/* 
	return a deformed 2D exp cov func 
	the PD mat is (1 0.5 ; 0.5 1) X beta^2
*/
function<double(double,double)> deform_exp_cov_func(double beta)
{
	assert(beta > 0);
	return [beta](double h1 , double h2){return exp(- 
		sqrt((h1*h1 + h1*h2 + h2*h2)) / beta);};
}

/*
	Return non-stationary correlation matrix
	Correlation kernel transformed based on the exponential kernel
	The kernel matrix at each location is a scalar matrix
		representing the range
	The scalar at each location equal to 0.1 + 0.2 * first coordinate value
	geom should be in the unit square
	2019/10/24
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


/* build the weak hierarchical tree */
void weak_H_tree(unsigned int r , vector<unsigned int>& row_offset_vec,
	vector<unsigned int>& col_offset_vec, vector<unsigned int>& blk_sz_vec)
{
	unsigned int n = 1 << (2*r);
	unsigned int m = 1 << r;
	row_offset_vec.resize(m-1);
	col_offset_vec.resize(m-1);
	blk_sz_vec.resize(m-1);
	int idx = 0;
	for(unsigned int q = 0 ; q < r ; q++)
	{
		unsigned int blk_sz = n >> (q+1);
		for(int i = 0 ; i < (1 << q) ; i++)
		{
			row_offset_vec[idx] = (2*i+1) * blk_sz;
			col_offset_vec[idx] = 2*i * blk_sz;
			blk_sz_vec[idx] = blk_sz;
			idx++;
		}
	}
}







