#include <iostream>
#include <functional>
#include "Dense"
#include "morton.h"


using namespace std;
using namespace Eigen;


int xy2d (int n, int x, int y); 
void rot(int n, int *x, int *y, int rx, int ry);


MatrixXd gen_dense_cov_mat(const MatrixXd& geom,
		function<double(double,double)> cov_kernel,
		const VectorXi& idx)
{
	size_t n = geom.rows();
	MatrixXd cov_mat(n , n);
	size_t N = n*n;
	#pragma omp parallel for
	for(size_t i = 0 ; i < N ; i++)
		cov_mat(i%n , i/n) = cov_kernel(
		geom(idx(i%n),0) - geom(idx(i/n),0),
		geom(idx(i%n),1) - geom(idx(i/n),1));
	return cov_mat;
}

MatrixXd gen_dense_cov_mat(const MatrixXd& geom,
		function<double(double)> cov_kernel,
		const VectorXi& idx)
{
	size_t n = geom.rows();
	MatrixXd cov_mat(n , n);
	size_t N = n*n;
	#pragma omp parallel for
	for(size_t i = 0 ; i < N ; i++)
		cov_mat(i%n , i/n) = cov_kernel((geom.row(idx(i%n)) - 
					geom.row(idx(i/n))).norm());
	return cov_mat;
}

/* generate m*m points on the regular grid  */
MatrixXd gen_grid_2D_unit_sq_geom(size_t m)
{
	size_t n = m*m;
	MatrixXd geom(n , 2);
	double unit = 1.0/m;
	#pragma omp parallel for
	for(size_t i = 0 ; i < n ; i++)
	{
		geom(i , 0) = (i/m)*unit;
		geom(i , 1) = (i%m)*unit;
	}
	return geom;
}

/* generate m*m points on the perturbed grid  */
MatrixXd gen_rand_2D_unit_sq_geom(size_t m)
{
	size_t n = m*m;
	MatrixXd geom(n , 2);
	double unit = 1.0/m;
	#pragma omp parallel for
	for(size_t i = 0 ; i < n ; i++)
	{
		geom(i , 0) = (i/m)*unit + 
			(((double) rand() / (RAND_MAX))*0.8)*unit;
		geom(i , 1) = (i%m)*unit + 
			(((double) rand() / (RAND_MAX))*0.8)*unit;
	}
	return geom;
}

/* geom should be contained in the 2D unit square */
VectorXi gen_Morton_2D_idx(const MatrixXd& geom)
{
	int n = geom.rows();
	uint_fast64_t encoded_val[n];
	VectorXi idx(n);
	#pragma omp parallel for
	for(int i = 0 ; i < n ; i++)
	{
		idx(i) = i;
		encoded_val[i] = morton2D_64_encode(
			(uint_fast32_t)(geom(i,0)*UINT32_MAX),
			(uint_fast32_t)(geom(i,1)*UINT32_MAX));
	}
	sort(idx.data(),idx.data()+idx.size(),[&encoded_val](int i1, int i2) 
		{return encoded_val[i1] < encoded_val[i2];});
	return idx;
}

/* 
	generate the Z shaped curve for a regular grid 
	input m, where n = m*m
	m should also be a multiple of 2
	assume the orginal order is column oriented
*/
VectorXi gen_Z_shape_2D_idx(int m)
{
	int n = m*m;
	VectorXi idx(n);
	uint_fast64_t encoded_val[n];
	#pragma omp parallel for
	for(int i = 0 ; i < n ; i++)
	{
		idx(i) = i;
		encoded_val[i] = morton2D_64_encode(i%m , i/m);
	}
	sort(idx.data(),idx.data()+idx.size(),[&encoded_val](int i1, int i2) 
		{return encoded_val[i1] < encoded_val[i2];});
	return idx;
}

/* 
	Order geom simply based on the second column
	In the ascending order
*/
Eigen::VectorXi gen_col_maj_idx(const Eigen::MatrixXd &geom)
{
	assert(geom.cols() == 2);
	int n = geom.rows();
	VectorXi idx(n);
	iota(idx.data(), idx.data() + n, 0);
	sort(idx.data(), idx.data() + n, [&geom](int i1, int i2) 
		{return geom(i1, 0) < geom(i2, 0);});
	return idx;
}

/* 
	Generate Hilbert curve order
	s1 >= s2
	s1 should be a power of 2
	2019/10/24
*/
Eigen::VectorXi gen_Hilbert_idx(int s1, int s2)
{
	int n = s1 * s2;
	VectorXi d(n);
	VectorXi idx(n);
	for(int i = 0; i < n; i++)
	{
		int i1 = i / s2;
		int j1 = i % s2;
		d(i) = xy2d(s1, i1, j1);
	}
	iota(idx.data(), idx.data() + n, 0);
	sort(idx.data(), idx.data() + n, [&d](int i1, int i2) 
		{return d[i1] < d[i2];});
	
	return idx;
}

/* 
	Generate S so that 
	A is Toeplitz is the same as
	S^T vec(A) = 0 
	A is of dimension m-by-m
*/
MatrixXd gen_Toeplitz_condi(unsigned int m)
{
	unsigned int num_iter = (m-1)*(m-1);
	MatrixXd S = MatrixXd::Zero(m*m , num_iter);
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < num_iter ; i++)
	{
		unsigned int row_num = i % (m-1);
		unsigned int col_num = i / (m-1);
		unsigned int idx1 = m*col_num + row_num;
		unsigned int idx2 = m*(col_num+1) + (row_num+1);
		S(idx1 , i) = 1;
		S(idx2 , i) = -1;
	}
	return S;
}


/*
	Generate S so that
	A is banded with depth q is the same as
	S^T vec(A) = 0
	A is of dimension m-by-m
*/
MatrixXd gen_banded_condi(unsigned int m , unsigned int q)
{
	unsigned int num_iter = (m-1-q) * (m-q) / 2;
	MatrixXd S = MatrixXd::Zero(m*m , num_iter * 2);
	unsigned tmp_int = m-q;
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < num_iter ; i++)
	{
		unsigned int row_num = i % tmp_int;
                unsigned int col_num = i / tmp_int;
                if(row_num <= col_num)
                {
                        col_num = tmp_int - col_num - 2;
                        row_num = tmp_int - row_num - 1;
                }
		unsigned int idx1 = m*col_num + row_num + q;
		unsigned int idx2 = m*(row_num+q) + col_num;
		S(idx1 , i) = 1;
		S(idx2 , num_iter+i) = 1;
	}
	return S;
}


/* 
	Generate S so that 
	A is banded Toeplitz with band depth q is the same as
	S^T vec(A) = 0 
	A is of dimension m-by-m
*/
MatrixXd gen_Toeplitz_banded_condi(unsigned int m , unsigned int q)
{
	unsigned int num_iter_banded = m-1-q;
	unsigned int num_iter_Toeplitz = (m-1) * (m-1);
	MatrixXd S = MatrixXd::Zero(m*m , num_iter_banded*2 + num_iter_Toeplitz);
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < num_iter_Toeplitz ; i++)
	{
		unsigned int row_num = i % (m-1);
		unsigned int col_num = i / (m-1);
		unsigned int idx1 = m*col_num + row_num;
		unsigned int idx2 = m*(col_num+1) + (row_num+1);
		S(idx1 , i) = 1;
		S(idx2 , i) = -1;
	}
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < num_iter_banded ; i++)
	{
		unsigned int idx1 = i+1+q;
		unsigned int idx2 = m*(i+q+1);
		S(idx1 , num_iter_Toeplitz+i) = 1;
		S(idx2 , num_iter_Toeplitz+num_iter_banded+i) = 1;
	}

	return S;
}

/*
	For Hilbert curve
	2019/10/24
*/
void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

/*
	For Hilbert curve
	2019/10/24
*/
int xy2d (int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(n, &x, &y, rx, ry);
    }
    return d;
}













