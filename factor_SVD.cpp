#include <iostream>
#include "Dense"
#include "mkl.h"
#include "factor_SVD.h"


using namespace std;
using namespace Eigen;


/* 
	Compute the Kron prod factors for a matrix block
	The block is assumed to be square
	Input:
		the whole matrix
		the row/col offset
		the block size
		the trunc level
		the row/col number of the right factor
	Output:
   		factors U and V
		singular values of the rearranged matrix
		Frobenius norm of the error
*/
TreeNode comp_UV_SVD(const MatrixXd& mat,
			int row_offset,
			int col_offset,
			int blk_sz,
			int V_num_row,
			int V_num_col,
			double abs_tol,
			unsigned int max_num_term,
			double* sing_val,
			double& err)
{
	int lead_dim = mat.rows();
	int U_num_row = blk_sz / V_num_row;
	int U_num_col = blk_sz / V_num_col; 
	const double* first_coef_pt = mat.data() +
		lead_dim*col_offset + row_offset;
	MatrixXd mat_rearranged = rearrange(first_coef_pt , 
			lead_dim , blk_sz , V_num_row , V_num_col);
	return factorize_dense_SVD(mat_rearranged , U_num_row , U_num_col ,
	V_num_row , V_num_col , abs_tol , max_num_term , sing_val , err);
}


/*
	Rearrange the matrix based on the block size
	Assume the matrix is square
	Column major is assumed
	Input:
		the input matrix as an array of double
		the leading dimension of the input matrix
		the dimension of the input matrix
		the number of row/col of the matrix block for rearrangement 
*/
MatrixXd rearrange(const double* mat , int lead_dim , int dim , 
			int blk_num_row , int blk_num_col)
{
	int num_coef = dim*dim;
	int num_blk = num_coef / blk_num_row / blk_num_col;
	int blk_sz = blk_num_row * blk_num_col;
	int num_blk_per_col = dim / blk_num_row;
	MatrixXd mat_rearrange(num_blk ,blk_sz);
	#pragma omp parallel for
	for(int i = 0 ; i < num_coef ; i++)
	{
		int row_num = i % dim;
		int col_num = i / dim;
		int blk_row_num = row_num / blk_num_row;
		int blk_col_num = col_num / blk_num_col;
		int row_num_in_blk = row_num % blk_num_row;
		int col_num_in_blk = col_num % blk_num_col;
		int row_num_new = num_blk_per_col*blk_col_num + blk_row_num;
		int col_num_new = blk_num_row*col_num_in_blk + row_num_in_blk;
		mat_rearrange(row_num_new , col_num_new) = 
			*(mat + lead_dim*col_num + row_num);
	}
	return mat_rearrange;
}


/*
	Do SVD directly to the input matrix
	Store the info in a TreeNode and return it 
	The input matrix is usually the result from
	the rearrange function
	mat.rows() = U_num_row*U_num_col
	mat.cols() = V_num_row*V_num_col

	Fixed two bugs on Apr 18th, 2019
*/
TreeNode factorize_dense_SVD(MatrixXd& mat , int U_num_row , int U_num_col ,
		int V_num_row , int V_num_col , double abs_tol , 
		unsigned int max_num_term , 
		double* sing_val , double& err)
{
	unsigned int mat_num_row = mat.rows();
	unsigned int mat_num_col = mat.cols();
	unsigned int num_sing_val = min(mat_num_row , mat_num_col);
	MatrixXd U(mat_num_row,num_sing_val);
	MatrixXd V_trans(num_sing_val,mat_num_col);
	double superb[num_sing_val-1];
	LAPACKE_dgesvd(LAPACK_COL_MAJOR , 'S' , 'S' , mat_num_row , 
		mat_num_col, mat.data() , mat_num_row , sing_val , U.data() , 
		mat_num_row , V_trans.data() , num_sing_val , superb);
	TreeNode node;
	node.num_term = 0;
	if(max_num_term == 0)
		max_num_term = num_sing_val;
	while(sing_val[node.num_term] > abs_tol && 
		node.num_term < max_num_term && 
		node.num_term < num_sing_val)
		node.num_term++;
	node.U.resize(mat_num_row , node.num_term);
	node.V.resize(mat_num_col , node.num_term);
	for(unsigned int i = 0 ; i < node.num_term ; i++)
	{
		double s = sqrt(sing_val[i]);
		node.U.col(i) = U.col(i) * s;
		node.V.col(i) = V_trans.row(i).array() * s;
	}
	node.U_num_row = U_num_row;
	node.U_num_col = U_num_col;
	node.V_num_row = V_num_row;
	node.V_num_col = V_num_col;
	err = 0.0;
	for(unsigned int i = node.num_term ; i < num_sing_val ; i++)
		err += sing_val[i]*sing_val[i];
	err = sqrt(err);
	return node;
}

	
/* 
	Compute the Kron prod factors for a matrix block
		subject to a prescribed structure defined by QU and QV
	The block is assumed to be square
	Input:
		the whole matrix
		the row/col offset
		the block size
		the trunc level
		the row/col number of the right factor
		the orthogonal matrices QU and QV
	Output:
   		factors U and V
		singular values of the rearranged matrix
		Frobenius norm of the error
*/
TreeNode comp_UV_SVD_prescribed(
		const MatrixXd& mat,
		int row_offset,
		int col_offset,
		int blk_sz,
		int V_num_row,
		int V_num_col,
		const MatrixXd& QU,
		const MatrixXd& QV,
		int U_num_constraint,
		int V_num_constraint,
		double abs_tol,
		unsigned int max_num_term,
		double* sing_val,
		double& err)
{
	int lead_dim = mat.rows();
	int U_num_row = blk_sz / V_num_row;
	int U_num_col = blk_sz / V_num_col; 
	int QU_dim = QU.rows();
	int QV_dim = QV.rows();
	const double* first_coef_pt = mat.data() +
		lead_dim*col_offset + row_offset;
	MatrixXd mat_rearranged = rearrange(first_coef_pt , 
			lead_dim , blk_sz , V_num_row , V_num_col);
	MatrixXd mat_prescribed = QU.block(0 , U_num_constraint ,
		QU_dim , QU_dim - U_num_constraint).transpose() *
		mat_rearranged * QV.block(0 , V_num_constraint ,
		QV_dim , QV_dim - V_num_constraint); 
	auto node = factorize_dense_SVD(mat_prescribed , U_num_row , U_num_col ,
	V_num_row , V_num_col , abs_tol , max_num_term , sing_val , err);
	node.U = QU.block(0 , U_num_constraint ,
		QU_dim , QU_dim - U_num_constraint) * node.U;
	node.V = QV.block(0 , V_num_constraint ,
		QV_dim , QV_dim - V_num_constraint) * node.V;
	return node;
}


MatrixXd TreeNode::uncompress()
{
	int num_row = U_num_row * V_num_row;
	int num_col = U_num_col * V_num_col;
	MatrixXd mat_recover(num_row , num_col);
	mat_recover.setZero();
	int N = num_row * num_col;
	#pragma omp parallel for
	for(int i = 0 ; i < N ; i++)
	{
		int row_num = i%num_row;
		int col_num = i/num_row;
		int U_row_num = row_num / V_num_row;
		int U_col_num = col_num / V_num_col;
		int V_row_num = row_num % V_num_row;
		int V_col_num = col_num % V_num_col;
		int U_mat_row_num = U_row_num + U_col_num * U_num_row;
		int V_mat_row_num = V_row_num + V_col_num * V_num_row;
		for(unsigned int j = 0 ; j < num_term; j++)
			mat_recover(row_num,col_num) += U(U_mat_row_num,j) *
				V(V_mat_row_num,j);
	}
	return mat_recover;
}


/*
	QR factorization
	Assume A.rows() >= A.cols()
	Returns the full Q factor
*/
void qr_full(MatrixXd& A , MatrixXd& Q , MatrixXd& R)
{
        unsigned int m = A.rows();
        unsigned int n = A.cols();
	if(m < n)
		throw("Number of rows should be larger than "
		"the number of columns for the QR decomp\n");
        double tau[m];
        R = MatrixXd(n , n);
	if(LAPACKE_dgeqrf(LAPACK_COL_MAJOR , m , n  , A.data() , m , tau))
		throw("LAPACKE_dgeqrf has non zero return value\n");
	#pragma omp parallel for
        for(unsigned int i = 0 ; i < n*n ; i++)
	{
		unsigned int row_num = i%n;
		unsigned int col_num = i/n;
		R(row_num , col_num) = 
			col_num < row_num ? 0 : A(row_num , col_num);
	}
	Q = MatrixXd::Zero(m , m);
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < m*n ; i++)
	{
		unsigned int row_num = i%m;
		unsigned int col_num = i/m;
		Q(row_num , col_num) = A(row_num , col_num);
	}
	if(LAPACKE_dorgqr(LAPACK_COL_MAJOR , m , m , n , Q.data() , m , tau))
		throw("LAPACKE_dorgqr has none zero return value\n");
}


/*
	QR factorization
	Assume A.rows() >= A.cols()
	Returns only the full Q factor
*/
void qr_full(MatrixXd& A , MatrixXd& Q)
{
        unsigned int m = A.rows();
        unsigned int n = A.cols();
	if(m < n)
		throw("Number of rows should be larger than "
		"the number of columns for the QR decomp\n");
        double tau[m];
	if(LAPACKE_dgeqrf(LAPACK_COL_MAJOR , m , n  , A.data() , m , tau))
		throw("LAPACKE_dgeqrf has non zero return value\n");
	Q = MatrixXd::Zero(m , m);
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < m*n ; i++)
	{
		unsigned int row_num = i%m;
		unsigned int col_num = i/m;
		Q(row_num , col_num) = A(row_num , col_num);
	}
	if(LAPACKE_dorgqr(LAPACK_COL_MAJOR , m , m , n , Q.data() , m , tau))
		throw("LAPACKE_dorgqr has none zero return value\n");
}


/*
	QR factorization
	Assume A.rows() >= A.cols()
	Returns the reduced Q factor in A
	2019/06/12
*/
void qr(MatrixXd& A , MatrixXd& R)
{
        unsigned int m = A.rows();
        unsigned int n = A.cols();
	if(m < n)
		throw("Number of rows should be larger than "
		"the number of columns for the QR decomp\n");
        double tau[m];
        R = MatrixXd(n , n);
	if(LAPACKE_dgeqrf(LAPACK_COL_MAJOR , m , n  , A.data() , m , tau))
		throw("LAPACKE_dgeqrf has non zero return value\n");
	#pragma omp parallel for
        for(unsigned int i = 0 ; i < n*n ; i++)
	{
		unsigned int row_num = i%n;
		unsigned int col_num = i/n;
		R(row_num , col_num) = 
			col_num < row_num ? 0 : A(row_num , col_num);
	}
	if(LAPACKE_dorgqr(LAPACK_COL_MAJOR , m , n , n , A.data() , m , tau))
		throw("LAPACKE_dorgqr has none zero return value\n");
}


int cholesky(MatrixXd& A)
{
        int n = A.rows();
        int success_code = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, A.data(), n);
	A.triangularView<StrictlyUpper>().setZero();
        return success_code;
}


/*
	wrapper for lower tri inverse
	2019/07/14
*/
int inverse(MatrixXd &A)
{
	return LAPACKE_dtrtri(LAPACK_COL_MAJOR , 'L' , 'N' , A.rows() , 
		A.data() , A.rows());
}
