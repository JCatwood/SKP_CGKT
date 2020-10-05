#ifndef FACTOR_SVD_H
#define FACTOR_SVD_H
#include <vector>
#include "Dense"
/*
	The new class stores the number of factors as
	well as the factors' number of rows/columns
	Each column of the Matrix stores one factor 
*/
class TreeNode {
public:
	unsigned int num_term;
	int U_num_row;
	int U_num_col;
	int V_num_row;
	int V_num_col;
	Eigen::MatrixXd U;
	Eigen::MatrixXd V;
	Eigen::MatrixXd uncompress();
	TreeNode(){}
	~TreeNode(){}
};


TreeNode comp_UV_SVD(const Eigen::MatrixXd& mat,
                	        int row_offset,
                	        int col_offset,
                	        int blk_sz,
                	        int V_num_row,
                	        int V_num_col,
                	        double abs_tol,
				unsigned int max_num_term,
				double* sing_val,
                	        double& err);


TreeNode comp_UV_SVD_prescribed(const Eigen::MatrixXd& mat,
                	        int row_offset,
                	        int col_offset,
                	        int blk_sz,
                	        int V_num_row,
                        	int V_num_col,
				const Eigen::MatrixXd& QU,
		                const Eigen::MatrixXd& QV,
               			int U_num_constraint,
		                int V_num_constraint,
                	        double abs_tol,
				unsigned int max_num_term,
				double* sing_val,
                	        double& err);


Eigen::MatrixXd rearrange(const double* mat , int lead_dim , int dim ,
                        int blk_num_row , int blk_num_col);


TreeNode factorize_dense_SVD(Eigen::MatrixXd& mat , 
			int U_num_row , int U_num_col ,
        	        int V_num_row , int V_num_col , 
			double abs_tol , unsigned int max_num_term, 
			double* sing_val , 
			double& err);


void qr_full(Eigen::MatrixXd& A , Eigen::MatrixXd& Q , Eigen::MatrixXd& R);			

void qr_full(Eigen::MatrixXd& A , Eigen::MatrixXd& Q);			


void qr(Eigen::MatrixXd& A , Eigen::MatrixXd& R);


TreeNode TreeNode_add_SVD(const TreeNode& node1 , const TreeNode& node2 ,
        double abs_tol);

int cholesky(Eigen::MatrixXd& A);

int inverse(Eigen::MatrixXd &A);
#endif
