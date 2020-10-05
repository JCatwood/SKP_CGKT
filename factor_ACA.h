#ifndef FACTOR_ACA_H
#define FACTOR_ACA_H
#include <functional>
#include "factor_SVD.h"
TreeNode comp_UV_ACA(const Eigen::MatrixXd &mat , int row_offset , int col_offset , 
        int blk_sz , int V_num_row , int V_num_col , double abs_tol , int 
        max_num_term , bool normalize);
TreeNode comp_UV_ACA_v2(const Eigen::MatrixXd &mat , int row_offset , int 
	col_offset , int blk_sz , int V_num_row , int V_num_col , double abs_tol , 
	int max_num_term , bool normalize);
TreeNode comp_UV_ACA(std::function<double(int,int)> mat , int row_offset ,
        int col_offset , int blk_sz , int V_num_row , int V_num_col , double abs_tol ,
        int max_num_term , bool normalize);
#endif
