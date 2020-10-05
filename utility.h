#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include "Dense"

unsigned int largest_factor_below_sqrt (unsigned int val);

std::vector<int> comp_factor_below_sqrt(unsigned int val);

std::vector<unsigned int> comp_factor(unsigned int val);

std::function<double(double)> sph_cov_func(double a);

std::function<double(double)> exp_cov_func(double beta);

std::function<double(double)> nugg_exp_cov_func(double beta, double nugget);

std::function<double(double,double)> deform_exp_cov_func(double beta);

Eigen::MatrixXd nc_exp_cov_mat(const Eigen::MatrixXd &geom, const int *idx);

void weak_H_tree(unsigned int r , std::vector<unsigned int>& row_offset_vec,
        std::vector<unsigned int>& col_offset_vec, 
	std::vector<unsigned int>& blk_sz_vec);
#endif
