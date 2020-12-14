#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include "Dense"

std::vector<unsigned int> comp_factor(unsigned int val);

std::function<double(double)> exp_cov_func(double beta);

Eigen::MatrixXd nc_exp_cov_mat(const Eigen::MatrixXd &geom, const int *idx);

#endif
