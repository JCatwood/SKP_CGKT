#ifndef GENERATE_MAT_H
#define GENERATE_MAT_H
#include <functional>
#include "Dense"


Eigen::MatrixXd gen_dense_cov_mat(const Eigen::MatrixXd& geom,
                	std::function<double(double,double)> cov_kernel,
			const Eigen::VectorXi& idx);

Eigen::MatrixXd gen_dense_cov_mat(const Eigen::MatrixXd& geom,
                	std::function<double(double)> cov_kernel,
			const Eigen::VectorXi& idx);

Eigen::MatrixXd gen_grid_2D_unit_sq_geom(size_t m);

Eigen::MatrixXd gen_rand_2D_unit_sq_geom(size_t m);

Eigen::VectorXi gen_Morton_2D_idx(const Eigen::MatrixXd& geom);

Eigen::VectorXi gen_Z_shape_2D_idx(int m);

Eigen::VectorXi gen_col_maj_idx(const Eigen::MatrixXd &geom);

Eigen::VectorXi gen_Hilbert_idx(int s1, int s2);

Eigen::MatrixXd gen_banded_condi(unsigned int m , unsigned int q);

Eigen::MatrixXd gen_Toeplitz_condi(unsigned int m);

Eigen::MatrixXd gen_Toeplitz_banded_condi(unsigned int m , unsigned int q);
#endif
