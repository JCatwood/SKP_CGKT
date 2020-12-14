#ifndef SIMULATIONS_H
#define SIMULATIONS_H
#include <functional>
#include <fstream>
#include "Dense"
void cmp_svd_aca_fm(int s1, int s2);
void cmp_idx_blk_sz(int kernelType, int s1, int s2, int odrType);
void relerr_nterm(int kernelType, int gridType);
void nterm_for_precision(int s1, int s2);
void cholesky_time(int m, int kernelType, int domainType, int k1, int k2, int k3,
        int crt_k1, int crt_k2, int crt_k3, int incr_k1, int incr_k2, int incr_k3,
        double epslBuild, double epslFactor, bool check);
std::function<double(double)> Matern_kernel(double beta, double nu);
void semivariog(const Eigen::MatrixXd &geom, const Eigen::VectorXd &y, 
	double distLB, double binLen, int nbin, Eigen::VectorXd &u, 
	Eigen::VectorXd &v, int *work, int lwork);
void GRF_app(int m, int kernelType, int domainType, std::ofstream &file);
#endif 
