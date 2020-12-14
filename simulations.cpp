#include <iostream>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <tr1/cmath>
#include <functional>
#include "Dense"
#include "mkl.h"
#include "generate_mat.h"
#include "utility.h"
#include "factor_SVD.h"
#include "factor_ACA.h"
#include "common_base_blk_chol.h"

using namespace std;
using namespace Eigen;

typedef std::chrono::time_point<std::chrono::steady_clock> TimeStamp;
void klrM_vec_mult(const KLR_mat &klrM, int n, const double *x, double *y, 
	double *work, int lwork);
void gaussian_vec(double *x, int n);

/*
	The simulation for comparing the times and relative errors of the
	  SVD framework and the ACA framework
*/
void cmp_svd_aca_fm(int s1, int s2)
{
	MatrixXd geom(s1*s2, 2);
	double unit = 1.0 / (double) s2;
	for(int i = 0; i < s1*s2; i++)
	{
		geom(i, 0) = (i / s2) * unit;
		geom(i, 1) = (i % s2) * unit;
	}

	auto kernel = exp_cov_func(0.3);

	VectorXi idx(s1*s2);
	iota(idx.data(), idx.data()+s1*s2, 0);

	MatrixXd covM = gen_dense_cov_mat(geom, kernel, idx);
	double covMNorm = covM.norm();

	TimeStamp start, end;
	double timeSVD, timeACA;

	double *singVal = new double[s1*s2];
	double err;
	start = std::chrono::steady_clock::now();
	TreeNode nodeSVD = comp_UV_SVD(covM, 0, 0, s1*s2, s2, s2, 1e-2, 0, singVal,
		err);
	end = std::chrono::steady_clock::now();
        timeSVD = std::chrono::duration<double> (end-start).count();
	double errSVD = err / covMNorm;
	cout << errSVD << " " << nodeSVD.num_term << " " << timeSVD << endl;
	
	function<double(int, int)> kernelWrap = [&geom, &kernel, &idx](int i, int j){
		return kernel((geom.row(i) - geom.row(j)).norm());};
	start = std::chrono::steady_clock::now();
	TreeNode nodeACA = comp_UV_ACA(kernelWrap, 0, 0, s1*s2, s2, s2, 1e-5, 0, 
		true);
	end = std::chrono::steady_clock::now();
        timeACA = std::chrono::duration<double> (end-start).count();
	MatrixXd covMACA = nodeACA.uncompress();
	double errACA = (covM - covMACA).norm() / covMNorm;
	cout << errACA << " " << nodeACA.num_term << " " << timeACA << endl;
}

/*
	The simulation on using different indexing methods and base block sizes
*/
void cmp_idx_blk_sz(int kernelType, int s1, int s2, int odrType)
{
	MatrixXd geom(s1*s2, 2);
	double unit = 1.0 / (double) s2;
	for(int i = 0; i < s1*s2; i++)
	{
		geom(i, 0) = (i / s2) * unit;
		geom(i, 1) = (i % s2) * unit;
	}

	VectorXi idx;
	if(odrType == 1)
	{
		idx.resize(s1*s2);
		iota(idx.data(), idx.data()+s1*s2, 0);
	}
	if(odrType == 2)
	{
		idx = gen_Morton_2D_idx(geom);
	}
	if(odrType == 3)
	{
		idx = gen_Hilbert_idx(s1, s2);
	}

	MatrixXd covM;
	if(kernelType == 1)
	{
		auto kernel = exp_cov_func(0.3);
		covM = gen_dense_cov_mat(geom, kernel, idx);
	}else
	{
		covM = nc_exp_cov_mat(geom, idx.data());
	}

	auto factors = comp_factor(s1 * s2);
	double *singVal = new double[s1 * s2];
	double err;
	double covMNorm = covM.norm();
	double thres = covMNorm * (1.0 - 1e-5);
	thres = thres * thres;
	for(auto m2 : factors)
		for(auto n2 : factors)
		{
			TreeNode nodeSVD = comp_UV_SVD(covM, 0, 0, s1*s2, m2, n2,
				1e-2, 0, singVal, err);
			int nterm = 0;
			double ssq = 0.0;
			while(ssq < thres)
			{
				ssq += singVal[nterm] * singVal[nterm];
				nterm++;
			}
			double mem = (m2*n2 + s1*s2*s1*s2 / m2 / n2) * nterm;
			cout << m2 << " " << n2 << " " << nterm << " " << mem << endl;
		}
	
	delete[] singVal;
}

/*
	The simulation on the relative error v.s. the number of Kronecker 
	  factors in the SKP representation
*/
void relerr_nterm(int kernelType, int gridType)
{
	double beta1 = 0.1;
	double beta2 = 0.3;
	int s1 = 32;
	int s2 = 32;
	MatrixXd geom(s1*s2, 2);
	double unit = 1.0 / (double) s2;
	mt19937 gen(0);
	uniform_real_distribution<> dist(0.0, 2.0);
	double unitRnd = unit * 0.4;
	if(gridType == 1)
	{
		for(int i = 0; i < s1*s2; i++)
		{
			geom(i, 0) = (i / s2) * unit;
			geom(i, 1) = (i % s2) * unit;
		}
	}else
	{
		for(int i = 0; i < s1*s2; i++)
		{
			geom(i, 0) = (i / s2) * unit + unitRnd * dist(gen);
			geom(i, 1) = (i % s2) * unit + unitRnd * dist(gen);
		}
	}
	MatrixXd covM;
	VectorXi idx(s1*s2);
	iota(idx.data(), idx.data() + s1*s2, 0);
        if(kernelType == 1)
        {
                auto kernel = exp_cov_func(0.3);
                covM = gen_dense_cov_mat(geom, kernel, idx);
        }else
	{
		auto kernelWrap = [beta1, beta2, &geom](int i, int j){
			double betaI = beta1 + (beta2 - beta1) * geom(i, 0);
			double betaJ = beta1 + (beta2 - beta1) * geom(j, 0);
			double dist = (geom.row(i) - geom.row(j)).squaredNorm();
			dist = 2 * dist / (betaI * betaI + betaJ * betaJ);
			dist = sqrt(dist);
			double coef = 2 * betaI * betaJ / (betaI * betaI + 
				betaJ * betaJ);
			coef = sqrt(coef);
			return coef * exp(-dist);
		};

		int n = s1 * s2;
		covM.resize(n, n);
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++)
				covM(i, j) = kernelWrap(i, j);
	}
	
	double *singVal = new double[s1*s2];
	double err;
	TreeNode nodeSVD = comp_UV_SVD(covM, 0, 0, s1*s2, s2, s2, 1e-2, 0, singVal, 
		err);
	
	double normSqK = 0.0;
	double *normRel = new double[s1*s2];
	for_each(singVal, singVal + s1*s2, [](double &x){x = x * x;});
	double normTtl = accumulate(singVal, singVal + s1*s2, 0.0);
	for(int i = 0; i < s1*s2; i++)
	{
		normSqK += singVal[i];
		normRel[i] = sqrt(max(normTtl - normSqK, 0.0) / normTtl);
		cout << normRel[i] << " ";
	}
	cout << endl;

	delete[] singVal;
	delete[] normRel;
}

/*
	The simulation testing the number of terms needed in the SKP 
	  representation to reach a certain accuracy
*/
void nterm_for_precision(int s1, int s2)
{
	vector<double> precVec {0.1, 0.01, 1e-3, 1e-4, 1e-5};
	MatrixXd geom(s1*s2, 2);
	double unit = 1.0 / (double) s2;
	double unitRnd = unit * 0.4;
	mt19937 gen(0);
	uniform_real_distribution<> dist(0.0, 2.0);
	for(int i = 0; i < s1*s2; i++)
	{
		geom(i, 0) = (i / s2) * unit + unitRnd * dist(gen);
		geom(i, 1) = (i % s2) * unit + unitRnd * dist(gen);
	}

	MatrixXd covM;
	VectorXi idx(s1*s2);
	iota(idx.data(), idx.data() + s1*s2, 0);
        auto kernel = exp_cov_func(0.3);
        covM = gen_dense_cov_mat(geom, kernel, idx);

	double *singVal = new double[s1*s2];
	double err;
	TreeNode nodeSVD = comp_UV_SVD(covM, 0, 0, s1*s2, s2, s2, 1e-2, 0, singVal, 
		err);
	for_each(singVal, singVal + s1*s2, [](double &x){x = x * x;});
	double normTtl = accumulate(singVal, singVal + s1*s2, 0.0);
	for_each(precVec.begin(), precVec.end(), [&normTtl](double &x)
		{x = (1.0 - x * x) * normTtl;});
	double normSqK = 0.0;
	int k2 = 0;
	for(int i = 0; i < s1*s2; i++)
	{
		normSqK += singVal[i];
		while(k2 < precVec.size() && normSqK > precVec[k2])
		{
			cout << i+1 << " ";
			k2++;
		}
		if(k2 == precVec.size())
			break;
	}
	cout << endl;
}

/*
	The simulation for timing the Cholesky factorization under the SKP
	@m the 2D grid's dimension, also the dimension of the base blocks
	@kernelType: 1 -> stationary exp 2 -> non-stationary exp
	@domainType: 1 -> fixed 2 -> expanding
	@k1, @k2, @k3 the max number of basis of covM, L, Linv, respectively 
	@crt_k1, @crt_k2, @crt_k3 the current max number of basis of 
	  covM, L, Linv, respectively 
	@incr_k1, @incr_k2, @incr_k3 the increments after correction for 
	  the max number of basis of covM, L, Linv, respectively 
	@epslBuild, the tolerance for discovering new basis during building the
	  SKP
	@epslFactor, the tolerance for discovering new basis during factorization
	@check, the Cholesky factor used for checking the relative error. 
	  Set to NULL if checking is not needed
*/
void cholesky_time(int m, int kernelType, int domainType, int k1, int k2, int k3,
	int crt_k1, int crt_k2, int crt_k3, int incr_k1, int incr_k2, int incr_k3,
	double epslBuild, double epslFactor, bool check)
{
	int n = m * m;
	stringstream buf;
	// geometry
	MatrixXd geom(n, 2);
	double unit = 1.0 / (double) m;
	for(int i = 0; i < n; i++)
	{
		geom(i, 0) = (i / m) * unit;
		geom(i, 1) = (i % m) * unit;
	}
	// beta
	double beta1, beta2;
	if(domainType == 1)
	{
		beta1 = 0.03;
		beta2 = 0.1;
	}else
	{
		beta1 = 0.03 / double(m / 32);
		beta2 = 0.1 / double(m / 32);
	}
	// kernel visitor
	function<double(int, int)> visitor;
	if(kernelType == 1)
        {
                auto kernel = exp_cov_func(beta2);
		visitor = [&geom, &kernel](int i, int j){
			return kernel((geom.row(i) - geom.row(j)).norm());};
        }else
        {
                visitor = [beta1, beta2, &geom](int i, 
			int j){
                        double betaI = beta1 + (beta2 - beta1) * (1.0 - geom(i, 0));
                        double betaJ = beta1 + (beta2 - beta1) * (1.0 - geom(j, 0));
                        double dist = (geom.row(i) - geom.row(j)).squaredNorm();
                        dist = 2 * dist / (betaI * betaI + betaJ * betaJ);
                        dist = sqrt(dist);
                        double coef = 2 * betaI * betaJ / (betaI * betaI +
                                betaJ * betaJ);
                        coef = sqrt(coef);
                        return coef * exp(-dist);
                };
        }
	// KLR for the correlation matrix
	TreeNode treeNode = comp_UV_ACA(visitor, 0, 0, n, m, m, epslBuild, 0, true);
	buf << treeNode.num_term << " ";
	KLR_mat klrM(k1, crt_k1, m, m);
	if(crt_k1 < treeNode.num_term)
	{
		cout << "k1 is smaller than the initial number of terms in the "
			"correlation matrix" << endl;
		exit(1);
	}
	klrM.num_term = treeNode.num_term;
        klrM.U.block(0,0,n,klrM.num_term) = treeNode.U;
        klrM.V.block(0,0,n,klrM.num_term) = treeNode.V;
        for(int i = 0 ; i < klrM.num_term ; i++)
        {
                double v_norm = klrM.V.col(i).norm();
                klrM.V.col(i) = klrM.V.col(i) / v_norm;
                klrM.U.col(i) = klrM.U.col(i) * v_norm;
        }
	// KLR for the Cholesky factor and its inverse
	// memory allocated in the common-base block Cholesky routine
	KLR_mat klrChol;
	KLR_mat klrInv;
	vector<MatrixXd> LDiag;
	// common-base block Cholesky
	TimeStamp start, end;
	start = std::chrono::steady_clock::now();
	cb_blk_chol_v2(klrM, klrChol, klrInv, LDiag, k2, k3, crt_k2, crt_k3, 
		incr_k1, incr_k2, incr_k3, epslFactor, NULL);
	end = std::chrono::steady_clock::now();
	double time_used = std::chrono::duration<double> (end-start).
		count();
	buf << klrM.num_term << " " << klrChol.num_term << " " << 
		klrInv.num_term << " " << time_used << " ";
	// check accuracy
	if(check)
	{
		MatrixXd covM(n, n);
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++)
				covM(i, j) = visitor(i, j);
		int code = cholesky(covM);
		if(code != 0)
		{
			cout << endl << "Cholesky failed for the check correlation "
				"matrix" << endl;
			exit(1);
		}
		MatrixXd test = klrChol.uncompress();
                for(int i = 0; i < m; i++)
                        test.block(i*m, i*m, m, m) = LDiag[i];
		buf << (test - covM).norm() / covM.norm() << " ";
	}
	cout << buf.str() << endl;
}

std::function<double(double)> Matern_kernel(double beta, double nu)
{
	assert(beta > 0.0);
	assert(nu > 0.0);
	assert(nu < 100.0);
	double c = pow(2.0, 1.0 - nu) / tgamma(nu);

	return [beta, nu, c](double h){return h < 1e-8 ? 1.0 + 1e-2 : 
		c * pow(h / beta, nu) * std::tr1::cyl_bessel_k(nu, h / beta);};
}

/*
	Compute the empirical semivariogram by sampling. Input:
          @geom n X 2, the locations of spatial variables
          @y values of spatial variables
          @distLB the lower bound of the distance when computing the empirical 
            semivariogram
          @binLen the length of each bin of distance
          @nLen number of bins for the distance
          @work temporary workspace
          @lwork the size of work
	Output:
	  @u the distances 
	  @v the computed empirical semivariogram
*/
void semivariog(const MatrixXd &geom, const VectorXd &y, double distLB, double 
	binLen, int nbin, VectorXd &u, VectorXd &v, size_t *work, int lwork)
{
	assert(geom.rows() == y.rows());
	assert(u.size() == nbin);
	assert(v.size() == nbin);
	assert(lwork >= nbin);
	size_t n = y.size();
	size_t ndist = min(size_t(1e8), n * (n-1) / 2);
	iota(u.data(), u.data() + nbin, 0.5);
	for_each(u.data(), u.data() + nbin, [distLB, binLen](double &x)
		{x = x * binLen + distLB;});
	fill(v.data(), v.data() + nbin, 0.0);
	fill(work, work + nbin, 0);
	std::default_random_engine generator;
	std::uniform_int_distribution<size_t> distribution(0, n-1);
//	#pragma omp parallel for
	for(size_t k = 0; k < ndist; k++)
	{
//		size_t i = k % n;
//              size_t j = k / n;
//              if( i <= j )
//              {
//                      j = n - j - 2;
//                      i = n - i - 1;
//              }
		size_t i = distribution(generator);
		size_t j = distribution(generator);
		if(i == j) continue;
		double dist = (geom.row(i) - geom.row(j)).norm();
		int binIdx = (dist - distLB) / binLen;
		if(binIdx > -1 && binIdx < nbin)
		{
//			#pragma omp atomic update
			work[binIdx]++;

//			#pragma omp atomic update
			v(binIdx) += (y[i] - y[j]) * (y[i] - y[j]);
		}
	}
	transform(v.data(), v.data() + nbin, work, v.data(), [](double x, size_t n)
		{return x / 2.0 / (double)n;});
}

void gaussian_vec(double *x, int n)
{
        std::mt19937 rng(rand()); // random-number engine is Mersenne Twister
        std::normal_distribution<double> normal;
        for (int i = 0; i < n; i++)
                x[i] = normal(rng);
}

/*
	SKP matrix-vector multiplication
	initial vector stored in @x
	matrix and vector dimensions are @n
	result stored in @y
	lwork should be no smaller than @n
*/
void klrM_vec_mult(const KLR_mat &klrM, int n, const double *x, double *y, 
	double *work, int lwork)
{
	assert(lwork >= n);
	assert(klrM.sz_U * klrM.sz_V == n);
	fill(y, y+n, 0.0);
	for(int k = 0; k < klrM.num_term; k++)
	{
		Map<const MatrixXd> U(klrM.U.col(k).data(), klrM.sz_U, klrM.sz_U);
		Map<const MatrixXd> V(klrM.V.col(k).data(), klrM.sz_V, klrM.sz_V);
		#pragma omp parallel for
		for(int i = 0; i < klrM.sz_U; i++)
		{
			Map<VectorXd> vi(work + klrM.sz_V * i, klrM.sz_V);
			Map<const VectorXd> xi(x + klrM.sz_V * i, klrM.sz_V);
			vi.noalias() = V * xi;
		}
		#pragma omp parallel for
		for(int i = 0; i < klrM.sz_U; i++)
		{
			cblas_dgemv(CblasColMajor, CblasNoTrans, klrM.sz_V, klrM.sz_U,
				1.0, work, klrM.sz_V, U.row(i).data(), U.rows(), 1.0,
				y + klrM.sz_V * i, 1);
		}
	}
}

/*
	The simulation for generating GRF on the 2D plain
	@m the dimension of the 2D grid
	@kernelType 1: exponential kernel 2: Whittle kernel
	@domainType 1: fixed 2: expanding
*/
void GRF_app(int m, int kernelType, int domainType, std::ofstream &file)
{
	int n = m * m;
	// geometry
	MatrixXd geom(n, 2);
	double unit = 1.0 / (double) m;
	for(int i = 0; i < n; i++)
	{
		geom(i, 0) = (i / m) * unit;
		geom(i, 1) = (i % m) * unit;
	}
	// beta
	double beta;
	if(domainType == 1)
	{
		beta = 0.1;
	}else
	{
		beta = 0.1 / double(m / 32);
	}
	// kernel visitor
	function<double(int, int)> visitor;
	if(kernelType == 1)
        {
                auto kernel = exp_cov_func(beta);
		visitor = [&geom, kernel](int i, int j){
			return kernel((geom.row(i) - geom.row(j)).norm());};
        }else
        {
                auto kernel = Matern_kernel(beta, 1.0);
		visitor = [&geom, kernel](int i, int j){
			return kernel((geom.row(i) - geom.row(j)).norm());};
        }
	// KLR for the correlation matrix
	TreeNode treeNode = comp_UV_ACA(visitor, 0, 0, n, m, m, 1e-5, 0, true);
	KLR_mat klrM(30, 30, m, m);
	if(30 < treeNode.num_term)
	{
		cout << "klrM's max num of term is smaller than the initial"
			" number of terms in the "
			"correlation matrix" << endl;
		exit(1);
	}
	klrM.num_term = treeNode.num_term;
        klrM.U.block(0,0,n,klrM.num_term) = treeNode.U;
        klrM.V.block(0,0,n,klrM.num_term) = treeNode.V;
        for(int i = 0 ; i < klrM.num_term ; i++)
        {
                double v_norm = klrM.V.col(i).norm();
                klrM.V.col(i) = klrM.V.col(i) / v_norm;
                klrM.U.col(i) = klrM.U.col(i) * v_norm;
        }
	// KLR for the Cholesky factor and its inverse
	// memory allocated in the common-base block Cholesky routine
	KLR_mat klrChol;
	KLR_mat klrInv;
	vector<MatrixXd> LDiag;
	// common-base block Cholesky
	TimeStamp start, end;
	start = std::chrono::steady_clock::now();
	cb_blk_chol_v2(klrM, klrChol, klrInv, LDiag, 30, 10, 30, 10, 
		2, 2, 1, 1e-3, NULL);
	end = std::chrono::steady_clock::now();
	double time_used = std::chrono::duration<double> (end-start).
		count();
	file << time_used << endl;
	// set upper tri of U to zero
	for(int k = 0; k < klrChol.num_term; k++)
	{
		Map<MatrixXd> Uk(klrChol.U.col(k).data(), klrChol.sz_U,
			klrChol.sz_U);
		Uk.triangularView<Upper>().setZero();
	}
	// simulate GRF
	VectorXd x(n);
	VectorXd y(n);
	VectorXd work(n);
	int nbin = 20;
	size_t *workInt = new size_t[nbin];
	for(int l = 0; l < 100; l++)
	{
		start = std::chrono::steady_clock::now();
		gaussian_vec(x.data(), n);
		klrM_vec_mult(klrChol, n, x.data(), y.data(), work.data(), 
			work.size());
		#pragma omp parallel for
		for(int i = 0; i < klrChol.sz_U; i++)
			cblas_dgemv(CblasColMajor, CblasNoTrans, klrChol.sz_V, 
				klrChol.sz_V,
				1.0, LDiag[i].data(), LDiag[i].rows(), x.data() + 
				klrChol.sz_V * i, 1, 1.0, y.data() + klrChol.sz_V * i,
				1);
		end = std::chrono::steady_clock::now();
		time_used = std::chrono::duration<double> (end-start).
			count();
		file << time_used << " ";
		// output mean, variance, and semivariogram
		double meanY = y.mean();
		for_each(y.data(), y.data() + n, [meanY](double &y){y -= meanY;});
		double varY = y.dot(y) / double(n);
		double distLB = 0.0;
		double binLen = 0.07;
		VectorXd u(nbin);
		VectorXd v(nbin);
		start = std::chrono::steady_clock::now();
		semivariog(geom, y, distLB, binLen, nbin, u, v, workInt, 
			nbin);
		end = std::chrono::steady_clock::now();
		time_used = std::chrono::duration<double> (end-start).
			count();
		file << time_used << endl;
		file << meanY << " " << varY << endl;
		file << u.transpose() << endl;
		file << v.transpose() << endl;
		for_each(workInt, workInt + nbin, [&file](size_t &x)
			{file << x << " ";});
		file << endl;
	}
	delete[] workInt;
}









