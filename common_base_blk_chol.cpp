#include <iostream>
#include <fstream>
#include <iomanip>
#include "mkl.h"
#include "factor_SVD.h"
#include "common_base_blk_chol.h"

#define OVERSAMPLE 20
#define RETROSIZE 1
#define SETBACK 3

//' Stores the coefficient matrix of V relative to Q
class CoefVToQ
{
public:
        int nrow;
        int ncol;
        /*
                choice for the initial dimensions of C
                        row: ncolMaxBXX
                        col: 2 X expected ncol of VX + OVERSAMPLE
        */
        Eigen::MatrixXd allocM;
        // initializer
        CoefVToQ(int nrowInit, int ncolInit) {
                nrow = 0;
                ncol = 0;
                allocM.resize(nrowInit, ncolInit);
        }
};

Eigen::MatrixXd gaussianMat(int m, int n); 
int update_C_MGS(CoefVToQ &C, int offset, int ncolNew, double &epslRel, double *work,
	int lwork);
int update_C_SVD(CoefVToQ &C, int offset, int max_rk, int ncolNew, double &epsl, 
	double *work, int lwork);
int update_QR_perm(const KLR_mat &C1, int numNewBaseC1, const KLR_mat &C2, int 
	numNewBaseC2, QRFactMGS &B, Eigen::MatrixXd &workM, Eigen::RowVectorXd 
	&workRowVec);
bool add_base_KLR_mat(KLR_mat &M, const Eigen::VectorXd &vec, int rowIdx, 
	double epsl);
bool add_base_KLR_mat(KLR_mat &M, const Eigen::VectorXd &vec, double *coord, 
	double epsl);
void MGS(KLR_mat &M, Eigen::Map<Eigen::VectorXd> &vec, int rowIdx = -1);
void MGS(KLR_mat &M, Eigen::Map<Eigen::VectorXd> &vec, double *coord, int lcoord);
void add_row_CoefVToQ(CoefVToQ &M, int nrowNew);
void diag_blk(int i, const KLR_mat &covM, const KLR_mat &L, const QRFactMGS &B22T, 
	double *array1, double *array2, double *array3, int dimU, int dimV, 
	int firstColIdx);
void Schur_coef(int i, const KLR_mat &covM, const KLR_mat &L, double *array1, int ld1,
	int dimU, int firstColIdx, int k2, bool inCorrection);
void new_base_routine(const QRFactMGS &B1, QRFactMGS &B2, const CoefVToQ &C1, 
	KLR_mat &klrM1, CoefVToQ &C2, int nBaseNew, bool isConj, int dimV, 
	Eigen::MatrixXd &workM, Eigen::RowVectorXd &workRowVec, const KLR_mat *klrM2);
int disc_new_base(double *array1, int ld1, bool transpose, int ncol, 
	double *array2, int ld2, double *array3, int ld3, const QRFactMGS &B, 
	CoefVToQ &C, const KLR_mat &klrM, const Eigen::MatrixXd &GaussM, double epsl, 
	double *work, int lwork);

using namespace std;
using namespace Eigen;

//' The SKP Cholesky factorization with the correction machanism 
//' Input:
//'   @covM stores the SKP representation of the input covariance matrix. Its basis 
//'     will be incremented for the Schur complements
//'   @k2Max decides the number of columns to allocate for the SKP representation 
//'     of L
//'   @k3Max similar with @k2Max but for L inverse
//'   @crt_k2Max the current max number of basis for L, which could be increased 
//'      after correction
//'   @crt_k3Max similar with @crt_k2Max but for L inverse
//'   @incr_k1, @incr_k2, @incr_k3 the increments after correction for 
//'      the max number of basis of @covM, @L, @Linv, respectively  
//'   @epslRel, the tolerance for discovering new basis
//'   @check, the Cholesky factor used for checking the relative error. 
//'      Set to NULL if checking is not needed

void cb_blk_chol_v2(KLR_mat &covM, KLR_mat &L, KLR_mat &Linv, std::vector<Eigen::
	MatrixXd> &LDiag, int k2Max, int k3Max, int crt_k2Max, int crt_k3Max, 
	int incr_k1, int incr_k2, int incr_k3, double epslRel, const MatrixXd *check)
{
	// init
	int dimU = covM.sz_U;
	int dimV = covM.sz_V;
	L.reset(k2Max, crt_k2Max, dimU, dimV);
	Linv.reset(k3Max, crt_k3Max, dimU, dimV);
	LDiag.resize(dimU);
	int nrowB = dimV * dimV;
	int ncolMaxB13T = covM.max_num_term * k3Max;
	int ncolMaxB22T = k2Max * k2Max + covM.num_term;
	QRFactMGS B13T(nrowB, ncolMaxB13T);
	QRFactMGS B22T(nrowB, ncolMaxB22T);
	CoefVToQ C1(ncolMaxB22T, 2*covM.max_num_term + OVERSAMPLE);
	CoefVToQ C2(ncolMaxB13T, 2*k2Max + OVERSAMPLE);
	MatrixXd GaussM = gaussianMat(dimU, max(covM.max_num_term, k2Max) + 
		OVERSAMPLE);
	assert(B22T.k + covM.num_term <= B22T.k_max);
	B22T.Q.block(0,0,nrowB,covM.num_term) = covM.V.block(0,0,nrowB,covM.num_term);
	B22T.k = covM.num_term;
	C1.allocM.block(0,0,covM.num_term,covM.num_term).setIdentity();
	C1.nrow = covM.num_term;
	C1.ncol = covM.num_term;

	// define tmp var
	MatrixXd tmpC(max(ncolMaxB22T, ncolMaxB13T), dimU);
	MatrixXd tmpCProj(max(ncolMaxB22T, ncolMaxB13T), dimU);
	MatrixXd workM(dimV*dimV, max(ncolMaxB22T, ncolMaxB13T));
	RowVectorXd workRowVec(max(ncolMaxB22T, ncolMaxB13T));
	MatrixXd tmpM(dimV, dimV);
	int firstColIdx = 0;
	int initCovMNumTerm = covM.num_term;
	covM.U.block(0, covM.num_term, covM.U.rows(), covM.max_num_term - 
		covM.num_term).setZero();

	// loop through each col
	for(int i = 0; i < dimU; i++)
	{
		int k2 = L.num_term;
		// diag chol
		diag_blk(i, covM, L, B22T, tmpC.col(i).data(), 
			tmpCProj.col(0).data(), tmpM.data(), dimU, dimV, firstColIdx);
		int succ = cholesky(tmpM);
		if(succ < 0)
		{
			printf("The %d argument in the Cholesky is invalid\n", -succ);
			exit(1);
		}
		if(succ > 0) // Implement the correction mechanism
		{
			if(i - SETBACK - RETROSIZE < firstColIdx)
			{
				printf("Cholesky failed when i = %d < firstColIdx + "
					"RETROSIZE + SETBACK. Cannot proceed.\n", i);
				exit(1);
			}
			i -= SETBACK;
			cout << "Cholesky correction at column " << i << endl;
			for(int j = i + 1; j < dimU; j++)
				LDiag[j] = LDiag[i];
			for(int k = 0; k < L.num_term; k++)
			{
				Map<MatrixXd> U(L.U.col(k).data(), dimU, dimU);
				for(int j = i + 1; j < dimU; j++)
					for(int i2 = j + 1; i2 < dimU; i2++)
					{
						U(i2, j) = U(i2 - (j - i), i);
					}
			}
			return;
		} // if(cholesky > 0)
		LDiag[i] = tmpM; // i-th diag blk in L
		// diagonal block inverse
		succ = inverse(tmpM);
		assert(succ == 0);
		// discover new base for Linv
		Map<VectorXd> tmpMVec(tmpM.data(), nrowB);
		double epsl = tmpMVec.norm() * epslRel; 
		MGS(Linv, tmpMVec, i*dimU+i);
		if(add_base_KLR_mat(Linv, tmpMVec, i*dimU+i, epsl))
		{
		        int numNewBaseQ13T = update_QR_perm(covM, 0, Linv, 1,
		                B13T, workM, workRowVec);
			add_row_CoefVToQ(C2, numNewBaseQ13T);
		}
		if(i < dimU-1)
		{
			// trailing column Schur
			Schur_coef(i, covM, L, tmpC.col(i+1).data(), tmpC.rows(), 
				dimU, firstColIdx, L.num_term, false);
			int nBaseNew = disc_new_base(tmpC.col(i+1).data(), 
				tmpC.rows(), false, dimU-i-1, tmpCProj.data(), 
				tmpCProj.rows(), tmpC.data(), tmpC.rows(), B22T, C1, 
				covM, GaussM, epslRel, workRowVec.data(), 
				workRowVec.size());
			if(nBaseNew > 0)
				new_base_routine(B22T, B13T, C1, covM, C2, nBaseNew,
					false, dimV, workM, workRowVec, &Linv);
			tmpC.block(0,0,C1.ncol,dimU-i-1) = covM.U.block(dimU*i+i+1,0,
				dimU-i-1,covM.num_term).transpose() - tmpC.block(0,0,
				C1.ncol,dimU-i-1);
			// trailing column Schur multiplied by 
			// diag blk inverse
			int k1 = covM.num_term;
			int k3 = Linv.num_term;
			workM.block(0,0,dimU-i-1,k1*k3).setZero();
			for(int k = 0; k < k1; k++)
			{
				cblas_dger(CblasColMajor, dimU-i-1, k3, 1.0, tmpC.
					data()+k, tmpC.rows(), Linv.U.
					data()+i*dimU+i, Linv.U.rows(), workM.data()+
					k*k3*workM.rows(), workM.rows());
			}
			// discover new base for L	
			nBaseNew = disc_new_base(workM.data(), workM.rows(), true,
				dimU-i-1, tmpCProj.data(), tmpCProj.rows(), 
				L.U.data()+dimU*i+i+1, L.U.rows(), B13T, C2, L,
				GaussM, epslRel, workRowVec.data(), 
				workRowVec.size());
			if(nBaseNew > 0)
				new_base_routine(B13T, B22T, C2, L, C1, nBaseNew,
					true, dimV, workM, workRowVec, NULL);
		} // if(i < dimU-1) update trailing blk in the col
	} // i = 0:dimU
}

// Generate a m-by-n Gaussian matrix for random sampling
Eigen::MatrixXd gaussianMat(int m, int n) 
{
	MatrixXd A(m, n); 
	std::mt19937 rng(rand()); // random-number engine is Mersenne Twister
	std::normal_distribution<double> normal;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			A(i,j) = normal(rng); 
	return A; 
}

/*
	Projects new coordinate vectors onto existing basis vectors, both are 
	  stored in @C
	The columns of @C starting from @offset stores the coordinates 
	  to be projected 
	@ncolNew the number of coordinate vectors to be projected
	@epslRel the tolerance used for finding new basis
	@work a temporary working space, should be of dimension at least C.nrow
*/
int update_C_MGS(CoefVToQ &C, int offset, int ncolNew, double &epslRel, double *work,
	int lwork)
{
	int nrow = C.nrow;
	int ncolCPrev = C.ncol;
	assert(lwork >= nrow);
	Map<VectorXd> v(work, nrow);
	for(int i = 0; i < ncolNew; i++)
	{
		v = C.allocM.col(offset+i).segment(0,nrow);
		double epsl = v.norm() * epslRel;
		for(int k = 0; k < C.ncol; k++)
		{
			double innerPd = C.allocM.col(k).segment(0,nrow).dot(v);
			cblas_daxpy(nrow, -innerPd, C.allocM.col(k).data(), 1, v.
				data(), 1);
//			v -= C.allocM.col(k).segment(0,nrow).dot(v) * 
//				C.allocM.col(k).segment(0,nrow);
		}
		double vNorm = v.norm();
		if(vNorm > epsl)
		{
			if(C.ncol >= offset)
			{
//				cout << "Warning: the number of bases reached upper "
//					"limit. New base not added\n";
			}
			else
			{
				C.allocM.col(C.ncol).segment(0,nrow) = v / vNorm;
				C.ncol++;
			}
		}
	}
	return C.ncol - ncolCPrev;
}

/*
	Update the Q, R factors when new basis are added
	@C1 @C2 stores the coordinates of the basis relative to B.Q
	@numNewBaseC1 @numNewBaseC2 stores the number of new basis
	The other inputs are temporary working spaces
*/
int update_QR_perm(const KLR_mat &C1, int numNewBaseC1, const KLR_mat &C2, int 
	numNewBaseC2, QRFactMGS &B, Eigen::MatrixXd &workM, Eigen::RowVectorXd 
	&workRowVec)
{
	int ncolNew = C1.num_term * C2.num_term - (C1.num_term - numNewBaseC1) * 
		(C2.num_term - numNewBaseC2);
	assert(B.k_max >= B.num_col + ncolNew);
	assert(B.k_max >= B.k + ncolNew);
	int ncolC1Prev = C1.num_term - numNewBaseC1;
	int ncolC2Prev = C2.num_term - numNewBaseC2;
	int ncolC2 = C2.num_term;
	int kPrev = B.k;
	int dimV = C1.sz_V;
	// shift crt col in B.R
	if(numNewBaseC2 > 0 && ncolC2Prev > 0)
		for(int i = ncolC1Prev-1; i > 0; i--)
		{
			workM.block(0,0,kPrev,ncolC2Prev) = B.R.block(0,i*ncolC2Prev,
				kPrev,ncolC2Prev);
			B.R.block(0,i*ncolC2,kPrev,ncolC2Prev) = workM.block(0,0,
				kPrev,ncolC2Prev);
		}
	B.num_col += ncolNew;
	
	// compute new vectorized matrices into workM
	for(int k = 0; k < ncolNew; k++)
	{
		if(k < ncolC1Prev*numNewBaseC2)
		{
			int i = k / numNewBaseC2;
			int j = k % numNewBaseC2;
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimV, 
				dimV, dimV, 1.0, C1.V.data()+i*C1.V.rows(), dimV, 
				C2.V.data()+(ncolC2Prev+j)*C2.V.rows(), dimV, 0.0, 
				workM.data()+k*workM.rows(), dimV);
//			Map<MatrixXd> tmpM(workM.col(k).data(), dimV, 
//				dimV);
//			Map<const MatrixXd> tmpM1(C1.V.col(i).data(), dimV, dimV);
//			Map<const MatrixXd> tmpM2(C2.V.col(ncolC2Prev+j).data(), dimV,
//				dimV);
//			tmpM.noalias() = tmpM1 * tmpM2.transpose();
		}
		else
		{
			int i = (k - ncolC1Prev*numNewBaseC2) / ncolC2;
			int j = (k - ncolC1Prev*numNewBaseC2) % ncolC2;
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimV, 
				dimV, dimV, 1.0, C1.V.data()+(ncolC1Prev+i)*C1.V.
				rows(), dimV, C2.V.data()+j*C2.V.rows(), dimV, 0.0, 
				workM.  data()+k*workM.rows(), dimV);
//			Map<MatrixXd> tmpM(workM.col(k).data(), dimV, dimV);
//			Map<const MatrixXd> tmpM1(C1.V.col(ncolC1Prev+i).data(), dimV,
//				dimV);
//			Map<const MatrixXd> tmpM2(C2.V.col(j).data(), dimV, dimV);
//			tmpM.noalias() = tmpM1 * tmpM2.transpose();
		}
	}

	// MGS (prev bases in B.Q)
	for(int k = 0; k < kPrev; k++)
	{
		cblas_dgemv(CblasColMajor, CblasTrans, dimV*dimV, ncolNew, 1.0, workM.
			data(), workM.rows(), B.Q.data()+k*B.Q.rows(), 1, 0.0, 
			workRowVec.data(), 1);
//		workRowVec.segment(0,ncolNew).noalias() = B.Q.col(k).transpose() * 
//			workM.block(0,0,dimV*dimV,ncolNew);
		for(int i = 0; i < ncolC1Prev; i++)
			for(int j = 0; j < numNewBaseC2; j++)
				B.R(k, i*ncolC2 + ncolC2Prev + j) = workRowVec(
					i*numNewBaseC2 + j);
		for(int i = 0; i < numNewBaseC1; i++)
			for(int j = 0; j < ncolC2; j++)
				B.R(k, (i+ncolC1Prev)*ncolC2 + j) = workRowVec(
					ncolC1Prev*numNewBaseC2 + i*ncolC2 + j);
		cblas_dger(CblasColMajor, dimV*dimV, ncolNew, -1.0, B.Q.data()+k*B.Q.
			rows(), 1, workRowVec.data(), 1, workM.data(), workM.rows());
//		workM.block(0,0,dimV*dimV,ncolNew) -= B.Q.col(k) * workRowVec.
//			segment(0,ncolNew);
	}
	// MGS (test new base)
	vector<int> RColIdxNew(ncolNew);
	iota(RColIdxNew.begin(), RColIdxNew.end(), 0);
	for(int &k : RColIdxNew)
	{
		if(k < ncolC1Prev*numNewBaseC2)
			k = (k/numNewBaseC2)*ncolC2 + k%numNewBaseC2 + ncolC2Prev;
		else
		{
			int i = (k - ncolC1Prev*numNewBaseC2) / ncolC2;
			int j = (k - ncolC1Prev*numNewBaseC2) % ncolC2;
			k = (i+ncolC1Prev)*ncolC2 + j;
		}
	}
	for(int k = 0; k < ncolNew; k++)
	{
		double FnormSq = workM.col(k).segment(0,dimV*dimV).squaredNorm();
		if(FnormSq > B.ALMOSTZERO)
		{
			double Fnorm = sqrt(FnormSq);
			transform(workM.data()+k*workM.rows(), workM.data()+k*workM.
				rows()+dimV*dimV, B.Q.data()+B.k*B.Q.rows(),
				[&Fnorm](double &coef){return coef / Fnorm;});
//			B.Q.col(B.k).noalias() = workM.col(k).segment(0,dimV*dimV) /
//				Fnorm;
			B.R.row(B.k).segment(0,B.num_col).setZero();
			B.R(B.k, RColIdxNew[k]) = Fnorm;
			if(k < ncolNew - 1)
			{
				cblas_dgemv(CblasColMajor, CblasTrans, dimV*dimV, 
					ncolNew-1-k, 1.0, workM.data()+(k+1)*workM.
					rows(), workM.rows(), B.Q.data()+B.k*B.Q.
					rows(), 1, 0.0, workRowVec.data(), 1); 
//				workRowVec.segment(0,ncolNew-1-k).noalias() = B.Q.
//					col(B.k).transpose() * workM.block(0,k+1, 
//					dimV*dimV,ncolNew-1-k);
				for(int l = k+1; l < ncolNew; l++)
					B.R(B.k, RColIdxNew[l]) = workRowVec(l-k-1);
				cblas_dger(CblasColMajor, dimV*dimV, ncolNew-1-k,
					-1.0, B.Q.data()+B.k*B.Q.rows(), 1, 
					workRowVec.data(), 1, workM.data()+(k+1)*
					workM.rows(), workM.rows());
//				workM.block(0,k+1,dimV*dimV,ncolNew-1-k) -= B.Q.
//					col(B.k) * workRowVec.segment(0,ncolNew-1-k);
			}
			B.k++;
		}
	}
	return B.k - kPrev;
}

/*
	Discovery new basis using a dense matrix
	@vec stores the vectorized dense matrix
	@M stores the SKP representation
	@rowIdx specifies the diagonal position of the dense matrix in @M
*/
bool add_base_KLR_mat(KLR_mat &M, const VectorXd &vec, int rowIdx, double epsl)
{
	int k = M.num_term;
	if(k == M.crt_max_num_term)
		return false;

	double norm = vec.norm();
	if(norm < epsl)
		return false;

	M.U.col(k).setZero();
	M.U(rowIdx, k) = norm;
	M.V.col(k).noalias() = vec / norm;
	M.num_term++;
	return true;
}

/*
	Project @vec onto existing basis of @M and store the projection
	  coordinates in @coord
*/
bool add_base_KLR_mat(KLR_mat &M, const VectorXd &vec, double *coord, double epsl)
{
	int k = M.num_term;
	if(k == M.crt_max_num_term)
		return false;

	double norm = vec.norm();
	if(norm < epsl)
		return false;

	M.U.col(k).setZero();
	*(coord) = norm;
	M.V.col(k).noalias() = vec / norm;
	M.num_term++;
	return true;
}

void MGS(KLR_mat &M, Eigen::Map<Eigen::VectorXd> &vec, int rowIdx)
{
	for(int i = 0; i < M.num_term; i++)
	{
		double dotProd = M.V.col(i).dot(vec);
		if(rowIdx >= 0) M.U(rowIdx, i) = dotProd;
		vec -= dotProd * M.V.col(i);
	}
}

void MGS(KLR_mat &M, Eigen::Map<Eigen::VectorXd> &vec, double *coord, int lcoord)
{
	assert(lcoord >= M.num_term);
	for(int i = 0; i < M.num_term; i++)
	{
		coord[i] = M.V.col(i).dot(vec);
		vec -= coord[i] * M.V.col(i);
	}
}

/*
	Add @nrowNew rows of zero to the coefficient matrix @M
	Used when a new basis is found
*/
void add_row_CoefVToQ(CoefVToQ &M, int nrowNew)
{
	if(nrowNew <= 0)
		return;
	assert(M.nrow + nrowNew <= M.allocM.rows());
	M.allocM.block(M.nrow, 0, nrowNew, M.ncol).setZero();
	M.nrow += nrowNew;
}

/*
	Find new bases from the column mixture
*/
int update_C_SVD(CoefVToQ &C, int offset, int max_rk, int ncolNew, double &epsl, 
	double *work, int lwork)
{
	// If C is reached maximum rank, return directly
	if(C.ncol >= max_rk)
		return 0;
	// Setup
	int nrow = C.nrow;
	int ncolCPrev = C.ncol;
	assert(lwork >= ncolNew * 2);
	Map<RowVectorXd> v(work, ncolNew);
	// MGS to orthogonalize
	for(int i = 0; i < ncolCPrev; i++)
	{
		// inner prod with existing base i
		v.noalias() = C.allocM.col(i).segment(0, nrow).transpose() * C.allocM.
			block(0, offset, nrow, ncolNew);
		// rank 1 update
		// substract the new col with their proj on base i
		cblas_dger(CblasColMajor, nrow, ncolNew, -1.0, C.allocM.col(i).data(),
			1, work, 1, C.allocM.col(offset).data(), C.allocM.rows());
	}
	// SVD to find U and singular values
	int nSingVal = min(ncolNew, nrow);
	int code = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', nrow, ncolNew, 
		C.allocM.col(offset).data(), C.allocM.rows(), work, NULL, 1, NULL,
		1, work + ncolNew);
	assert(code == 0);
	// Discover new base blocks
	for(int i = 0; i < nSingVal; i++)
	{
		if(C.ncol >= max_rk)
			break;
		if(work[i] > epsl)
		{
			C.allocM.col(C.ncol).segment(0, nrow) = C.allocM.col(offset +
				i).segment(0, nrow);
			C.ncol++;
		}else
			break;
	}
	return C.ncol - ncolCPrev;
}

/*
	Compute the first diag blk in the Schur complement
	array1 stores the L L^T coef cumulated from column firstColIdx
	array2 stores array1's projection on Q
	array3 stores the dense block used for Cholesky
	returns the diagonal block before Cholesky
*/
void diag_blk(int i, const KLR_mat &covM, const KLR_mat &L, const QRFactMGS &B22T, 
	double *array1, double *array2, double *array3, int dimU, int dimV, 
	int firstColIdx)
{
        int k2 = sqrt(B22T.num_col);
	int nrowB = dimV * dimV;
	fill(array1, array1 + k2*k2, 0.0);
        if(firstColIdx > 0)
        {
                for(int k = RETROSIZE; k > 0; k--)
                {
                        int rowIdx = (firstColIdx-k)*dimU + i;
                        cblas_dger(CblasColMajor, k2, k2, 1.0, covM.U.data()+rowIdx, 
				covM.U.rows(), covM.U.data()+rowIdx, covM.U.rows(), 
				array1, k2);
                }
        }else
		// the second scenario
		firstColIdx = -firstColIdx;
        for(int j = firstColIdx; j < i; j++) // col idx in L 
        {
                int rowIdx = j*dimU + i; // i row j col
                cblas_dger(CblasColMajor, k2, k2, 1.0, L.U.data()+rowIdx, 
			L.U.rows(), L.U.data()+rowIdx, L.U.rows(), array1, k2);
        }
        // array2 = B22T.R * array1
	cblas_dgemv(CblasColMajor, CblasNoTrans, B22T.k, B22T.num_col, 1.0, 
		B22T.R.data(), B22T.R.rows(), array1, 1, 0.0, array2, 1);
        // array3 = covM.V * covM.U.row(i*dimU+i)
        cblas_dgemv(CblasColMajor, CblasNoTrans, nrowB, covM.num_term,
                1.0, covM.V.data(), nrowB, covM.U.data() + i*dimU+i,
                covM.U.rows(), 0.0, array3, 1);
        // array3 = -B22T.Q * tmpCProj.col(0) + array3
        cblas_dgemv(CblasColMajor, CblasNoTrans, nrowB, B22T.k, -1.0, B22T.Q.data(), 
		nrowB, array2, 1, 1.0, array3, 1);
}

/*
	Compute the coefficients of the Schur complement (tailing column only) 
		relative to B
	columns of array1 stores the Schur coefficients
	if(inCorrection) add the RETROSIZE columns before the column firstColIdx
*/
void Schur_coef(int i, const KLR_mat &covM, const KLR_mat &L, double *array1, int ld1,
	int dimU, int firstColIdx, int k2, bool inCorrection)
{
	assert(ld1 >= k2*k2);
	for(int j = 0; j < dimU - i - 1; j++)
		fill(array1 + ld1*j, array1 + ld1*j + k2*k2, 0.0);
        #pragma omp parallel for
	for(int j = 0; j < dimU - i - 1; j++)
        {
		int j1 = j + i + 1;
		if(!inCorrection && firstColIdx > 0)
		{
			for(int k = RETROSIZE; k > 0; k--)
			{
				int colIdxI = (firstColIdx-k)*dimU+i;
	                        int colIdxJ = (firstColIdx-k)*dimU+j1;
	                        cblas_dger(CblasColMajor, k2, k2, 1.0,
	                                covM.U.data()+colIdxI,
	                                covM.U.rows(), covM.U.data()+
	                                colIdxJ, covM.U.rows(), array1 + ld1*j, k2);
			}
		}
                for(int i1 = firstColIdx; i1 < i; i1++)
                {
                        int colIdxI = i1*dimU+i;
                        int colIdxJ = i1*dimU+j1;
                        cblas_dger(CblasColMajor, k2, k2, 1.0, L.U.data() + 
				colIdxI, L.U.rows(), L.U.data() + colIdxJ, 
				L.U.rows(), array1 + ld1*j, k2);
                }
        }
}

/*
	Update the V, Q, R factors when new bases are added to C
	Formulate m2 X m2 dimensional base block with B1 and store in klrM1.V
	Update Q, R factors in B2, where work and workRowVec are tmp storage
		isConj = T => QR = klrM1 * klrM1^T
		else => QR = klrM1 * klrM2^T
	Add new rows to C2
*/
void new_base_routine(const QRFactMGS &B1, QRFactMGS &B2, const CoefVToQ &C1, 
	KLR_mat &klrM1, CoefVToQ &C2, int nBaseNew, bool isConj, int dimV, 
	Eigen::MatrixXd &workM, Eigen::RowVectorXd &workRowVec, const KLR_mat *klrM2)
{
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                dimV*dimV, nBaseNew, B1.k, 1.0, B1.Q.data(),
                B1.Q.rows(), C1.allocM.col(C1.ncol-
                nBaseNew).data(), C1.allocM.rows(), 0.0, klrM1.
                V.col(klrM1.num_term).data(), klrM1.V.rows());
        klrM1.num_term += nBaseNew;
        int numNewBase;
	if(isConj)
		numNewBase = update_QR_perm(klrM1, nBaseNew, klrM1, nBaseNew, B2, 
			workM, workRowVec);
	else
		numNewBase = update_QR_perm(klrM1, nBaseNew, (*klrM2), 0, B2, 
			workM, workRowVec);

        add_row_CoefVToQ(C2, numNewBase);
}

/*
	Discover new bases for C given the coefficients relative to B, 
		stored in array1
	array1 stores the original coefficients relative to B
	array2 stores the projection of array1 onto Q
	array3 stores the coefficients relative to the updated C
*/
int disc_new_base(double *array1, int ld1, bool transpose, int ncol, 
	double *array2, int ld2, double *array3, int ld3, const QRFactMGS &B, 
	CoefVToQ &C, const KLR_mat &klrM, const Eigen::MatrixXd &GaussM, double epsl, 
	double *work, int lwork)
{
	// Project array1 onto B.Q
	if(transpose)
	        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, B.k,
	                ncol, B.num_col, 1.0, B.R.data(), B.R.
	                rows(), array1, ld1, 0.0, array2, ld2);
	else
	        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.k,
	                ncol, B.num_col, 1.0, B.R.data(), B.R.
	                rows(), array1, ld1, 0.0, array2, ld2);
        // Gaussian mixture
        int ncolGaussM = min(ncol+OVERSAMPLE, klrM.max_num_term + OVERSAMPLE);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.k,
                ncolGaussM, ncol, 1.0, array2, ld2, GaussM.data(),
                GaussM.rows(), 0.0, C.allocM.col(klrM.max_num_term).data(),
                C.allocM.rows());
        // Discover new base for klrM
        int nBaseNew = update_C_SVD(C, klrM.max_num_term, klrM.crt_max_num_term, 
		ncolGaussM, epsl, work, lwork);
        assert(klrM.num_term + nBaseNew <= klrM.crt_max_num_term);
        // Store coef of new columns in array3
	if(transpose)
	        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncol, C.ncol,
	                B.k, 1.0, array2, ld2, C.allocM.data(), C.allocM.rows(), 0.0,
	                array3, ld3);
	else
	        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, C.ncol, ncol,
	                B.k, 1.0, C.allocM.data(), C.allocM.rows(), array2, ld2, 0.0,
	                array3, ld3);
        return nBaseNew;
}

MatrixXd KLR_mat::uncompress() const
{
        int dim = sz_U * sz_V;
        MatrixXd M = MatrixXd::Zero(dim,dim);
        for(int i = 0 ; i < num_term ; i++)
        {
                Map<const MatrixXd> tmpM1(U.col(i).data() , sz_U , sz_U);
                Map<const MatrixXd> tmpM2(V.col(i).data() , sz_V , sz_V);
                for(int i1 = 0 ; i1 < sz_U ; i1++)
                        for(int j1 = 0 ; j1 < sz_U ; j1++)
                                M.block(i1*sz_V,j1*sz_V,sz_V,sz_V) += tmpM1(i1,j1)*
                                        tmpM2;
        }
        return M;
}

