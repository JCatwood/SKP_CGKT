#ifndef COMMON_BASE_BLK_CHOL_H
#define COMMON_BASE_BLK_CHOL_H
#include <exception>
#include <vector>
#include "Dense"


/*
	Store a matrix a sum of Kron prod
	\sum (U_i \otime V_i)
	U_i and V_i should be square
	vec(U_i) and vec(V_i) are stored in U , V
	U.cols = V.cols = max_num_term
	The valid columns, however, are those before num_term
	2019/07/09
	Added a new member `crt_max_num_term`
	2019/12/06
*/
class KLR_mat
{
public:
	int max_num_term;
	int crt_max_num_term;
	int num_term;
	int sz_U , sz_V;
	Eigen::MatrixXd U , V;

	KLR_mat(){}
	KLR_mat(int max_allowed, int crt_max, int U_dim, int V_dim)
	{
		max_num_term = max_allowed;
		crt_max_num_term = crt_max;
		num_term = 0;
		sz_U = U_dim;
		sz_V = V_dim;
		U = Eigen::MatrixXd::Zero(sz_U*sz_U , max_num_term);
		V = Eigen::MatrixXd::Zero(sz_V*sz_V , max_num_term);
	}
	void reset(int max_allowed, int crt_max, int U_dim, int V_dim) // destructive
	{
		max_num_term = max_allowed;
		crt_max_num_term = crt_max;
		num_term = 0;
		sz_U = U_dim;
		sz_V = V_dim;
		U = Eigen::MatrixXd::Zero(sz_U*sz_U , max_num_term);
		V = Eigen::MatrixXd::Zero(sz_V*sz_V , max_num_term);
	}
	~KLR_mat(){}
	Eigen::MatrixXd uncompress() const;
};

/*
        Store the QR factor of a rank-deficient matrix
        Q has extra columns and R has extra rows
        Use k to store the valid number of columns in Q or rows in R
        Use MGS to build Q and R is the coord
	Add a permutation matrix member permuteM
        2019/08/29
*/
class QRFactMGS
{
public:
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
        int k;
        int k_max;
        int num_row;
        int num_col;
        const double ALMOSTZERO = 1e-8;
        std::vector<int> permute; // apply a permutation matrix initialized by 
                // permute(0:num_col-1) to the RHS of R should make B = QR
	// initializer
        QRFactMGS(int num_row_init , int k_max_init)
        {
                k_max = k_max_init;
                k = 0;
                num_row = num_row_init;
                num_col = 0;
                permute = std::vector<int>(k_max , 0);
                Q = Eigen::MatrixXd::Zero(num_row , k_max);
                R = Eigen::MatrixXd::Zero(k_max, k_max);
        }
};

/*
        Exception when the dense Cholesky routine fails
        2019/07/16
*/
class CholException : public std::exception {
public:
        int i;
        int j;
};

void cb_blk_chol_v2(KLR_mat &covM, KLR_mat &L, KLR_mat &Linv, std::vector<Eigen::
        MatrixXd> &LDiag, int k2Max, int k3Max, int crt_k2Max, int crt_k3Max,
        int incr_k1, int incr_k2, int incr_k3, double epslRel,
        const Eigen::MatrixXd *check);
#endif
