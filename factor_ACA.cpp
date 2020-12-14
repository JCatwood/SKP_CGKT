#include <iostream>
#include <fstream>
#include <cmath>
#include "Dense"
#include "factor_ACA.h"
#include "factor_SVD.h"


using namespace std;
using namespace Eigen;


/*
	Factorize the matrix block based on ACA algo
	@mat the dense matrix, whose block is to be factorized
	@normalize defines if a MGS routine should follow the block 
	  ACA algorithm
*/
TreeNode comp_UV_ACA(const Eigen::MatrixXd &mat , int row_offset , int col_offset , 
	int blk_sz , int V_num_row , int V_num_col , double abs_tol , int 
	max_num_term , bool normalize)
{
	// init
	TreeNode node;
	node.num_term = 0;
	node.V_num_row = V_num_row;
	node.V_num_col = V_num_col;
	node.U_num_row = blk_sz / V_num_row;
	node.U_num_col = blk_sz / V_num_col;
	// tmp var
	int tmp_max_num_term = log2(blk_sz); // heuristic
	int U_num_coef = node.U_num_row * node.U_num_col;
	int V_num_coef = node.V_num_row * node.V_num_col;
	MatrixXd U_tmp(U_num_coef , tmp_max_num_term);
	MatrixXd V_tmp(V_num_coef , tmp_max_num_term);
	int tmp_num_term = 0;
	// var across iter
	int crt_blk_row = 0; 
	int crt_blk_col = 0;
	int max_row_idx_l1 = 0;
	int max_col_idx_l1 = 0;
	int max_row_idx_l2 = 0;
	int max_col_idx_l2 = 0;
	while(true)
	{
		int crt_row_offset = row_offset + crt_blk_row*V_num_row;
		int crt_col_offset = col_offset + crt_blk_col*V_num_col;
		MatrixXd resiM = mat.block(crt_row_offset , crt_col_offset , 
			V_num_row , V_num_col);
		for(int i = 0 ; i < tmp_num_term ; i++)
		{
			Map<MatrixXd> ViM(V_tmp.col(i).data() , V_num_row , 
				V_num_col);
			resiM = resiM - ViM * U_tmp(crt_blk_col*node.U_num_row + 
				crt_blk_row , i);
		}
		resiM.cwiseAbs().maxCoeff(&max_row_idx_l2 , &max_col_idx_l2); 
		MatrixXd coefM(node.U_num_row , node.U_num_col);
		for(int i = 0 ; i < node.U_num_row ; i++)
			for(int j = 0 ; j < node.U_num_col ; j++)
			{
				coefM(i,j) = mat(row_offset + i*V_num_row + 
					max_row_idx_l2 , col_offset+j*V_num_col + 
					max_col_idx_l2);
				int U_row_idx = i + j*node.U_num_row;
				int V_row_idx = max_row_idx_l2 + max_col_idx_l2*
					V_num_row;
				coefM(i,j) = coefM(i,j) - U_tmp.row(U_row_idx).
					segment(0,tmp_num_term).dot(V_tmp.row(
					V_row_idx).segment(0,tmp_num_term));
			}
		if(coefM.cwiseAbs().maxCoeff(&max_row_idx_l1 , &max_col_idx_l1) > 
			abs_tol && abs(coefM(crt_blk_row , crt_blk_col)) > 1e-10)
		// add one col to U and V
		{
			U_tmp.col(tmp_num_term) = Map<VectorXd>(coefM.data() , 
				U_num_coef) / coefM(crt_blk_row , crt_blk_col);
			V_tmp.col(tmp_num_term) = Map<VectorXd>(resiM.data() ,
				V_num_coef);
			tmp_num_term++;
			crt_blk_row = max_row_idx_l1;
			crt_blk_col = max_col_idx_l1;
			if(tmp_num_term == U_tmp.cols())
			{
				cout << "Warning: the heuristic maximum number of "
					"terms reached when performing ACA approx " <<
					endl;
				break;
			}
		}else if(abs(coefM(max_row_idx_l1 , max_col_idx_l1)) > abs_tol)
		// continue but does not add to U, V
		{
			crt_blk_row = max_row_idx_l1;
			crt_blk_col = max_col_idx_l1;
			continue;
		}else
		// secondary diag check
		{
			bool pass = true;
			for(int i = blk_sz-1 ; i > (blk_sz>>1) ; i--)
			{
				int j = blk_sz - i;
				double resi = mat(row_offset+i , col_offset+j);
				crt_blk_row = i / V_num_row;
				crt_blk_col = j / V_num_col;
				int crt_row_idx = i % V_num_row;
				int crt_col_idx = j % V_num_col;
				int U_row_idx = crt_blk_row + crt_blk_col*
					node.U_num_row;
				int V_row_idx = crt_row_idx + crt_col_idx*
					V_num_row;
				resi = resi - U_tmp.row(U_row_idx).
					segment(0,tmp_num_term).dot(V_tmp.row(
					V_row_idx).segment(0,tmp_num_term));
				if(abs(resi) > abs_tol)
				{
					pass = false;
					cout << "Secondary diagonal test failed when "
						"building the Kronecker "
						"representation with ACA" << endl;
					break;
				}
			}
			if(pass)
				break;
		}
	} // while(true)
	if(normalize)
	{
		// orthogonalization
		static const double ALMOSTZERO = 1E-8;
		static const double CORR_TOL = 1E-8;
		VectorXd coord(tmp_num_term);
		vector<bool> isBase(tmp_num_term);
		int num_term = 0;
		for(int i = 0 ; i < tmp_num_term ; i++)
		{
			for(int j = 0 ; j < i ; j++)
			{
				coord(j) = V_tmp.col(j).dot(V_tmp.col(i));
				V_tmp.col(i) -= V_tmp.col(j) * coord(j);
				U_tmp.col(j) += U_tmp.col(i) * coord(j);
			}
			double norm = V_tmp.col(i).norm();
			if(norm < ALMOSTZERO)
			{
				U_tmp.col(i).setZero();
				V_tmp.col(i).setZero();
				isBase[i] = false;
			}else
			{
				U_tmp.col(i) = U_tmp.col(i) * norm;
				V_tmp.col(i) = V_tmp.col(i) / norm;
				isBase[i] = true;
				num_term++;
			}
		}
		// build node
		node.num_term = num_term;
		node.U.resize(U_num_coef , num_term);
		node.V.resize(V_num_coef , num_term);
		int count = 0;
		for(int i = 0 ; i < tmp_num_term ; i++)
			if(isBase[i])
			{
				node.U.col(count) = U_tmp.col(i);
				node.V.col(count) = V_tmp.col(i);
				count++;
			}
		// check orthogonality
	        for(int i = 0 ; i < node.num_term ; i++)
			for(int j = 0 ; j < i ; j++)
		                if(abs(node.V.col(i).dot(node.V.col(j))) > CORR_TOL)
		                {
		                        cout << "The orthogonalization for the ACA "
						"bases failed. Exit(1)" << endl;
					exit(1);
		                }
	} // if(normalize)
	else{
		node.U = U_tmp;
		node.V = V_tmp;
		node.num_term = tmp_num_term;
	} // if(!normalize)

	return node;
}

/*
	Same as the definition above but @mat is a function
*/
TreeNode comp_UV_ACA(std::function<double(int,int)> mat , int row_offset , 
	int col_offset , int blk_sz , int V_num_row , int V_num_col , double abs_tol ,
	int max_num_term , bool normalize)
{
	// init
	TreeNode node;
	node.num_term = 0;
	node.V_num_row = V_num_row;
	node.V_num_col = V_num_col;
	node.U_num_row = blk_sz / V_num_row;
	node.U_num_col = blk_sz / V_num_col;
	// tmp var
	int tmp_max_num_term = log2(blk_sz) + 30; // heuristic
	int U_num_coef = node.U_num_row * node.U_num_col;
	int V_num_coef = node.V_num_row * node.V_num_col;
	MatrixXd U_tmp(U_num_coef , tmp_max_num_term);
	MatrixXd V_tmp(V_num_coef , tmp_max_num_term);
	int tmp_num_term = 0;
	// var across iter
	int crt_blk_row = 0; 
	int crt_blk_col = 0;
	int max_row_idx_l1 = 0;
	int max_col_idx_l1 = 0;
	int max_row_idx_l2 = 0;
	int max_col_idx_l2 = 0;
	while(true)
	{
		int crt_row_offset = row_offset + crt_blk_row*V_num_row;
		int crt_col_offset = col_offset + crt_blk_col*V_num_col;
		MatrixXd resiM(V_num_row , V_num_col);
		for(int i = 0 ; i < V_num_row ; i++)
			for(int j = 0 ; j < V_num_col ; j++)
				resiM(i,j) = mat(crt_row_offset+i , crt_col_offset+j);
		for(int i = 0 ; i < tmp_num_term ; i++)
		{
			Map<MatrixXd> ViM(V_tmp.col(i).data() , V_num_row , 
				V_num_col);
			resiM = resiM - ViM * U_tmp(crt_blk_col*node.U_num_row + 
				crt_blk_row , i);
		}
		resiM.cwiseAbs().maxCoeff(&max_row_idx_l2 , &max_col_idx_l2); 
		MatrixXd coefM(node.U_num_row , node.U_num_col);
		for(int i = 0 ; i < node.U_num_row ; i++)
			for(int j = 0 ; j < node.U_num_col ; j++)
			{
				coefM(i,j) = mat(row_offset + i*V_num_row + 
					max_row_idx_l2 , col_offset+j*V_num_col + 
					max_col_idx_l2);
				int U_row_idx = i + j*node.U_num_row;
				int V_row_idx = max_row_idx_l2 + max_col_idx_l2*
					V_num_row;
				coefM(i,j) = coefM(i,j) - U_tmp.row(U_row_idx).
					segment(0,tmp_num_term).dot(V_tmp.row(
					V_row_idx).segment(0,tmp_num_term));
			}
		if(coefM.cwiseAbs().maxCoeff(&max_row_idx_l1 , &max_col_idx_l1) > 
			abs_tol && abs(coefM(crt_blk_row , crt_blk_col)) > 1e-10)
		// add one col to U and V
		{
			U_tmp.col(tmp_num_term) = Map<VectorXd>(coefM.data() , 
				U_num_coef) / coefM(crt_blk_row , crt_blk_col);
			V_tmp.col(tmp_num_term) = Map<VectorXd>(resiM.data() ,
				V_num_coef);
			tmp_num_term++;
			crt_blk_row = max_row_idx_l1;
			crt_blk_col = max_col_idx_l1;
			if(tmp_num_term == U_tmp.cols())
			{
				cout << "Warning: the heuristic maximum number of "
					"terms reached when performing ACA approx " <<
					endl;
				break;
			}
		}else if(abs(coefM(max_row_idx_l1 , max_col_idx_l1)) > abs_tol)
		// continue but does not add to U, V
		{
			crt_blk_row = max_row_idx_l1;
			crt_blk_col = max_col_idx_l1;
			continue;
		}else
		// secondary diag check
		{
			bool pass = true;
			for(int i = blk_sz-1 ; i > (blk_sz>>1) ; i--)
			{
				int j = blk_sz - i;
				double resi = mat(row_offset+i , col_offset+j);
				crt_blk_row = i / V_num_row;
				crt_blk_col = j / V_num_col;
				int crt_row_idx = i % V_num_row;
				int crt_col_idx = j % V_num_col;
				int U_row_idx = crt_blk_row + crt_blk_col*
					node.U_num_row;
				int V_row_idx = crt_row_idx + crt_col_idx*
					V_num_row;
				resi = resi - U_tmp.row(U_row_idx).
					segment(0,tmp_num_term).dot(V_tmp.row(
					V_row_idx).segment(0,tmp_num_term));
				if(abs(resi) > abs_tol)
				{
					pass = false;
//					cout << "Secondary diagonal test failed when "
//						"building the Kronecker "
//						"representation with ACA" << endl;
					break;
				}
			}
			if(pass)
				break;
		}
	} // while(true)
	if(normalize)
	{
		// orthogonalization
		static const double ALMOSTZERO = 1E-8;
		static const double CORR_TOL = 1E-8;
		VectorXd coord(tmp_num_term);
		vector<bool> isBase(tmp_num_term);
		int num_term = 0;
		for(int i = 0 ; i < tmp_num_term ; i++)
		{
			for(int j = 0 ; j < i ; j++)
			{
				coord(j) = V_tmp.col(j).dot(V_tmp.col(i));
				V_tmp.col(i) -= V_tmp.col(j) * coord(j);
				U_tmp.col(j) += U_tmp.col(i) * coord(j);
			}
			double norm = V_tmp.col(i).norm();
			if(norm < ALMOSTZERO)
			{
				U_tmp.col(i).setZero();
				V_tmp.col(i).setZero();
				isBase[i] = false;
			}else
			{
				U_tmp.col(i) = U_tmp.col(i) * norm;
				V_tmp.col(i) = V_tmp.col(i) / norm;
				isBase[i] = true;
				num_term++;
			}
		}
		// build node
		node.num_term = num_term;
		node.U.resize(U_num_coef , num_term);
		node.V.resize(V_num_coef , num_term);
		int count = 0;
		for(int i = 0 ; i < tmp_num_term ; i++)
			if(isBase[i])
			{
				node.U.col(count) = U_tmp.col(i);
				node.V.col(count) = V_tmp.col(i);
				count++;
			}
		// check orthogonality
	        for(int i = 0 ; i < node.num_term ; i++)
			for(int j = 0 ; j < i ; j++)
		                if(abs(node.V.col(i).dot(node.V.col(j))) > CORR_TOL)
		                {
		                        cout << "The orthogonalization for the ACA "
						"bases failed. Exit(1)" << endl;
					exit(1);
		                }
	} // if(normalize)
	else{
		node.U = U_tmp;
		node.V = V_tmp;
		node.num_term = tmp_num_term;
	} // if(!normalize)

	return node;
}
