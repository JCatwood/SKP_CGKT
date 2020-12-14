#include <iostream>
#include <chrono>
#include <fstream>
#include "hlib.hh"
#include "generate_mat.h"
#include "utility.h"
#include "factor_SVD.h"
#include "factor_ACA.h"
#include "common_base_blk_chol.h"

using namespace std;
using namespace HLIB;
using real_t = HLIB::real;

/*
	The non-stationary exponential kernel
*/
class NSExpCovf : public TCoeffFn<real_t> {
private:
        const double beta1;
        const double beta2;
        const vector<T2Point> *xy;
public:
        NSExpCovf(double beta1In, double beta2In, const vector<T2Point> *xyIn)
                : beta1(beta1In), beta2(beta2In), xy(xyIn) {}
        void eval(const vector<idx_t> &rowidxs, const vector<idx_t> &colidxs,
                real_t *matrix ) const
        {
                const size_t  n = rowidxs.size();
                const size_t  m = colidxs.size();
                for ( size_t  j = 0; j < m; ++j )
                {
                    const int  idx1 = colidxs[ j ];
                    for ( size_t  i = 0; i < n; ++i )
                    {
                        const int  idx0 = rowidxs[ i ];
                        double     value;
                        if ( idx0 == idx1 )
                            value = 1.0;
                        else
                        {
                            const double betaI = beta1 + (beta2 - beta1) * (*xy)
                                [idx0][0];
                            const double betaJ = beta1 + (beta2 - beta1) * (*xy)
                                [idx1][0];
                            const double xdiff = (*xy)[idx0][0] - (*xy)[idx1][0];
                            const double ydiff = (*xy)[idx0][1] - (*xy)[idx1][1];
                            double dist = xdiff * xdiff + ydiff * ydiff;
                            dist = 2 * dist / (betaI * betaI + betaJ * betaJ);
                            dist = sqrt(dist);
                            double coef = 2 * betaI * betaJ / (betaI * betaI +
                                betaJ * betaJ);
                            coef = sqrt(coef);
                            value = coef * exp(-dist);
                        }
                        matrix[ j*n + i ] = value;
                    }
                }
        }
        matform_t  matrix_format  () const { return symmetric; }
        bool       is_complex     () const { return false; }
};

/*
	The simulation function to find the memory footprint of the 
	  hierarchical matrix
*/
void H_mem(int kernelType, int s1, int s2, double h)
{
        try
        {
                INIT();
                CFG::set_verbosity( 3 );
                const size_t n = s1 * s2;
		if(h < 0.0)
	                h = 1.0 / double(s1);
                std::vector< T2Point > vertices;
                vertices.resize(n);
                std::vector< double * > verticesCp( n );
                for ( size_t i = 0; i < n; i++ )
                {
                        double x = h * double(i/s2);
                        double y = h * double(i%s2);
                        vertices[i] = T2Point(x, y);
                }
                TCoordinate coord(vertices);
                TAutoBSPPartStrat part_strat;
                TBSPCTBuilder ct_builder( & part_strat );
                auto ct = ct_builder.build( &coord );
                TStdGeomAdmCond adm_cond( 2.0 );
                TBCBuilder bct_builder;
                auto bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
                double sigma = 1.0;
                double beta = 0.3;
                double nu = 0.5;
                TPSMatrixVis  mvis;
                if(kernelType == 1)
                {
                        TMaternCovCoeffFn<T2Point> matern_coefffn(sigma, beta, nu,
                                vertices);
                        TPermCoeffFn< real_t > coefffn( & matern_coefffn,
                                ct->perm_i2e(), ct->perm_i2e() );
                        TACAPlus< real_t > aca( & coefffn );
                        TDenseMBuilder< real_t > h_builder( & coefffn, & aca );
                        auto acc = fixed_prec( 1e-5);
                        auto A = h_builder.build( bct.get(), acc );
                        cout << A->byte_size() << endl;
//        	        mvis.print( A.get(), "tmp.ps" );
                }else
                {
                        NSExpCovf nsExpCovf(0.1, 0.3, &vertices);
                        TPermCoeffFn< real_t > coefffn( & nsExpCovf,
                                ct->perm_i2e(), ct->perm_i2e() );
                        TACAPlus< real_t > aca( & coefffn );
                        TDenseMBuilder< real_t > h_builder( & coefffn, & aca );
                        auto acc = fixed_prec( 1e-5);
                        auto A = h_builder.build( bct.get(), acc );
                        cout << A->byte_size() << endl;
//	                mvis.print( A.get(), "tmp.ps" );
                }
                DONE();
        }
        catch ( Error & e )
        {
                std::cout << e.to_string() << std::endl;
        }

}

