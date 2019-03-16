#define EIGEN_USE_NEW_STDVECTOR
#include "CD.h"
#include "Timer.h"
#include <fstream>
#include <sstream>
#include <ctime>
/**
(D_i)J* ( p )
*/

using namespace Eigen;


int main() {
	/// Basic Parameter Setting
	// floating data type
	using T = double;
	// surprisingly, double is significantly faster!!
	// dimension
	constexpr int d = 20;
	constexpr T dom_min = T(-3);
	constexpr T dom_max = T(3);
	constexpr T dom_ran = dom_max - dom_min;
	// level of spatial discretization. (N+1) points will be used
	constexpr int N = 10;
	// constexpr int N_nonbox = N * 3; // 3 times better resolution (total 3^2 = 9 times )
	// Set Maximum T and discretize time:
	const T T_max = T(0.5);
	int TN = 5;
	std::vector<T> tvec(TN);
	for (int i = 0; i < TN; ++i) {
		tvec[i] = (i + 1)*T_max / TN;
	}
	// discretization of space
	nVec<T, N + 1> Zx, Zy;
	// initialize Zx, Zy
	for (int p = 0; p < N + 1; ++p) {
		Zx[p] = dom_min + dom_ran / N * p;
		Zy[p] = Zx[p];
	}


	/// Problem setting
	dMat<T, d> M, NC, ND, QC, QD;
	M.setZero();
	M(0, 1) = 1;
	M(1, 0) = -62500; M(1, 1) = -250;
	M(2, 3) = 1;
	M(3, 1) = -2450000; M(3, 3) = -130;
	M(4, 2) = -10000; M(4, 4) = -2;
	M(5, 2) = 900; M(5, 6) = 1;
	M(6, 3) = -7260000; M(6, 5) = -62500; M(6, 6) = -1;
	M(7, 8) = 1;
	M(8, 4) = 40000;  M(8, 5) = 40000; M(8, 7) = -40000; M(8, 8) = -200;
	M(9, 7) = 1;
	M *= T(0.001); //M set
	NC.setZero(); NC(1, 0) = T(1); NC(3, 0) = T(10000); NC *= T(0.001); //NC
	ND.setZero();
	QC.setZero(); QD.setZero();
	for (int i = 0; i < d; ++i) {
		QC(i, i) = T(3);
		if (i + 1 < d) {
			QC(i, i + 1) = T(1); QC(i + 1, i) = T(1);
		}
	}
	QC *= 0.001;
	// set Tri-diagonal QC and QD ( at t = 0 )

	dVec<T, d> ac, ad;
	ac.setZero(); ac(0) = -10; ac(1) = -5; ac *= .001;
	ad.setZero(); 
	bool b_empty = true, QCt_const = true, QDt_const = true;
	bool act_const = true, adt_const = true;
	// std::cout << " Matrices are initialized. \n";
	// std::cout << M << '\n' << NC << '\n' << ND << '\n' << QC << '\n' << QD << '\n';
	int rand_init_size = 1;

	/// Solve

	// PHI will be the solution
	nMat<T, N + 1> PHI;

	// set seed for random numbers
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	// set example to solve
	std::string ex_num = "5.3";

	bool box_region = true;

	if (box_region) {
		std::cout << " Solving Exmple  " << ex_num << ": \n";
		CD<T, d, N + 1> EX{ T_max, dom_min, dom_max, rand_init_size,
			Zx, Zy, M, NC, ND, QC, QD, ac, ad, b_empty, QCt_const, QDt_const, act_const, adt_const };
		EX.setL(T(128.)); // 2^7
		{ // set initial
			std::string tstr = "0.0";
			std::ofstream out("Sol_Ex_" + ex_num + "_00_" + tstr + ".csv");
			if (out.is_open()) {
				EX.initial(PHI);
				// write_stream(out, Zx, Zy, PHI);
				for (int i = 0; i < N + 1; ++i) {
					for (int j = 0; j < N + 1; ++j) {
						out << Zx[i] << ',' << Zy[j] << ',' << PHI(i, j) << '\n';
					}
				}
			}
			else {
				std::cout << "Fail to write at t = 0. \n";
			}
			out.close();
		}
		{	// solve for t>0
			int tcount = 1;
			// measure time
			simple_timer::timer<'s', long double> tim;
			for (auto t : tvec) {
				std::stringstream oss; // set file name
				if (tcount < 10) { // pad 0
					oss << "Sol_Ex_" << ex_num << "_0" << tcount << "_" << t << ".csv";
				}
				else {
					oss << "Sol_Ex_" << ex_num << "_" << tcount << "_" << t << ".csv";
				}
				std::ofstream out(oss.str());
				if (out.is_open()) {
					EX.solve_by_CD(PHI, t);
					//write_stream(out, Zx, Zy, PHI);
					for (int i = 0; i < N + 1; ++i) {
						for (int j = 0; j < N + 1; ++j) {
							out << Zx[i] << ',' << Zy[j] << ',' << PHI(i, j) << '\n';
						}
					}
					//std::cout << " average iterations / pt = " << EX.get_avgit() << '\n';
				}
				else {
					std::cout << "Fail to write at t = " << t << ". \n";
				}
				out.close();
				tcount++;
			}

			auto time_tot = tim.tock();
			int total_pts = tvec.size() * (N + 1)*(N + 1);
			std::cout << " Total points = " << tvec.size() << " * " << (N + 1) << "^2 = " << total_pts <<
				"\n and  Elapsed time = " << time_tot << ". ( " << time_tot.getlen() / total_pts * 1000. << " ms/pt )\n" << std::endl;
		}
	}

	std::cin.get();

	return 0;
}