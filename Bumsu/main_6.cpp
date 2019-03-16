#define EIGEN_USE_NEW_STDVECTOR
#include "CD.h"
#include "Timer.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
/**
(D_i)J* ( p )
*/

using namespace Eigen;

template<typename T, int d>
void solve_and_record(int cnt);

int main() {
	/// Basic Parameter Setting
	// floating data type
	using T = double;
	// surprisingly, double is significantly faster!!
	// dimension
	//constexpr int d = 5; // 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
	int cnt = 4;
	std::cout << "Enter cnt: ";
	std::cin >> cnt;
	solve_and_record<T, 2>(cnt);
	solve_and_record<T, 3>(cnt);
	solve_and_record<T, 4>(cnt);
	solve_and_record<T, 5>(cnt);
	solve_and_record<T, 6>(cnt);
	solve_and_record<T, 7>(cnt);
	solve_and_record<T, 8>(cnt);
	solve_and_record<T, 9>(cnt);
	solve_and_record<T, 10>(cnt);
	solve_and_record<T, 11>(cnt);
	solve_and_record<T, 12>(cnt);
	solve_and_record<T, 13>(cnt);
	solve_and_record<T, 14>(cnt);
	solve_and_record<T, 15>(cnt);
	solve_and_record<T, 16>(cnt);
	solve_and_record<T, 17>(cnt);
	solve_and_record<T, 18>(cnt);
	solve_and_record<T, 19>(cnt);
	solve_and_record<T, 20>(cnt);
	std::cout << "cnt was " << cnt << '\n';



	return 0;
}

template<typename T, int d>
void solve_and_record(int cnt) {
	constexpr T dom_min = T(-3);
	constexpr T dom_max = T(3);
	constexpr T dom_ran = dom_max - dom_min;
	// level of spatial discretization. (N+1) points will be used
	constexpr int N = 60;
	// constexpr int N_nonbox = N * 3; // 3 times better resolution (total 3^2 = 9 times )
	// Set Maximum T and discretize time:
	const T T_max = T(0.1);
	int TN = 1;
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

	std::stringstream oss; oss << "Sol_Ex_6_t_0.1_d_" << d << "cnt_" << cnt << ".csv";
	std::ofstream dataout(oss.str());
	std::stringstream oss2; oss2 << "dimension_vs_time" << cnt << ".csv";
	std::ofstream out(oss2.str(), std::fstream::app);
	if (out.is_open() && dataout.is_open()) {
		/// Problem setting
		dMat<T, d> M, NC, ND, QC, QD;
		M.setZero(); NC.setZero();
		for (int i = 0; i < d; ++i) {
			M(i, i) = T(1);
			NC(i, i) = T(1);
			if (i > 0) {
				M(i, i - 1) = T(1);
				M(i - 1, i) = T(1);
				NC(i, i - 1) = T(0.5);
				NC(i - 1, i) = T(0.5);
			}
		} // M, NC set

		ND.setIdentity(); ND *= T(0.1); // ND set

		QC.setZero();
		for (int i = 0; i < d; ++i) {
			QC(i, i) = T(3);
			if (i + 1 < d) {
				QC(i, i + 1) = T(1); QC(i + 1, i) = T(1);
			}
		}
		QC *= 0.1; // set Tri-diagonal QC
		QD.setIdentity(); QD *= T(0.01); // set QD

		dVec<T, d> ac, ad;
		ac.setZero();
		ad.setZero();

		bool b_empty = false, QCt_const = true, QDt_const = true;
		bool act_const = true, adt_const = true;
		//std::cout << " Matrices are initialized. \n";
		//std::cout << M << '\n' << NC << '\n' << ND << '\n' << QC << '\n' << QD << '\n';
		int rand_init_size = 1;

		/// Solve

		// PHI will be the solution
		nMat<T, N + 1> PHI;

		// set seed for random numbers
		std::srand(static_cast<unsigned>(std::time(nullptr)));

		// set example to solve
		std::string ex_num = "6";

		std::cout << " Solving Exmple  " << ex_num << " with d = " << d << " \n";

		bool box_region = true;
		double time_d; // empty vector
		int total_pts = 0;
		CD<T, d, N + 1> EX{ T_max, dom_min, dom_max, rand_init_size,
			Zx, Zy, M, NC, ND, QC, QD, ac, ad, b_empty, QCt_const, QDt_const, act_const, adt_const };
		EX.setL(T(4));

		int tcount = 1;
		// measure time
		simple_timer::timer<'s', long double> tim;

		EX.solve_by_CD(PHI, tvec[0]);

		auto time_tot = tim.tock();

		for (int i = 0; i < N + 1; ++i) {
			for (int j = 0; j < N + 1; ++j) {
				dataout << Zx[i] << ',' << Zy[j] << ',' << PHI(i, j) << '\n';
			}
		}

		total_pts = tvec.size() * (N + 1)*(N + 1);
		time_d = (time_tot.getlen() / double(total_pts)); // time in sec
		std::cout << " Total points = " << tvec.size() << " * " << (N + 1) << "^2 = " << total_pts <<
			"\n and  Elapsed time = " << time_tot << ". ( " << time_tot.getlen() / total_pts * 1000. << " ms/pt )\n" << std::endl;
		out << d << ',' << std::setprecision(9) << time_d << ',' << total_pts << ',' << tvec[0] << '\n';
	}
	else {
		std::cout << "Failed to write! \n";
	}
	out.close();
	dataout.close();
	// std::cin.get();
}