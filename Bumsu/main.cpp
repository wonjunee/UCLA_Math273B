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

//template<typename T>
//bool in_region(T x, T y, T t) {
//	if (t < 0.1) {
//		T e = x * x*T(4. / 25.) + y * y;
//		return (e < T(2)) && (e > T(0.5));
//	}
//	
//	return true;
//}
//template<typename T>
//void set_Z(std::vector<std::pair<T, T>>& Z, int N, T t, T dom_min, T dom_max) {
//	Z = std::vector<std::pair<T, T>>(0);
//	Z.reserve(N*N);
//	auto dom_ran = dom_max - dom_min;
//	int N2 = 3 * N + 1;
//	for (size_t p = 0; p < N2; ++p) { // 3 times better resolution
//		for (size_t q = 0; q < N2; ++q) { // in each direction
//			T x = dom_min + dom_ran / N2 * p;
//			T y = dom_min + dom_ran / N2 * q;
//			if (in_region(x, y, t)) {
//				Z.push_back(std::pair<T, T>(x, y));
//			}
//		}
//	}
//}


int main() {
	/// Basic Parameter Setting
	// floating data type
	using T = double;
	// surprisingly, double is significantly faster!!
	// dimension
	constexpr int d = 6;
	constexpr T dom_min = T(-3);
	constexpr T dom_max = T(3);
	constexpr T dom_ran = dom_max - dom_min;
	// level of spatial discretization. (N+1) points will be used
	constexpr int N = 60;
	// constexpr int N_nonbox = N * 3; // 3 times better resolution (total 3^2 = 9 times )
	// Set Maximum T and discretize time:
	const T T_max = T(0.5);
	int TN = 20;
	std::vector<T> tvec(TN); // tvec{ 0.1, 0.3, 0.5 };
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

	//std::cout << "Parameter initialization done \n";
	//for (auto t : tvec) { std::cout << t << ' '; } std::cout << '\n';
	//for (size_t i = 0; i < d; ++i) { std::cout << Zx[i] << ' '; } std::cout << '\n';
	/// Problem setting
	dMat<T, d> M, NC, ND, QC, QD;
	//M << -20, 0, 0, 0, 0, 0,
	//	0, -25, 0, 0, 0, 0,
	//	0, 0, 0, 0, 0, 0,
	//	-.744, -.032, 0, -.154, -.0052, 1.54,
	//	-.02, 0, .0386, -.996, -.000295, -.117; // 6 by 6
	M.setZero();
	M(0, 0) = -20; M(1, 1) = -25;
	M(3, 0) = -.744; M(3, 1) = -.032; M(3, 3) = -.154; M(3, 4) = -.0052; M(3, 5) = 1.54;
	M(4, 0) = .337; M(4, 1) = -1.12; M(4, 3) = .249; M(4, 4) = -.1; M(4, 5) = -5.2;
	M(5, 0) = -.02; M(5, 2) = .0386; M(5, 3) = -.996; M(5, 4) = -.000295; M(5, 5) = -.0117;
	M *= T(0.1); //M
	NC.setZero(); NC(0, 0) = T(0.1 * 20); NC(1, 1) = T(0.1 * 25); //NC
	ND.setIdentity(); ND *= T(.1); //ND
	QC.setZero(); QD.setZero();
	for (int i = 0; i < d; ++i) {
		QC(i, i) = T(.1 * 3);
		QD(i, i) = T(.1);
		if (i + 1 < d) {
			QC(i, i + 1) = T(.1); QC(i + 1, i) = T(.1);
			QD(i, i + 1) = T(.025); QD(i + 1, i) = T(.025);
		}
	} // set Tri-diagonal QC and QD ( at t = 0 )

	dVec<T, d> ac, ad;
	ac.setZero(); ac(0) = -10; ac(1) = -5; ac *= .1;
	ad.setZero(); ad(0) = -10; ad(1) = -5; ad *= .1;
	bool b_empty = false, QCt_const = false, QDt_const = false;
	bool act_const = false, adt_const = false;
	//std::cout << " Matrices are initialized. \n";
	//std::cout << M << '\n' << NC << '\n' << ND << '\n' << QC << '\n' << QD << '\n';
	int rand_init_size = 1;

	/// Solve

	// PHI will be the solution
	nMat<T, N + 1> PHI;

	// set seed for random numbers
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	// set example to solve
	std::string ex_num = "5.1";
	std::cout << " Enter example number: (5.1 or 5.2): ";
	std::getline(std::cin, ex_num);

	bool box_region = true;

	if (box_region) {
		std::cout << " Solving Exmple  " << ex_num << ": \n";
		if (ex_num == "5.1") {

			b_empty = true, QCt_const = true, QDt_const = true;
			act_const = true, adt_const = true; rand_init_size = 1;
		}
		else if (ex_num == "5.2") {
			b_empty = false, QCt_const = false, QDt_const = false;
			act_const = false, adt_const = false; rand_init_size = 20;
		}
		CD<T, d, N + 1> EX{ T_max, dom_min, dom_max, rand_init_size,
			Zx, Zy, M, NC, ND, QC, QD, ac, ad, b_empty, QCt_const, QDt_const, act_const, adt_const };
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
				std::stringstream oss;
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
	/*
	else { // region is not a box
		std::cout << " Solving Exmple  " << ex_num << ": \n";
		if (ex_num == "5.1") {

			b_empty = true, QCt_const = true, QDt_const = true;
			act_const = true, adt_const = true;
		}
		else if (ex_num == "5.2") {
			b_empty = false, QCt_const = false, QDt_const = false;
			act_const = false, adt_const = false;
		}
		// set Z
		std::vector<std::pair<T, T>> Z;
		CD<T, d, N + 1> EX{ T_max, dom_min, dom_max,
			Z, M, NC, ND, QC, QD, ac, ad, b_empty, QCt_const, QDt_const, act_const, adt_const };
		// Z is a reference, so we can change it outside.
		{ // initial
			set_Z(Z, N, 0, dom_min, dom_max);
			// save initial data
			std::string tstr = "0.0";
			std::ofstream out("Sol_Ex_" + ex_num + "_" + tstr + ".csv");
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
		{ // solve for other t >0
			//std::cout << " Solution initialization done. Start solving system... \n";
			int total_pts=0;
			// measure time
			simple_timer::timer<'s', long double> tim;
			for (auto t : tvec) {
				set_Z(Z, N, t, dom_min, dom_max); // set new region
				std::stringstream oss;
				oss << "Sol_Ex_" << ex_num << "_" << t << ".csv";
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
				total_pts += Z.size();
			}
			auto time_tot = tim.tock();
			std::cout << " Total points = " << total_pts <<
				"\n and  Elapsed time = " << time_tot << ". ( " << time_tot.getlen() / total_pts * 1000. << " ms/pt )\n" << std::endl;
		}
	}
	*/

	std::cin.get();

	return 0;
}