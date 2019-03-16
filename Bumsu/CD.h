#ifndef ___CD_EIG___
#define ___CD_EIG___

#define EIGEN_USE_NEW_STDVECTOR
#define _USE_MATH_DEFINES

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>


/// type definitions
using namespace Eigen;
// vector of size d (=dimension)
template<typename T, int d>
using dVec = Matrix<T, d, 1>;


// vector of dynamic size N
template<typename T>
using VectorN = Matrix< T, Dynamic, 1 >;

// matrix of size d*d ( d = dimension)
template<typename T, int d>
using dMat = Matrix<T, d, d>;


// matrix of size n (=#discretization)
template<typename T, int n>
using nMat = Matrix<T, n, n>;

// array of dVec
template<typename T, int d>
using dVecArr = std::vector<dVec<T, d>, aligned_allocator<dVec<T, d> > >;

// array of dMat
template<typename T, int d>
using dMatArr = std::vector<dMat<T, d>, aligned_allocator<dMat<T, d> > >;

/** Matrix of dVecs (e.g. set of all locations z's)
Saved as a vector (1-dim array). Need to access via [N*(x-idx) + (y-idx)]
having total N^2 elements.
Recall that although the whole domain is d-dimensional, our domain of interest lies on a
plane = {0}^a X [-3,3] X {0}^b X [-3,3] X {0}^c  (a+b+c = d-2).
variables: Zx, Zy  // are stored in a vector Z.
// Z[1] = Z1, Z[2] = Z2. Type of Z is locVar
*/
template<typename T, int n>
using nVec = Matrix<T, n, 1>;

//template <typename T, int n>
//using locVar = std::vector<nVec<T, n>, aligned_allocator<nVec<T, n>>>;

/// non-member functions
///** function matexp : compute the matrix exponential e^tM
//@param1 t : a scalar
//@param2 M : a matrix
//@param3 res : resulting matrix
//*/
//template <typename T, int d>
//void matexp(T t, const dMat<T, d>& M, dMat<T, d>& res);


/** function quad
@param1 A : matrix of size d * d
@param2 x : vector of size d
@return   : <x, Ax>
*/
template <typename T, int d>
T quad(const dMat<T, d>& A, const dVec<T, d>& x);

/** function write_stream
@param1 out : output stream
@param2 Z   : all possible locations ( (Z1,Z2))
@param3 PHI : solution, on [-3,3]^2 only. (i.e. on a plane = {0}^a X [-3,3] X {0}^b X [-3,3] X {0}^c,
			  a+b+c = d-2)
*/
//template <typename T, int n>
//std::ostream& write_stream(std::ostream& out, const nVec<T, n>& Zx, const nVec<T, n>& Zy, const nMat<T, n>& PHI);






/// class CD
template<typename T, int d, int n> // dimension & discretization of domain given when created
class CD {
	// constants
	int Nt;
	const int a, b; // domain (Z): a-th, and b-th dim are nonzero (remember it is on a plane)
	const T T_max; // must be given
	T L; // Lipschitz constant
	T tol, peps;	int maxit; // default parameters for iterations
	const T dom_min, dom_max; // [-3, 3]
	const int rand_init_size;
	//unsigned long long int itcnt = 0; 
	bool b_empty;
	// matrices
	const dMat<T, d>& M, NC, ND, QC, QD; // must be given
	dMat<T, d> NCt, NDt, QCt, QDt; // time-dependent case
	bool QCt_const, QDt_const;

	/**
	A = - e^(-(T-t)M)*NC, B = - e^(-(T-t)M)*ND,
	AQAT_BQBT = A*Q_C*A^T, BQBY = B*Q_D*B^T, AT = A^T, BT = B^T.
	 They are arrays (in time). Computed in compute_matrices
	*/
	dMatArr<T, d> AQAT, BQBT;

	//vectors
	const nVec<T, n>& Zx, Zy; // must be given
	const dVec<T, d>& ac, ad; // must be given
	dVec<T, d> act, adt; // time-dependent case
	bool act_const, adt_const;
	dVec<T, d> diagA; // here A means J = 1/2( <x,Ax>-1)
	std::vector<std::pair<T, T>>&& Z; // if Z is given, Zx and Zy must not be given and not used.
	bool regional = false;
	/**
	AacBad = A*ac + B*ad. (time-dependent)
	convenient to have for computing integarl of H.
	*/
	dVecArr<T, d> AacBad;
public:
	/** Constructor
	@template parameters: T: double/float, d: dimension, n: level of discretization (space)
	@params: Z, M, NC, ND, ac, ad also must be given
	other members are default constructed
	*/
	CD(T _T, T _dmin, T _dmax, int _rand_init_size,
		const nVec<T, n>& _Zx, const nVec<T, n>& _Zy,
		const dMat<T, d>& _M, const dMat<T, d>& _NC, const dMat<T, d>& _ND,
		const dMat<T, d>& _QC, const dMat<T, d>& _QD,
		const dVec<T, d>& _ac, const dVec<T, d>& _ad,
		bool _bempty, bool _qct_const, bool _qdt_const, bool _act_const, bool _adt_const)
		: T_max(_T), a(0), b(1), dom_min(_dmin), dom_max(_dmax), rand_init_size(_rand_init_size),
		tol(T(0.5e-7)), Nt(10), maxit(500), L(T(4)), peps(0.0025), 
		Zx(_Zx), Zy(_Zy), M(_M),
		NC(_NC), ND(_ND), QC(_QC), QD(_QD), ac(_ac), ad(_ad),
		NCt(_NC), NDt(_ND), QCt(_QC), QDt(_QD), act(_ac), adt(_ad),
		b_empty(_bempty), QCt_const(_qct_const), QDt_const(_qdt_const),
		act_const(_act_const), adt_const(_adt_const),
		regional(false), Z(std::vector<std::pair<T, T>>{}) // initialize Z to be an empty one
		// initialize t-dep. mat/vec
	{
		diagA[0] = static_cast<T>(1.), diagA[1] = static_cast<T>(25. / 4.);
		for (int i = 2; i < d; ++i) {
			diagA[i] = static_cast<T>(.5);
		}
		//std::cout << "CD object is created. \n";
	};
	/** constructor getting a specific region other than a box */
	//CD(T _T, T _dmin, T _dmax, std::vector<std::pair<T, T>>& Z,
	//	const dMat<T, d>& _M, const dMat<T, d>& _NC, const dMat<T, d>& _ND,
	//	const dMat<T, d>& _QC, const dMat<T, d>& _QD,
	//	const dVec<T, d>& _ac, const dVec<T, d>& _ad,
	//	bool _bempty, bool _qct_const, bool _qdt_const, bool _act_const, bool _adt_const)
	//	: T_max(_T), a(0), b(1), dom_min(_dmin), dom_max(_dmax),
	//	tol(T(0.5e-7)), Nt(10), maxit(500), L(T(4)), peps(0.0025), rand_init_size(10),
	//	M(_M), NC(_NC), ND(_ND), QC(_QC), QD(_QD), ac(_ac), ad(_ad),
	//	NCt(_NC), NDt(_ND), QCt(_QC), QDt(_QD), act(_ac), adt(_ad),
	//	b_empty(_bempty), QCt_const(_qct_const), QDt_const(_qdt_const),
	//	act_const(_act_const), adt_const(_adt_const),
	//	regional(true)
	//	// initialize t-dep. mat/vec
	//{
	//	// 5 pts/ 0.1 s
	//	diagA[0] = static_cast<T>(1.), diagA[1] = static_cast<T>(25. / 4.);
	//	for (int i = 2; i < d; ++i) {
	//		diagA[i] = static_cast<T>(.5);
	//	}
	//	//std::cout << "CD object is created. \n";
	//};


	/// functions given in problem
	/** Function J, Jstar, H : as in the paper

	*/
	T Jstar(const dVec<T, d>& p);
	T J(const dVec<T, d>& p);
	/**  dt = t/Nt.
		returns H( ti*dt, p);
		CAUTION: value of H depends on Nt
	*/
	T H(int ti, const dVec<T, d>& p);
	/** Function F = Jstar(p) + \int_0^t H(s,p) ds - <z,p>
				Note that phi(t,p) = -F(t,z,p*),  where p* is the solution dF = 0
	*/
	T F(T t, const dVec<T, d>& p, const dVec<T, d> z);
	/** Function dF : Finite diefference of F,
		=[F(s, p + eps * e_i) - F(s, p)] / eps
	*/
	T dF(T t, const dVec<T, d>& p, const dVec<T, d> z, int i);

	void solve_by_CD(nMat<T, n>& PHI, T t);
	void solve_by_CD(std::vector<T>& PHI, T t);

	void compute_matrices(T t); // compute A and ATQA at each t, this function must be called before calling H and DH

	// fortunately, NC and ND are always constant in our example
	//void compute_NCt(T t);
	//void compute_NDt(T t);
	void compute_QCt(T t);
	void compute_QDt(T t);
	void compute_act(T t);
	void compute_adt(T t);


	//double ddF(double t, const dvec& p, const dvec& z, int i, double dp);
	// bool Find_dpi(double& dp, const dvec& p, double t, const dvec& z, int i, double alpha);

	/** function initial fills PHI with the initial condition
	@param : the solution PHI at initial time (t=0)
	*/
	void initial(nMat<T, n>& PHI); // region is a box
	void initial(std::vector<T>& PHI); // region is given as a vector of pts

	/** Helper functions*/
	/** function rvec : fill the argument with uniform random elements in [min, max]
	@param p : vector to be filled
	@param min, max : range
	*/
	void rvec(dVec<T, d>& p);
	/*T get_avgit() {
		return static_cast<T>(itcnt) / static_cast<T>(n) / static_cast<T>(n)/ static_cast<T>(rand_init_size);
	}*/

	/// mutators
	void setL(T);
};

/// member function definitions
/**
First, compute_@@t functions: updates matrix/vector @@ for given t
*/
/** Fortunately, our NC and ND are always constant. */
//template<typename T, int d, int n>
//void CD<T, d, n>::compute_NCt(T t) {
//	// problem dependent
//	NCt = Nc;
//}
//template<typename T, int d, int n>
//void CD<T, d, n>::compute_NDt(T t) {
//	// problem dependent
//	NCt = Nc;
//}
template<typename T, int d, int n>
void CD<T, d, n>::compute_QCt(T t) { // compute Q_C(t)
	if (QCt_const) {
		// do nothing
	}
	else {
		QCt = T(std::exp(-2 * t))*QC;
	}
}
template<typename T, int d, int n>
void CD<T, d, n>::compute_QDt(T t) { // compute Q_D(t)
	if (QDt_const) {
		// do nothing
	}
	else {
		QDt = T(std::exp(2 * t))*QD;
	}
}
template<typename T, int d, int n>
void CD<T, d, n>::compute_act(T t) { // compute a_c(t)
	if (act_const) {
		//do nothing
	}
	else { // problem dependent
		// note that ac != act at time 0
		act[a] = ac[a] - T(0.1*std::cos(M_PI*t / T_max)); 
		act[b] = ac[b] - T(0.1*std::sin(M_PI*t / T_max));
	}
}
template<typename T, int d, int n>
void CD<T, d, n>::compute_adt(T t) { // compute a_d(t)
	if (adt_const) {
		//do nothing
	}
	else { // problem dependent
		adt[a] = ad[a] - T(0.1*std::cos(M_PI*t / T_max));
		adt[b] = ad[b] - T(0.1*std::sin(M_PI*t / T_max));
	}
}

// J*(y) = 1/2* (<y, A^-1 y> + 1)
template<typename T, int d, int n>
T CD<T, d, n>::Jstar(const dVec<T, d>& p) {
	T sum = T(0);
	for (int i = 0; i < d; ++i) {
		sum += p[i] * p[i] / diagA[i];
	}
	return (sum + T(1))*T(0.5);
}

// J(x) = 1/2* (<x,Ax> - 1)
template<typename T, int d, int n>
T CD<T, d, n>::J(const dVec<T, d>& p) {
	T sum = T(0);
	for (int i = 0; i < d; ++i) {
		sum += diagA[i] * p[i] * p[i];
	}
	return (sum - T(1))*T(.5);
}

template<typename T, int d, int n>
T CD<T, d, n>::H(int ti, const dVec<T, d>& p) { // H at time ti*dt
	if (b_empty) { // no b-dep.
		return std::sqrt(quad(AQAT[ti], p)) + p.dot(AacBad[ti]);
	}
	else {
		return std::sqrt(quad(AQAT[ti], p)) - std::sqrt(quad(BQBT[ti], p))
			+ p.dot(AacBad[ti]);
	}
}

template<typename T, int d, int n>
T CD<T, d, n>::F(T t, const dVec<T, d>& p, const dVec<T, d> z) {
	T dt = t / Nt;
	T cumsum = T(0), integ=T(0);
	dVec<T, d> cumvec; cumvec.setZero();
	for (int ti = 0; ti < Nt; ++ti) {
		//integ += H(ti*t / Nt, p);
		//// or the below:
		if (b_empty) { // no b-dep.
			cumsum += std::sqrt(quad(AQAT[ti], p));
		}
		else {
			cumsum += std::sqrt(quad(AQAT[ti], p)) - std::sqrt(quad(BQBT[ti], p));
		}
		cumvec += AacBad[ti];
		//// this should be slightly faster
	}
	//return Jstar(p) - p.dot(z) + integ*dt;
	return Jstar(p) - p.dot(z - cumvec * dt) + cumsum * dt;
}

template<typename T, int d, int n>
T CD<T, d, n>::dF(T t, const dVec<T, d>& p, const dVec<T, d> z, int i) {
	T sum = p[i] / diagA[i] - z[i]; // D_i Jstar (p) - D_i < z,p>
	// d_i (\int H) is left
	T cumsum = T(0), dt = t / Nt;
	dVec<T, d> p2(p);  p2[i] += peps;
	for (int ti = 0; ti < Nt; ++ti) {
		cumsum += H(ti, p2) - H(ti, p);
	}
	return sum + (cumsum * (dt / peps));
}


// if region is given, as a vector of locations, PHI should be just a vector
template<typename T, int d, int n>
void CD<T, d, n>::solve_by_CD(std::vector<T>& PHI, T t) {
	int n2 = Z.size();
	if (PHI.size() != n2) {
		PHI = std::vector<T>(n2);
	}
	// set Nt
	Nt = static_cast<int>((t + tol) / 0.005);
	if (Nt > 100) { Nt = 100; } // truncate if Nt too large
	compute_matrices(t);
	//std::cout << "\t solv - compute_matrices done\n";

	// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
	// parallelize this double for-loop!
#pragma omp parallel for schedule(dynamic)
	for (int loc = 0; loc < n2; ++loc) { // instead of double for loop

		dVec<T, d> z; z.setZero(); z[a] = Z[loc].first; z[b] = Z[loc].second; // for each location

		// can also parallelize doing this!
		T phival = std::numeric_limits<T>::max(); // phi value at (t,z). easy to detect error
												  // randomly initialize p
		for (int kk = 0; kk < rand_init_size; ++kk) { // for each random p,
			int cnt = 0; // counts converged coordinates
			dVec<T, d> p; rvec(p); // random p at this stage
			T alpha = T(1.) / L; // 1/L (Lipschitz const)
			bool it_flag = true; // becomes false if stopping criteria is met
								 // coordinate descent
			for (int itnum = 0; itnum < maxit && it_flag; ++itnum) {
				for (int i = 0; i < d && it_flag; ++i) { // for each coordinate
					T dp = -alpha * dF(t, p, z, i);
					p[i] += dp;
					//std::cout << "\t solv - dF is computed\n";
					if (std::abs(dp) > tol) {
						cnt = 0;
						if (itnum == maxit - 1) { // if reached maxit,
							itnum = 0;
							// set a new Lipschitz constant
							alpha /= T(2);
						}
					}
					else {
						//std::cout << "\t solv - dF < tol \n";
						cnt++;
						if (cnt == d) {
							it_flag = false;
							//itcnt += itnumall;
							T Ftemp = F(t, p, z); // new F value
							if (Ftemp < phival) {
								phival = Ftemp; // set to minimizing one
							}
						}
					}
					//std::cout << "\t solv - iteration, i = " << ii<< " j = " << jj << "\n";
				}
			}
		}
		PHI(loc) = -phival;
		//}
		// std::cout << " i = " << ii << '\n';
		//}
	}
	std::cout << " t = " << t << " done !" << std::endl; // flush buffer

}

// if region is given as a box
template<typename T, int d, int n>
void CD<T, d, n>::solve_by_CD(nMat<T, n>& PHI, T t) {
	//itcnt = 0;

	// set Nt
	T ddt = 0.02; // for samll t (< 0.1), use smaller ddt. This gives 5pts/0.1s
	Nt = static_cast<int>((t + tol) / ddt); 
	if (Nt > 50) { Nt = 50; } // truncate if Nt too large
	compute_matrices(t);
	//std::cout << "\t solv - compute_matrices done\n";

	// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
	// parallelize this double for-loop!
	int n2 = n * n;
#pragma omp parallel for schedule(dynamic)
	for (int iijj = 0; iijj < n2; ++iijj) { // instead of double for loop
		/*if(iijj==0){
			std::cout << "There are " << omp_get_num_threads() << " threads \n";
		}*/
		int ii = iijj / n; // x idx
		int jj = iijj % n; // y idx
		/*for (int ii = 0; ii < n; ++ii) {
			for (int jj = 0; jj < n; ++jj) {*/
			// set z (location)
		dVec<T, d> z; z.setZero(); z[a] = Zx[ii]; z[b] = Zy[jj];

		// can also parallelize doing this!
		T phival = std::numeric_limits<T>::max(); // phi value at (t,z). easy to detect error
		// randomly initialize p
		for (int kk = 0; kk < rand_init_size; ++kk) { // for each random p,
			int cnt = 0; // counts converged coordinates
			dVec<T, d> p; rvec(p); // random p at this stage
			T alpha = T(1.) / L; // 1/L (Lipschitz const)
			bool it_flag = true; // becomes false if stopping criteria is met
			// coordinate descent
			for (int itnum = 0; itnum < maxit && it_flag; ++itnum) {
				for (int i = 0; i < d && it_flag; ++i) { // for each coordinate
					T dp = -alpha * dF(t, p, z, i);
					p[i] += dp;
					//std::cout << "\t solv - dF is computed\n";
					if (std::abs(dp) > tol) {
						cnt = 0;
						if (itnum == maxit - 1) { // if reached maxit,
							itnum = 0;
							// set a new Lipschitz constant
							alpha /= T(2);
						}
					}
					else {
						//std::cout << "\t solv - dF < tol \n";
						cnt++;
						if (cnt == d) {
							it_flag = false;
							//itcnt += itnumall;
							T Ftemp = F(t, p, z); // new F value
							if (Ftemp < phival) {
								phival = Ftemp; // set to minimizing one
							}
						}
					}
					//std::cout << "\t solv - iteration, i = " << ii<< " j = " << jj << "\n";
				}
			}
		}
		PHI(ii, jj) = -phival;
		//}
		// std::cout << " i = " << ii << '\n';
	//}
	}
	std::cout << " t = " << t << " done !" << std::endl; // flush buffer
}




template<typename T, int d, int n>
void CD<T, d, n>::compute_matrices(T t) {
	//std::cout << "\t solv - compute matrice - start \n";
	if (b_empty) { // no b-dep.
		AQAT = dMatArr<T, d>(Nt);
		AacBad = dVecArr<T, d>(Nt);
		T dt = t / Nt;
		for (int i = 0; i < Nt; ++i) {
			dMat<T, d> expM = -(T_max - i * dt)*M;
			expM = expM.exp();
			//std::cout << "\t solv - compute matrices - Matrix exp done. \n";
			// compute_NCt(T_max - i * dt);
			dMat<T, d> A = -expM * NCt;
			//std::cout << "\t solv - compute matrices - A declared. \n";
			compute_QCt(T_max - i * dt);
			//std::cout << "\t solv - compute matrices - Qct computed. \n";
			AQAT[i] = A * QCt * (A.transpose());
			//std::cout << "\t solv - compute matrices - AQAT computed. \n";
			compute_act(T_max - i * dt);
			//std::cout << "\t solv - compute matrices - act computed. \n";
			AacBad[i] = A * act; // compute A*a_c(T-t)
			//std::cout << "\t solv - compute matrices - AacBad computed. \n";
		}
		// avgAacBad *= dt; // get average
	}
	else {
		AQAT = dMatArr<T, d>(Nt);
		BQBT = dMatArr<T, d>(Nt);
		AacBad = dVecArr<T, d>(Nt);
		T dt = t / Nt;
		for (int i = 0; i < Nt; ++i) {
			dMat<T, d> expM = -(T_max - i * dt)*M;
			expM = expM.exp();
			//compute_NCt(T_max - i * dt);
			//compute_NDt(T_max - i * dt);
			dMat<T, d> A = -expM * NCt;
			dMat<T, d> B = -expM * NDt;
			compute_QCt(T_max - i * dt);
			compute_QDt(T_max - i * dt);
			AQAT[i] = A * QCt * (A.transpose());
			BQBT[i] = B * QDt * (B.transpose());
			compute_act(T_max - i * dt);
			compute_adt(T_max - i * dt);
			AacBad[i] = (A*act + B * adt); // at time T-t.
		}
		// avgAacBad *= dt; // get average
	}
}


template<typename T, int d, int n>
void CD<T, d, n>::initial(nMat<T, n>& PHI) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			// z is a point on the domain
			dVec<T, d> z;
			z.setZero();
			z[a] = Zx[i];
			z[b] = Zy[j];
			PHI(i, j) = J(z); // initial value
		}
	}
}

template<typename T, int d, int n>
void CD<T, d, n>::initial(std::vector<T>& PHI) {
	auto n2 = Z.size();
	for (int loc = 0; loc < n2; ++loc) {
		// z is a point on the domain
		dVec<T, d> z;
		z.setZero();
		z[a] = Z[loc].first;
		z[b] = Z[loc].second;
		PHI[loc] = J(z); // initial value
	}
}


/// non-member function definitions


//template <typename T, int n>
//std::ostream& write_stream(std::ostream& out, const nVec<T, n>& Zx, const nVec<T, n>& Zy, const nMat<T, n>& PHI) {
//	for (int p = 0; p < n; ++p) {
//		for (int q = 0; q < n; ++q) {
//			out << Zx[p] << ", " << Zy[q] << ", "
//				<< PHI(p, q) << '\n';
//		}
//	}
//	return out;
//}


template <typename T, int d, int n>
void CD<T, d, n>::rvec(dVec<T, d>& p) {
	// uniform with rmin ~ rmax
	T rrange = dom_max - dom_min;
	for (int i = 0; i < d; ++i) {
		p[i] = dom_min + (static_cast<T>(std::rand())*rrange / RAND_MAX);
	}
}

// returns <x,Ax>
template <typename T, int d>
T quad(const dMat<T, d>& A, const dVec<T, d>& x) {
	return x.dot(A*x); // <x, Ax>
}


template <typename T, int d, int n>
void CD<T, d, n>::setL(T newL) {
	L = newL;
}


#endif