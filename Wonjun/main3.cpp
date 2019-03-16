#include <iostream>
#include <fstream>
#include <time.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <vector>
#include <Eigen/Dense>

using namespace std;
class CD
{
	double T;
	double small_t;

	Eigen::Matrix<double, 2, 2> M;
	Eigen::Matrix<double, 2, 2> N_C;
	Eigen::Matrix<double, 2, 2> N_D;
	Eigen::Matrix<double, 2, 2> Q_C;
	Eigen::Vector2d a_C;
	Eigen::Matrix<double, 2, 2> Q_D;
	Eigen::Vector2d a_D;

	int Nt;
	double ds;
	double sigma;
	int count;

	int maxit;
	int dim;
	double eps;
	std::vector<Eigen::Matrix<double, 2, 2> > ACiT_array;
	std::vector<Eigen::Matrix<double, 2, 2> > ADiT_array;

public:
	CD(double small_t_, double T_)
	{
		T = T_;
		small_t = small_t_;

		M << 0, 1,
			 -2, -3;

		N_C << 0.5,  0,
			     0, 0.5;

		N_D << 1, 0,
			   0, 0.5;

		Q_C << 0.3, 0.1,
			   0.1, 0.3;

		Q_D << 0.4, 0.2,
			   0.2, 0.4;

		a_C << -0.5, -0.75;
		a_D << 0.5, 0;

		Nt = 10;
		ds = 1.0 * small_t / Nt;

		sigma = 0.01;
		count = 0;
		dim = 2;
		eps = 0.5e-7;
		maxit = 500;

		for(int j=0; j<Nt; ++j)
		{
			Eigen::Matrix<double, 2, 2> A = -(T-j*ds) * M;
			A = A.exp();
			double tmp = A(0,1);
			A(0,1) = A(1,0);
			A(1,0) = tmp;
			Eigen::Matrix<double, 2, 2> A_C = -A * N_C;
			Eigen::Matrix<double, 2, 2> A_D = -A * N_D;
			ACiT_array.push_back(A_C.transpose());
			ADiT_array.push_back(A_D.transpose());
		}
	}

	double dFi(Eigen::Vector2d& p, Eigen::Vector2d& z, int i)
	{
		Eigen::Vector2d piplus(2);
		if(i==0)
		{
			piplus(0) = p(0)+ sigma;
			piplus(1) = p(1) ;
		}
		else
		{
			piplus(0) = p(0);
			piplus(1) = p(1) + sigma;
		}

		double Jpart;

		if(i==0)
		{
			Jpart = piplus(0)*piplus(0) - p(0)*p(0);
		}
		else
		{
			Jpart = 4.0/25.0 * (piplus(1)*piplus(1) - p(1)*p(1));
		}
		Jpart *= 0.5;

		double Hpart = 0;
		for(unsigned int j=0; j< Nt; ++j)
		{
			Hpart += H(piplus,j) - H(p,j);
		}
		return (Jpart + Hpart * ds) / sigma - z(i);
	}

	double H(Eigen::Vector2d& p, int j)
	{
		Eigen::Matrix<double, 2, 2> ACiT = ACiT_array[j];
		Eigen::Matrix<double, 2, 2> ADiT = ADiT_array[j];
		Eigen::Vector2d ac = ACiT * p;
		Eigen::Vector2d ad = ADiT * p;
		Eigen::Vector2d bc = Q_C * ac;
		Eigen::Vector2d bd = Q_D * ad;
		return sqrt(ac(0)*bc(0) + ac(1)*bc(1)) + a_C.dot(ac) 
			- sqrt(ad(0)*bd(0) + ad(1)*bd(1)) + a_D.dot(ad);
	}

	double calculate(Eigen::Vector2d p, Eigen::Vector2d& z)
	{
		int count = 0;
		double L = 5.0;
		double alpha = 1.0/L;

		bool stop = false;

		for(int k=0; k<maxit; ++k)
		{
			// cout << "k=" << k << endl;
			for(int i=0; i<dim; ++i)
			{
				double dp = - alpha * dFi(p,z,i);
				p(i) += dp;

					
				

				if(abs(dp) > eps)
				{
					count = 0;
					if(k==maxit-1)
					{
						k=0;
						alpha /=2;
						L *= 2;
					}
				}
				else
				{
					count++;
				}

				if(count == dim)
				{
					stop = true;
					break;
				}

			}
			if(stop) break;
		}
		return calculate_F(p,z);
	}

	double calculate_F(Eigen::Vector2d& p, Eigen::Vector2d& z)
	{
		double Jpart = 0.5 + 0.5 * (p(0)*p(0) + 4.0/25.0 * p(1)*p(1));

		double Hpart = 0;
		for(unsigned int j=0; j< Nt; ++j)
		{
			Hpart += H(p,j);
		}
		return Jpart + Hpart * ds - z.dot(p);
	}
};

int main()
{
	int N = 60;

	

	Eigen::Vector2d p; p << 0.2,0.2;

	double t = clock();

	for(int ind = 1; ind <=14; ++ind)
	{
		double small_t = ind * 0.05;
		ofstream out;
		string filename = to_string(ind) + "file3.dat";
		out.open(filename);

		CD c(small_t,0.7);

		for(int i=0; i<N+1; ++i)
		{
			for(int j=0; j<N+1; ++j)
			{
				double ii = -3 + 0.1*i;
				double jj = -3 + 0.1*j;

				Eigen::Vector2d z; z << ii,jj;

				out << ii << "," << jj << "," << c.calculate(p,z) << endl;
				
			}
		}
		out.close();
	}
	

	// cout << (clock() - t)/CLOCKS_PER_SEC << endl;

	

	
}