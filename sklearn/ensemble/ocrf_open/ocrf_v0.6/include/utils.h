#ifndef UTILS_H_
#define UTILS_H_

#include "../include/global.h"

#define debug(str) std::cout<<"DEBUG:"<<__LINE__<<":"<<__FILE__<<":"<<str<<std::endl
#define debugv(str,value) std::cout<<"DEBUG:"<<__LINE__<<":"<<__FILE__<<":"<<str<<":"<<value<<std::endl

class Utils
{
	static u_int x,y,z,c; /* Seed variables */

	public :
		static char		_c;
		static ofstream		flog;

		static u_int*		sort(vector<double> & array);
		static u_int 		partition(vector<double> & array, u_int index[], u_int l, u_int r);
		static void 		quickSort(vector<double> & array, u_int index[], u_int left, u_int right);
		static u_int		randInt(int n);
		static void 		tokenize(const string& str, vector<string>& tokens, const string& delimiters);
		static double		from_string(const string & s);

		template<typename T> std::string to_string(T var);
		static string 		to_string(int var);
		static string 		to_string(double var);
		static string 		to_string(u_int var);


		static void 		badException(void);
		static void		print(string text,bool end=true);
		static double 		lnFunc(double val);
		static double       abs(double val);
		static double       from_scientific(string * s);
		static u_int*       samplingWithoutReplacement(u_int nb,u_int max);
		static bool         contains(const u_int * tab,u_int val,u_int taille);

static double randgauss(double mean,double var);
		static double randBoule(double rayon,int dim);

//static template<typename T> vector<T> getMatCol(vector<vector<T> >& mat,int ind);

};

template<typename T> std::string Utils::to_string(T var) {
	std::stringstream ss;
	ss << var;
	return ss.str();
}

template<typename T> std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {

	for(auto a:v) out<<a<<" ";
	return out;
	/*
	int i;
	for (i = 0; i < (int) v.size(); i++) {
		if (i < (int) v.size() - 1)
			out << v.at(i) << "\n";
		else
			out << v.at(i) << '\n';
	}

	return out;
	*/
}


template<typename T> int fromString(const string& src, T& dest) {

	stringstream ist;
	ist<<src;
	ist >> dest;

	//return (ist >> dest) != 0;
	return ist.gcount();
}




#endif /*UTILS_H_*/
