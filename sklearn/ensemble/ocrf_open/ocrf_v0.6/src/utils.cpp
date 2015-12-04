
#include "../include/utils.h"


char Utils::_c = 'a';
ofstream Utils::flog(DATA_LOG);
// init values for the Random Number Generator
u_int Utils::x = 123456789, Utils::y = 362436000, Utils::z = 521288629, Utils::c = 7654321;

/****************************************************************
 * Sort a vector of double values with the quicksort algorithm
 */
u_int*	Utils::sort(vector<double> & array)
{
	u_int * index = new u_int[array.size()];
	for(u_int i=0;i<array.size();i++) index[i] = i;
	quickSort(array,index,0,array.size()-1);
	return index;
}


//template<typename T> vector<T> Utils::getMatCol(vector<vector<T> >& mat,int ind){
//	vector<T> temp;
//
//	for(int i=0;i<mat.size();i++){
//		temp.push_back(mat.at(i).at(ind));
//	}
//
//	return temp;
//}

u_int 	Utils::partition(vector<double> & array, u_int index[], u_int l, u_int r)
{
	double pivot = array[index[(l+r)/2]];
	u_int help;

	while(l<r)
	{
		while((array[index[l]] < pivot) && (l<r)) l++;
		while((array[index[r]] > pivot) && (l<r)) r--;

		if(l<r)
		{
			help = index[l];
			index[l] = index[r];
			index[r] = help;
			l++;
			r--;
		}
	}
	if((l==r) && (array[index[r]] > pivot)) r--;

	return r;
}

void 	Utils::quickSort(vector<double> & array, u_int index[], u_int left, u_int right)
{
	if(left < right)
	{
		int middle = partition(array,index,left,right);
		quickSort(array,index,left,middle);
		quickSort(array,index,middle+1,right);
	}
}

void 	Utils::tokenize(const string& str, vector<string>& tokens, const string& delimiters)
{
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

/*
 * A simple function for sampling random integer between 0 and n
 * param :
 * 		n : the sup bound.
 *
 * return a random integer >= 0 and <n
 */
u_int 	Utils::randInt(int n)
{
	/*
	unsigned long long t, a = 698769069ULL;
	x = 69069*x+12345;
	y ^= (y<<13); y ^= (y>>17); y ^= (y<<5);
	t = a*z+c; c = (t>>32);
	return (x+y+(z=t))%n;
	*/

	//version srand
double ranval = (double)(rand()%(n));///10;
return ranval;

}

double 	Utils::from_string(const string & s)
{
    double result;
    istringstream stream(s);
    stream >> result;
    return result;
}

double Utils::from_scientific(string * s) {
    double result;
    vector<string> tokens;
    Utils::tokenize(*s, tokens, "e");
    result = from_string(tokens[0]);
    int puiss = (int)(from_string(tokens[1]));
    result = result * pow(10,puiss);
    return result;
}




string 	Utils::to_string(double var)
{
    ostringstream tmp;
    tmp << var;
    return tmp.str();
}

string 	Utils::to_string(int var)
{
    ostringstream tmp;
    tmp << var;
    return tmp.str();
}

string 	Utils::to_string(u_int var)
{
    ostringstream tmp;
    tmp << var;
    return tmp.str();
}


void 	Utils::badException(void)
{
    cout << "Exception inconnue reçue !" << endl;
    exit(-1);
}

void	Utils::print(string txt,bool end)
{
	if(!end) cout << "\r";
	cout << txt;
	cout.flush();

	if(!end) flog << "\r";
	flog << txt;
	flog.flush();

}

double	Utils::lnFunc(double val)
{
	return (val * log(val));
}

double Utils::abs(double val) {
    if (val > 0) {
        return val;
    }
    else {
        return -val;
    }
}

bool Utils::contains(const u_int * tab,u_int val,u_int taille) {
    bool cont=false;
    //u_int * tabTemp=tab;


    for (u_int i=0;i<taille;i++) {

        if (tab[i] == val) {

            cont = true;
            break;
        }

    }
    return cont;
}


u_int* Utils::samplingWithoutReplacement(u_int nb,u_int max) {
    bool found;
    u_int temp;
    u_int * tab = new u_int[nb];
    for (u_int i=0;i<nb;i++) {
        tab[i] = -1;//
    }
    for (u_int i=0;i<nb;i++) {
        found = false;
        while(!found) {
            temp = randInt(max);
            if (!contains(tab,temp,nb)) {
                found = true;
            }
        }
        tab[i] = temp;
    }
    return tab;
}


double Utils::randgauss(double mean,double var) {
        float x1, x2, w, y1;
        double ret;

         do {
                 x1 = 2.0 * ((double)randInt(10000))/10000.0 - 1.0;
                 x2 = 2.0 * ((double)randInt(10000))/10000.0 - 1.0;
                 w = x1 * x1 + x2 * x2;
         } while ( w >= 1.0 );

         w = sqrt( (-2.0 * log( w ) ) / w );
         y1 = x1 * w;
         ret = y1*var+mean;
         return ret;
}

double Utils::randBoule(double x,int dim) {

//double chisquaredistribution(const double v, const double x);
//double val=alglib::chisquaredistribution((double)dim, x);

//cout<<val<<endl;

	throw "Deprecated !";

return -1;
}




