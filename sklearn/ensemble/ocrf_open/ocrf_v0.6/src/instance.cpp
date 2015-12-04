
#include "../include/instance.h"

/*
 * Constructor
 * param:
 * 		ii : an id integer
 * 		w : a weight
 * 		vals : a vector of data. The instance is allowed to be empty just after its instanciation, so the vals vector can be NULL.
 */
		Instance::Instance(u_int ii, vector<double>* vals)
{
	id = ii;
	originalId=ii;
	if(vals != NULL)
	vect = (*vals);
}

		Instance::~Instance(){
		}
/*
 * Copy constructor
 */
		Instance::Instance(const Instance &inst)
{

    id = (const_cast<Instance*>(&inst))->id;
    originalId=(const_cast<Instance*>(&inst))->getOriginalId();

   for(vector<double>::iterator it=(const_cast<Instance*>(&inst))->vect.begin();it!=(const_cast<Instance*>(&inst))->vect.end();it++)
        vect.push_back(*it);
}

/*
 * The function to add a value in the instance.
 * param :
 * 		v : the double value to be added
 */
void 	Instance::add(double v) { vect.push_back(v); }

vector<double> * Instance::getVect() {return &vect;}
vector<double> Instance::getVectSimple() {return vect;}

/*
 * accessor to a pointed double value
 * param :
 * 		attInd : attribute index that point to a double value in the vals vector
 */
double 	Instance::at(u_int attInd)
{
	if(attInd<0 || attInd>=vect.size()) return 0.0;
	return vect[attInd];
}

/*
 * accessor to the identifiant of the instance.
 */
u_int 	Instance::getId() { return id; }


string	Instance::toString()
{
	string out = "(";
	out += Utils::to_string((int) id);
	out += ") ";

	for(u_int i=0;i<vect.size();i++)
	{
		out += Utils::to_string(vect[i]);
		out += " ";
	}

	return out;
}

u_int   Instance::getClass() {
    return vect[vect.size()-1];
}

void Instance::modClass(u_int newClass){
    vect[vect.size()-1] = newClass;
}



