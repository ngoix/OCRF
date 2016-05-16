
#include "../include/dataset.h"

/* *************************************************************
 * Constructor
 */
DataSet::DataSet()
{
	idsInst = 0;
	idsAtt = 0;
	nbRef = 0;
}

DataSet::DataSet(DataSet * d) {
    attributes = d->attributes;
    data = d->data;

    idsInst = d->idsInst;
	idsAtt = d->idsAtt;
	nbRef = d->nbRef;
}

/* *************************************************************
 * Destructor
 */
DataSet::~DataSet()
{
    attributes.clear();
    data.clear();//call destructor of each element i.e. Instance::~Instance()
}

int DataSet::getNBAttribute() {
    return attributes.size();
}


/* *************************************************************
 * Add an attribute to the description vector of attribute objects
 * param :
 * 		n : the name of the attribute to be added
 * 		t : the type of the attribute to be added
 * 		mod : a vector of moadalities for nominal attributes to be added
 *
 * return the current number of attributes (minus the class attribute)
 */
int 		DataSet::addAttribute(string n, attType t, vector<string> * mod)
{
	Attribute* att= new Attribute(idsAtt,n,t,mod);
	idsAtt++;
	attributes.push_back(*att);
	delete att;
	return attributes.size()-1;
}

int		DataSet::addAttribute(Attribute & a)
{
	Attribute* att=new Attribute(a.id,a.name,a.type,&(a.modal));
	attributes.push_back(*att);

	delete att;
	return attributes.size()-1;
}

/* *************************************************************
 * Add an instance to the dataset
 * param :
 * 		w : th weight of the instance to be added
 * 		vals : the vector of double valued elements
 *
 * return the current number of attributes (minus the class attribute)
 */
int		DataSet::addInstance(vector<double> * vals, int orig_id)
{

	Instance* inst= new Instance(idsInst,vals);

	if(orig_id!=-1) inst->setOriginalId(orig_id);

	idsInst++;

	data.push_back(*inst);

	delete inst;
	return data.size()-1;
}




/* *************************************************************
 * return the size of the dataset (i.e. the number of instances)
 */
u_int		DataSet::size(){ return data.size(); }

/* *************************************************************
 * return the dimension of the dataset (i.e. the number of attributes)
 */
u_int		DataSet::dim(){ return attributes.size(); }

/* *************************************************************
 * return the instance of the dataset at the given index
 */
Instance *	DataSet::getInstance(u_int instInd){ return &data[instInd]; }

/* *************************************************************
 * return the attribute of the description vector at the given index
 */
Attribute *	DataSet::getAttribute(u_int attInd){ return &attributes.at(attInd); }

/* *************************************************************
 * return the value of the instance at the given index, concerning the attribute at the given index
 * param :
 * 		ii : index of the instance
 * 		attInd : index of the attribute
 */
double 		DataSet::getValue(u_int instInd, u_int attInd){ return data[instInd].at(attInd); }

/* *************************************************************
 * return the number of modalities for the attribute at the given index
 */
u_int		DataSet::getNbModal(u_int attInd){ return attributes[attInd].getNbModal(); }



void DataSet::affbase() {

    for (int i=0;i<(int)data.size();i++) {
        cout << data[i].toString()<<"\n";
    }
}
