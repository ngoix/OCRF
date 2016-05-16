
#include "../include/datahandler.h"

/*********************************************************
 * @brief Constructor.
 * @param
 * 		dataset : the dataset to be handled
 *		cl : an index that gives the class position in the attribute vector
 * 		full : a boolean that indicate if the subset has to be fully handled. Set to true by default
 */
DataHandler::DataHandler(DataSet * dataset, u_int cl, bool full){

	filename = "";
	set = dataset;
	set->nbRef++;

    classInd = cl;
	u_int nbClass = getNbClass();
	distrib = new double[nbClass];

iter=0;

	for(u_int i=0;i<nbClass;i++) distrib[i] = 0.0;
	if(full)
	{

		for(u_int i=0;i<set->size();i++)
		{

Instance*  inst = set->getInstance(i);

			subset.push_back(inst);

int indC=getClass(inst);
int idInst=(inst)->getId();
			weights.insert(pair<int,double>(idInst,1));
			distrib[indC]++;

		}
	}
}

/* ********************************************************
 * Destructor
 */
DataHandler::~DataHandler()
{

		delete[] distrib;
	set->nbRef--;

	if(set->nbRef== 0) {
	    delete set;
	}
}


vector<double>	DataHandler::normalize(double **minmax,vector<double>moyenne,bool calcMoy){

	int nbAttrib=set->getNBAttribute()-1;
	vector<Instance> dataTemp=set->getData();
	vector<Instance> dataTempNorm=set->getData();

	vector<double> moy((int)nbAttrib,0);
	vector<double> moyTemp((int)nbAttrib,0);
	vector<double> stdDev((int)nbAttrib,0);
	double valMax=0;
	double val=0;

	int nbData=dataTemp.size();

ofstream fichier("./resultats/temp.arff",ios::out);
ofstream fichierNorm("./resultats/tempNorm.arff",ios::out);

if(!calcMoy){
	moy=moyenne;
}
else{
	for(int i=0;i<nbData;i++){
		moyTemp=dataTemp[i].getVectSimple();

		for(int j=0;j<nbAttrib;j++){
			moy[j]+=moyTemp[j]/nbData;

			val+=moyTemp[j]*moyTemp[j];

			stdDev[j]+=val/nbData;

			fichier<<moyTemp[j]<<",";
		}
		fichier<<"\n";


		if(val>valMax) valMax=val;
		val=0;
	}
valMax=sqrt(valMax);

		for(int j=0;j<nbAttrib;j++){
			stdDev[j]-=moy[j]*moy[j];
			stdDev[j]=sqrt(stdDev[j]);
		}

}

	for(int i=0;i<nbData;i++){
		moyTemp=dataTemp[i].getVectSimple();

		for(int j=0;j<nbAttrib;j++){
			moyTemp[j]-=moy[j];
			moyTemp[j]/=minmax[1][j];//valMax;//stdDev[j];//valMax;

			fichierNorm<<moyTemp[j]<<",";
		}
		fichierNorm<<"\n";
		dataTempNorm[i].setVect(moyTemp);
	}

set->setData(dataTempNorm);
//moy.clear();
moyTemp.clear();
stdDev.clear();

dataTemp.clear();
dataTempNorm.clear();

return moy;
}


double** DataHandler::computeMinMaxOutlier(){


	u_int nbCarac = set->dim()-1;//getNbAttributes();

	    double **minmaxvaloutlier = new double*[2];

    minmaxvaloutlier[0] = new double [nbCarac];//min max pour chacune des dimensions
    minmaxvaloutlier[1] = new double [nbCarac];

//cerr<<"init minmax 3"<<endl;
//cerr << "1";

    for (u_int c=0;c<nbCarac;c++){
    for (u_int j=0;j<set->size();j++){

    	if(set->getInstance(j)->getClass()==OUTLIER){
        minmaxvaloutlier[0][c] = (double)set->getInstance(j)->at(c);
        minmaxvaloutlier[1][c] = (double)set->getInstance(j)->at(c);
        break;
    	}

    }
    }


    for (u_int i=0;i<set->size();i++) {
    	if(set->getInstance(i)->getClass()==OUTLIER){

    	Instance* inst=set->getInstance(i);

        for (u_int c =0;c<nbCarac;c++) {

            if( (double)inst->at(c) < minmaxvaloutlier[0][c] ) {
                minmaxvaloutlier[0][c] = (double)inst->at(c);
            }
            if( (double)inst->at(c) > minmaxvaloutlier[1][c] ) {
                minmaxvaloutlier[1][c] = (double)inst->at(c);
            }
        }
    }

    }

	return minmaxvaloutlier;

}

double** DataHandler::computeMinMax(){

	u_int nbCarac = set->dim()-1;//trainSet->getNbAttributes();
	    double **minmaxval = new double*[2];

    minmaxval[0] = new double [nbCarac];//min max pour chacune des dimensions
    minmaxval[1] = new double [nbCarac];

    for (u_int c=0;c<nbCarac;c++){
    	if(c==getClassInd()) continue;
        minmaxval[0][c] = (double)set->getInstance(0)->at(c);
        minmaxval[1][c] = (double)set->getInstance(0)->at(c);
    }

    for (u_int i=0;i<set->size();i++) {

    	Instance* inst=set->getInstance(i);

        for (u_int c =0;c<nbCarac;c++) {
if(c==getClassInd()) continue;

            if( (double)inst->at(c) < minmaxval[0][c] ) {
                minmaxval[0][c] = (double)inst->at(c);
            }
            else if( (double)inst->at(c) >= minmaxval[1][c] ) {
                minmaxval[1][c] = (double)inst->at(c);
            }

        }
    }

	return minmaxval;

}

double** DataHandler::computeMinMaxClass(u_int cl){
	u_int nbCarac = set->dim()-1;//trainSet->getNbAttributes();
	    double **minmaxvalclass = new double*[2];

    minmaxvalclass[0] = new double [nbCarac];//min max pour chacune des dimensions
    minmaxvalclass[1] = new double [nbCarac];

    for (u_int c=0;c<nbCarac;c++){
if(c==getClassInd()) continue;
    for (u_int j=0;j<set->size();j++){

    	if(set->getInstance(j)->getClass()==cl){
        minmaxvalclass[0][c] = (double)set->getInstance(j)->at(c);
        minmaxvalclass[1][c] = (double)set->getInstance(j)->at(c);
        break;
    	}

    }
    }

//remplissage minmax
    for (u_int i=0;i<set->size();i++) {

    	if(set->getInstance(i)->getClass()==cl){

    	Instance* inst=set->getInstance(i);

        for (u_int c =0;c<nbCarac;c++) {
if(c==getClassInd()) continue;
            if( (double)inst->at(c) < minmaxvalclass[0][c] ) {
                minmaxvalclass[0][c] = (double)inst->at(c);
            }
            if( (double)inst->at(c) > minmaxvalclass[1][c] ) {
                minmaxvalclass[1][c] = (double)inst->at(c);
            }
        }
    }

    }

	return minmaxvalclass;

}


/* ********************************************************
 * Adds an instance of the dataset as a member of the subset handled. If the instance has already been added to the subset, its weight is just increase by the given weight value.
 * param :
 * 		inst : the instance to be added
 *		w : the weight of the given instance
 */
void		DataHandler::addInstance(Instance * inst, double w){
	if(weights.find(inst->getId()) != weights.end()){
		weights[inst->getId()] += w;
	}
	else
	{
		subset.push_back(inst);
		weights.insert(pair<int,double>(inst->getId(),w));
	}
	distrib[getClass(inst)] += w;
}

/* ********************************************************
 * Gives the number of instance in the subset
 */
u_int 		DataHandler::size(){ return subset.size(); }

/* ********************************************************
 * Gives the size of the subset, according to instance weights
 */
double		DataHandler::w_size()
{
	double res= 0.0;
	for(v_inst_it it=subset.begin();it!=subset.end();it++)
		res += weights[(*it)->getId()];
	return res;
}

/* ********************************************************
 * Gives the dimensionality of the feature space.
 */
u_int		DataHandler::dim(){ return set->dim(); }

u_int		DataHandler::getSize(){ return set->size(); }

/* ********************************************************
 * Indicates weather or not the subset is empty
 */
bool 		DataHandler::empty() { return subset.empty(); }

/* ********************************************************
 * Gives a begin iterator to the subset vector of instance pointers.
 */
v_inst_it	DataHandler::begin() { return subset.begin(); }

/* ********************************************************
 * Gives an end iterator to the subset vector of instance pointers.
 */
v_inst_it	DataHandler::end() { return subset.end(); }

/* ********************************************************
 * Gives a pointer to the dataset
 */
DataSet * 	DataHandler::getDataSet(){ return set; }

/* ********************************************************
 * Gives the number of class modalities
 */
u_int		DataHandler::getNbClass() { return set->getNbModal(classInd); }

/* ********************************************************
 * Gives the number of attributes
 */
u_int		DataHandler::getNbAttributes() { return (dim()-1); }

/* ********************************************************
 * Gives the class value of a given instances of the subset
 * param :
 * 		inst : a pointer to an instance. It is supposed to be in the dataset.
 */
u_int 		DataHandler::getClass(Instance * inst) { return (u_int) (inst->at(classInd)); }

/* ********************************************************
 * Gives a pointer to an instance from a given index
 * param :
 * 		ii : an index of an instance in the subset. Indicates the position of the instance to be returned in the subset container.
 */
Instance *	DataHandler::getInstance(u_int instInd)
{
	if(instInd<0 || instInd>=subset.size()) return NULL;
	return subset[instInd];
}

/* ********************************************************
 * Gives a pointer to an attribute object
 * param :
 * 		ii : an index of an attribute in the dataset attribute vector
 */
Attribute * 	DataHandler::getAttribute(u_int attInd)
{
	if(attInd<0 || attInd>=dim()) return NULL;
	return set->getAttribute(attInd);
}

/* ********************************************************
 * Gives the number of instances in the subset that belong to the class corresponding to the given index
 * param :
 * 		ii : an index of a class value
 */
double		DataHandler::getDistrib(u_int clasInd)
{
	if(clasInd<0 || clasInd>=getNbClass()) return 0.0;
	return distrib[clasInd];
}

/* ********************************************************
 * Gives the distribution array, containing all the class distributions in the subset
 */
double * 	DataHandler::getDistrib(){ return distrib; }

/* ********************************************************
 * Gives the class index in the attribute vector
 */
u_int		DataHandler::getClassInd(){ return classInd; }

/* ********************************************************
 * Gives the weight of the given instance.
 *	param :
 * 		inst : a pointer to the instance of which the weight is returned.
 */
double		DataHandler::getWeight(Instance * inst)
{
	int instId = inst->getId();
	if(weights.find(instId) != weights.end())
		return weights[instId];
	else
		return 1.0;
}

double      DataHandler::getWeight(u_int Id) {
    if(weights.find(Id) != weights.end())
		return weights[Id];
	else
		return 1.0;
}

/* ********************************************************
 * Gives the name of the the filename used to load the dataset
 */
string		DataHandler::getFileName(){ return filename; }

/* ********************************************************
 * Set the class position in the attribute vector of the dataset.
 *	param :
 *		cl : the position of the class in the attribute vector.
 */
void		DataHandler::setClassInd(u_int cl)
{
	if(set->getAttribute(cl)->is_nominal()) classInd = cl;
	//else cl = 0;//classInd remain unchanged
}

/* ********************************************************
 * Set the name of the file used to load the dataset.
 *	param :
 *		file : the name of the file
 */
void 		DataHandler::setFileName(string file){ filename = file; }


/* ********************************************************
 * Gives a "string" description of the subset.
 */
string		DataHandler::toString()
{

	string out = "";
	out += "\tTaille de la base : ";
	out += Utils::to_string((int)w_size());

	out += "\n\tNombre d'attributs : ";
	out += Utils::to_string((int)dim()-1);

	out += "\n\tNombre de classes : ";


	out += Utils::to_string((int)getNbClass());
	out += "\n\tDistribution des classes : \n";

	for(u_int i=0;i<getNbClass();i++)
	{
		out += Utils::to_string(distrib[i]);
		out += "\t";
	}

	return out;
}

void DataHandler::changeClass(u_int indInst, u_int newClass) {
    (set->getInstance(indInst))->modClass(newClass);
}

void DataHandler::afficheBase() {

    for (int i=0;i<(int)size();i++) {
        cout << getInstance(i)->toString()<<" poids : "<<getWeight(getInstance(i))<<"\n";
    }

}
