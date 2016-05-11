#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_

#include "../include/utils.h"
#include "../include/dataset.h"

/********************************************************************
*
*   Name:           class DataHandler
*
*   Description:  	Structure to handle data.
* 		represents a subset of the dataset "set".
* 		"subset" is thus a set of pointer to dataset's instances.
*
*********************************************************************/
class DataHandler
{




	private:
		string		filename;

		DataSet *	set;
		v_inst		subset;

		double * 	distrib;
		u_int 		classInd;

		map<int,double>	weights;

int iter;

	public :
				DataHandler(DataSet * dataset, u_int cl, bool full = true);
		virtual		~DataHandler();


double** computeMinMax();
double** computeMinMaxOutlier();
double** computeMinMaxClass(u_int cl);

		void		addInstance(Instance * inst, double w = 1.0);

		u_int 		size();
		double		w_size();
		u_int 		dim();
		u_int getSize();
		bool		empty();

		v_inst_it	begin();
		v_inst_it	end();

		void        changeClass(u_int indInst, u_int newClass);

		DataSet * 	getDataSet();
		u_int		getNbClass();
		u_int		getNbAttributes();
		Instance *	getInstance(u_int instInd);
		Attribute * 	getAttribute(u_int attInd);
		double		getDistrib(u_int clasInd);
		double * 	getDistrib();
		u_int		getClassInd();
		u_int		getClass(Instance * inst);
		double		getWeight(Instance * inst);
		double      getWeight(u_int Id);
		string 		getFileName();

		void		setClassInd(u_int cl);
		void		setFileName(string file);

		string 		toString();
		void        afficheBase();

		void setData(DataSet* temp){set=temp;}
		//void normalize();
		vector<double>	normalize(double **minmaxval,vector<double>moyenne,bool calcMoy);
};

#endif /*DATAHANDLER_H_*/
