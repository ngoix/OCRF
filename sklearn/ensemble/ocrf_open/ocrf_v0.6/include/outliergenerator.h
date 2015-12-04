#ifndef OUTLIERGENERATOR_H_
#define OUTLIERGENERATOR_H_

#include "../include/datahandler.h"
#include "../include/dataset.h"
#include "../include/instance.h"

class OutlierGenerator {

private:

	bool amontGen; //< The outlier data are generated before the induction of the forest
	bool optimize_gen; //< Optimization option for outlier generation (biased roulette-wheel, derivative ...etc instead of uniform)
	bool bag;//< outlier generation in bootstrap
	bool rsm;//< outlier generation in projected subspaces
	double bagsize;
	int nbsubspace;
	int * vectorsubspace;//TODO:naming
	int nbOutlier;//beta*nbTarget
	double alpha_mod;//alpha-1

	vector<Instance> data;
	vector<int> listRSM;
//vector<vector<double> > histoTarget;
	vector<vector<int> > histoOutlier;
	vector<vector<double> > boundsInterTarget;

//DataSet * newData;

public:

	OutlierGenerator();
	OutlierGenerator(bool amontGenTemp, bool baggy, bool randsm, double bsize,
			int nbsub, int nbO, double tz, bool optimize_gen_temp);
	virtual ~OutlierGenerator();

	DataHandler * generateProcess(DataHandler * base, u_int **t,
			const vector<vector<int> >& histoOutlierTemp,
			const vector<vector<double> >& boundsInterTargetTemp);
	DataHandler * bagg(DataHandler *base);
	DataHandler * randomsubspacemethod(DataHandler * base, u_int **t);
	DataHandler * generateOutlierData(DataHandler * base);
	double parzenKernelValue(double x, int d);
	Instance * optimizeGeneration(u_int placeCarac, u_int id, u_int nbCarac,
			double taillezone, double ** minmaxval);

	static DataHandler ** genere10folds(DataHandler * data);
	static DataSet * transformeBaseOC(DataHandler * base, u_int target);
	static DataHandler ** transformeBaseOC(DataHandler * base, u_int target,
			u_int pourcentApp);
	static void genereArffOneClass(DataHandler * base, int indClassTarget,
			unsigned long nbOutlier, string nomficApp, string nomficTest,
			double taillezone, u_int pourcentApp);
	Instance * genereInstance(u_int id, u_int nbCarac, double ** minmaxvalue,
			double taillezone, u_int placeCarac);
	Instance * genereInstanceBoule(u_int id, u_int nbCarac, double ** minmaxval,
			double taillezone, u_int placeCarac);

	DataHandler * generateOutlierData(DataHandler * base,
			const vector<vector<int> >& histoOutlierTemp,
			const vector<vector<double> >& boundsInterTargetTemp);

	void rouletteWheel(DataHandler* handlerBaseOrig,
			vector<vector<int> >& histoOutlier,
			vector<vector<double> >& boundsInterTarget, double taillezone);

};

#endif
