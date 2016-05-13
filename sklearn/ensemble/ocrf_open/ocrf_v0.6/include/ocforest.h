#ifndef OCFOREST_H_
#define OCFOREST_H_

#include "../include/rndtree.h"
#include "../include/bagging.h"
#include "../include/rfinduc.h"
#include "../include/outliergenerator.h"

struct params_forest {
	int param_nbTrees;
	int param_KRFS;
	int param_KRSM;
	int param_percentBagging;/**< size of the bootstrap sample (in percentage of the original training set size) */
	bool param_bagging;
	bool param_rsm;
	bool param_logTrees;/**< Used for visualization of target and generated outlier data distribution*/
	bool param_studyNodes;/**< Log of nodes configuration for 2D/3D visualization; valid for dimension 2 and 3 only;higher dimensions are ignored.*/

	/**
	 * Default values to be extracted from default configuration file///TODO:default configuration file
	 */
	params_forest(){}

};

struct params_oneclass {
	int param_nbOutlierData;/**<$nbOutlier=\beta\cdot nbTarget$*/
	double param_alpha;/**< $\Omega_{outlier}=param_alphaZone\cdot\Omega_{target}$*/
	bool param_rejectOutOfBounds;/**<true to reject instantly outlier data that are outside the domain of the target data (false is default); less flexible out-of-class generalization;*/
	bool param_optimize;/**<true:biased roulette-wheel method;false:uniform generation;*/
	bool param_generateInOriginalDataSet; ///amontGen;
	bool param_method;/**<learning method for OCRF;
	0 (default value) for outlier generation in Random Subspaces (RSM) projected bootstrap sets (bagging+rsm);
	1 for outlier generation in the original set, before the induction of the forest; TODO: redundancy with generateInOriginalDataSet, bagging, rsm*/
	std::vector<std::vector<int> > param_histogramOutlierData;/**< 1D computed distribution of outlier data obtained from prior information extracted from the original learning set*/
	std::vector<std::vector<double> > param_boundsInterTargetData;/**< Contains bounds information obtained from target distribution*/
};

/********************************************************************
 *
 *   Name: class _OCRFStats
 *
 *   Description: private class for internal use only
 *
 *********************************************************************/
class _OCRFStats {
	friend class OCForest;

private:
	int nbNodesTotal;
	int nbLeavesTotal;
	int nbLeavesTargetTotal;
	int nbLeavesOutlierTotal;
	int nbLevelsTotal;

	double varNodesTotal;
	double varLeavesTotal;
	double varLeavesTargetTotal;
	double varLeavesOutlierTotal;
	double varLevelsTotal;
	double timeTrain;
	const char * loadingBar;

	_OCRFStats() {
		nbNodesTotal = 0;
		nbLevelsTotal = 0;
		nbLeavesTotal = 0;
		nbLeavesTargetTotal = 0;
		nbLeavesOutlierTotal = 0;

		varNodesTotal = 0;
		varLeavesTotal = 0;
		varLeavesTargetTotal = 0;
		varLeavesOutlierTotal = 0;
		varLevelsTotal = 0;
		timeTrain = 0.0;
		loadingBar = "|/-\\";
	}
};

class OCForest: public RFInduc {
private:

	int Ltree;/**< Number of trees in the forest*/
	int r;/**< size of the bootstrap sample (in percentage of the original training set size) */
	int nbRSM;/**< Dimension of the randomly selected subspaces*/

	bool rejectOutOfBounds;
	bool optimize_gen;/**< Optimize the generation process : uniform, biased roulette-wheel, derivative*/
	bool log_trees;
	bool studyNodes;/**< Log of nodes configuration for 2D/3D visualization; valid for dimension 2 and 3 only;higher dimensions are ignored.*/

	DataHandler* handltempAmont;/**< DataHandler helper in case of generation before the induction of the forest */

	vector<vector<int> > histoOutlier;
	vector<vector<double> > boundsInterTarget;

	OutlierGenerator * generator;
	u_int ** listsubspace;/**< List of sub-spaces dimensions indexes for each tree in case of RSM */
	bool generateInOriginalDataSet;
	bool rsm;
	bool bagging;
	int nbOutlier;
	double alpha_domain;
	int K_RFS;

	string database_name;
	_OCRFStats OCRFStats;

public:

	OCForest(params_forest pForest, params_oneclass pOneclass, string dbName);
	OCForest(int lTree, int nbFeat, int rBagg, int nbOutliers,
			double taillezone, int nbSubspace, bool amontGen, bool bag,
			bool randsm, bool rejet_bornes_temp, bool optimize_gen_temp,
			string dbName, const vector<vector<int> >& histoOutlierTemp,
			const vector<vector<double> >& boundsInterTargetTemp,
			bool log_trees_temp);
	virtual ~OCForest();

	DForest * growForest(DataHandler * set);
	void displayListSubspace();u_int ** getlistsubspace();
	void setNbFeatParam(int k);

	int getNbNodes();
	int getNbLeaves();
	int getNbLeavesTarget();
	int getNbLeavesOutlier();
	int getNbLevels();

	double getVarNbNodes();
	double getVarNbLeaves();
	double getVarNbLeavesTarget();
	double getVarNbLeavesOutlier();
	double getVarNbLevels();

};

#endif /*OCFOREST_H_*/

