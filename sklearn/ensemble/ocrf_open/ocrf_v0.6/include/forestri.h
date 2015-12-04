#ifndef FORESTRI_H_
#define FORESTRI_H_

#include "../include/rndtree.h"
#include "../include/bagging.h"
#include "../include/rfinduc.h"

class ForestRI: public RFInduc {
private:
	int k;	//number of features randomly selected in RFS procedure
	int r;//size of the bootstrap sample (in percentage of the original training set size)

	Bagging * bagger;
	RndTree * rndtreeInduc;

public:
	ForestRI(int lTree, int kFeat, int rBagg);
	virtual ~ForestRI();

	DForest * growForest(DataHandler * set);

	void setNbFeatParam(int k);
};

#endif /*FORESTRI_H_*/
