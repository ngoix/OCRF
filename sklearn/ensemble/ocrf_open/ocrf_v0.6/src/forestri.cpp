
#include "../include/forestri.h"

ForestRI::ForestRI(int lTree, int kFeat, int rBagg)
:RFInduc(lTree,false),k(kFeat),r(rBagg)
{
	bagger = new Bagging(r);
	rndtreeInduc = new RndTree(k,false);
}

ForestRI::~ForestRI(){}

DForest * ForestRI::growForest(DataHandler * set)
{

	if(disp)
	{
		Utils::print("ForestRI (");
		Utils::print(Utils::to_string(L));
		Utils::print(",");
		Utils::print(Utils::to_string(k));
		Utils::print(",");
		Utils::print(Utils::to_string(r));
		Utils::print(") ... ");
	}
	double _time = ((double) clock());
	_time = _time / CLOCKS_PER_SEC;
	double averageTime = 0.0;

	trainSet = set;

	vector<DataHandler *> bootstraps;
	for(int i=0;i<L;i++)
	{
		DataHandler * handl = bagger->generateBootstrap(trainSet);
		bootstraps.push_back(handl);
	}



	DForest * res = new DForest(trainSet);
	for(int i=0;i<L;i++)
	{
cerr << ".";
		// Launch the induction of the current Decision Tree
		DTree * tree = rndtreeInduc->growTree(bootstraps[i]);
		averageTime += tree->stat_getTimeTrain();

		F_DTree * ftree = new F_DTree(tree);
		res->addTree(ftree);

		delete tree;
		delete bootstraps[i];

	}
cout << "\n";
	// Update forest statistics
	averageTime = averageTime/L;
	_time = (((double) clock())/CLOCKS_PER_SEC) - _time;
	res->stat_setTimeTrain(_time);
	res->stat_setMeanDTTimeTrain(averageTime);

	string tmp = Utils::to_string(_time);
	tmp += " secondes";
	if(disp) Utils::print(tmp);

	delete bagger;
	delete rndtreeInduc;

	return res;

}


/* *******************************************************
 * Accessor to change the value of the nbFeat parameter
 * No validity check of the k value is done here while it is done at the growTree algorithm beginning
 */
void		ForestRI::setNbFeatParam(int nbFeat){ k = nbFeat; rndtreeInduc->setNbFeatParam(k); }

