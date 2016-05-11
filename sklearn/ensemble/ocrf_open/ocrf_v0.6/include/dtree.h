#ifndef DTREE_H_
#define DTREE_H_

#include "../include/datahandler.h"
#include "../include/node.h"
#include "../include/result.h"


/********************************************************************
*
*   Name: class _DTStats
*
*   Description: private class for internal use only
*
*********************************************************************/
class _DTStats
{
	friend class DTree;




	private:
		bool 	bNodes;
		bool 	bLeaves;
		bool 	bLeavesTarget;
		bool 	bLeavesOutlier;
		bool 	bLevels;

		u_int 	nbNodes;
		u_int 	nbLeaves;
		u_int 	nbLeavesTarget;
		u_int 	nbLeavesOutlier;
		u_int 	nbLevels;
		double	timeTrain;
		const char * loadingBar;

			_DTStats()
			{
				bNodes = false;
				bLeaves = false;
				bLeavesTarget = false;
				bLeavesOutlier = false;
				bLevels = false;

				nbLeaves = 0;
				nbLeavesTarget = 0;
				nbLeavesOutlier = 0;
				nbNodes = 0;
				nbLevels = 0;
				timeTrain = 0.0;
				loadingBar = "|/-\\";
			}

			//~_DTStats();
};


/********************************************************************
*
*   Name: class DTree
*
*   Description:  Structure for representing a Decision Tree classifier
*       contains :
*           a pointer to the root node.
*           a pointer to the Data handler containing the training set.
* 	    a structure that contains statistics
*
*********************************************************************/
class DTree
{





	private:
		Node * 			root;
		DataHandler * 		trainSet;
		_DTStats 		stats;

bool rejectOutOfBounds;
double ** minmax;

		int			nodeIds;

		void 			computeStats(Node *);


	public:
					DTree(DataHandler * dataset);
					virtual ~DTree();


void 			computeStructure(Node *node,string nom);
int setConstraintRoot(bool rejet_nodes_temp,double ** minmaxval){
	minmax=minmaxval;
	rejectOutOfBounds=rejet_nodes_temp;
return 0;
}

bool getRejetNodes(){return rejectOutOfBounds;};
double** getMinMax(){return minmax;};

		Node *  		getRoot();
		DataHandler * 		getTrainSet();

		void 			addNode(Node * par, DataHandler * subset);
		void 			addLeaf(Node * par, DataHandler * subset, int predict);
		void			transformIntoLeaf(Node * par, u_int childInd, u_int (*p_predict)(DataHandler *));

		u_int 			predict(Instance * inst);
		Result *		test(DataHandler * testSet);

		u_int     		stat_getNbNodes();
		u_int     		stat_getNbLeaves();
		u_int     		stat_getNbLeavesTarget();
		u_int     		stat_getNbLeavesOutlier();
		u_int     		stat_getNbLevels();
		double			stat_getTimeTrain();
		char    		stat_getLoadingBar(u_int n);
		void 			stat_setTimeTrain(double time);

		string 			statsToString();
		string  		toString();
};

#endif /*DTREE_H_*/
