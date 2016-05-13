#ifndef RNDTREE_H_
#define RNDTREE_H_

#include "../include/utils.h"
#include "../include/dtinduc.h"

/********************************************************************
*
*   Name:           class RndTree
*
*   Description:  	Implementation class of the Random Tree Induction algorithm.
*       it inherits from the DTinduc class. Thus it must implement the growTree method.
*
*       A Random Tree is a tree grown with a modified CART algorithm, that implement a random feature selection
*       at each node for the splitting criterion selection.
*
*********************************************************************/
class RndTree : public DTInduc
{
	private:
		// Parameters
		int 		nbFeat;		// the size of the random subset of features selected for each node split
		bool 		gin;
		bool 		disp;
		int 		prePrun;	//pp the number of instances in a node under which the node is considered as a leaf
						//=> It is used to specify wether or not to used a prepruning. If not has to be set to 0

		double 		log2;
		bool 		rndFeatSel;///TODO: default value

		int  		growSubTree(Node * node, u_int ** sortedInd);
		Rule * 		featSelection(Node * node, u_int ** sortedInd);
		Rule * 		randomFeatSelection(Node * node, u_int ** sortedInd);
		long double	evalAttribute(Node * node, u_int attInd, u_int * sortedInd, double * splitPoint, double gini0, double w_size);
		long double 	eval(double n, u_int nbClass, double ** distribs, double * tots, u_int nbSplit);
		long double 	gini(double * distrib, u_int nbClass, double tot);
		long double	entropy(double * distrib, u_int nbClass, double tot);
		u_int ***	partitionNode(Node * node, Rule * rule, u_int ** sortedInd);
		bool  		is_leaf(DataHandler * set, u_int lvl);
		int		getPrediction(DataHandler * set);

	public:
        			RndTree(int nbf, bool d, bool gin=false);
		virtual		~RndTree();

		DTree * 	growTree(DataHandler * set);

		void		setNbFeatParam(int k);
};

#endif /*RNDTREE_H_*/
