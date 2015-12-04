#ifndef CART_H_
#define CART_H_

#include "../include/utils.h"
#include "../include/dtinduc.h"

/********************************************************************
*
*   Name:           class CART
*
*   Description:  	Implementation class of the CART Induction algorithm.
*       it inherits from the DTinduc class. Thus it must implement the growTree method.
*
*
*********************************************************************/
class Cart : public DTInduc
{
	private:
		// Parameters
		bool 		gin;
		bool 		disp;
		int 		prePrun;	//pp the number of instances in a node under which the node is considered as a leaf
						//=> It is used to specify weather or not to used a prepruning. If not has to be set to 0

		double 		ln2;

		int  		growSubTree(Node * node, u_int ** sortedInd);
		Rule * 		featSelection(Node * node, u_int ** sortedInd);
		long double	evalAttribute(Node * node, u_int attInd, u_int * sortedInd, double * splitPoint, double gini0, double w_size);
		long double 	eval(double n, u_int nbClass, double ** distribs, double * tots, u_int nbSplit);
		long double 	gini(double * distrib, u_int nbClass, double tot);
		long double	entropy(double * distrib, u_int nbClass, double tot);
		u_int ***	partitionNode(Node * node, Rule * rule, u_int ** sortedInd);
		bool  		is_leaf(DataHandler * set, u_int lvl);
		int		getPrediction(DataHandler * set);

	public:
        			Cart(bool d, bool gin=false);
		virtual		~Cart();

		DTree * 	growTree(DataHandler * set);
};

#endif /*CART_H_*/
