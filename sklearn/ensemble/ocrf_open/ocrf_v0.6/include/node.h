#ifndef NODE_H_
#define NODE_H_

#include "../include/utils.h"
#include "../include/datahandler.h"
#include "../include/rule.h"

#define ROOT (u_int) 1
#define NODE (u_int) 2
#define LEAF (u_int) 3
#define ROOTLEAF (u_int) 4



class DTree;
class Node;


/********************************************************************
*
*   Name:           class Node
*
*   Description:  	Structure for representing a node
*
*********************************************************************/
class Node
{
	friend class 		DTree;

    protected :

	u_int 			id;
	u_int 			lvl;
	u_int 			type;

	DTree * 		tree;
	Node * 			parent;
	vector<Node *> 		children;
	DataHandler *		subset;

	Rule * 			splitRule;

	int 			prediction;

	void    		addChild(Node * node);
	void    		removeChild(Node * node);

	explicit		Node(DTree * dt, DataHandler * set, int ii);
	explicit 		Node(DTree * dt, DataHandler * set, int ii, Node * par, u_int level);
	explicit 		Node(DTree * dt, DataHandler * set, int ii, Node * par, u_int level, int predict);
        			Node():id(0),lvl(0),type(0),prediction(-1){
        				splitRule=nullptr;
        				subset=nullptr;
        				parent=nullptr;
        				tree=nullptr;
        			}

    public:
	virtual 		~Node();

	DataHandler * 		getDataSet();
	u_int     		getSize();
	u_int     		getId();
	u_int     		getLvl();
	u_int   		getNbChildren();
	Node *  		getChild(u_int i);
	Node *  		getParent();
	DTree * 		getTree();
	Rule * 			getSplitRule();
	u_int			getPrediction();

	void 			setRule(Rule * rule);

	void 			makeLeaf(int predic);
	void			rootToLeaf(int predic);

	bool    		is_leaf();
	bool is_root();
	bool 			is_parent(Node * n);

	u_int			predict(Instance * inst);

	string			toString();
};

#endif /*NODE_H_*/
