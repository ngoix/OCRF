#ifndef F_DTREE_H_
#define F_DTREE_H_

#include "../include/f_node.h"
#include "../include/dtree.h"




class F_DTree
{
	private :



		vector<u_int>	bag;

		F_Node *	root;
		Node *	root_node;
		DTree *	tree_temp;

DataHandler* trainSet;


vector<int> list_dim;///TODO:to remove to use only listsubspaceTree

u_int* listsubspaceTree;

double ** minmax;
bool rejectOutOfBounds;

		u_int 	nbNodes;
		u_int 	nbLeaves;
		u_int 	nbLeavesTarget;
		u_int 	nbLeavesOutlier;
		u_int 	nbLevels;


		void 		createFDTree(Node * node, F_Node * f_node);



	public :
				F_DTree();
				F_DTree(DTree * tree);
				F_DTree(string filename);
		virtual 	~F_DTree();

		vector<u_int>	getBag();

		int 		getNbNode();
		int 		getNbLeaves();
		int 		getNbLeavesOutlier();
		int 		getNbLeavesTarget();
		int 		getNbLevels();

		u_int		predict(Instance * inst);
		u_int 	 	recursPredict(Instance * inst, F_Node * node);

F_Node* getRoot(){return root;};
Node* getRootNode(){return root_node;};

void setListDim(vector<int> list);
void setListSubspace(u_int* list);

		void 		save(string filename, int id);
};

#endif /*F_DTREE_H_*/
