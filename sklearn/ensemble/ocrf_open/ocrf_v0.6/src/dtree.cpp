
#include "../include/dtree.h"


/* ********************************************************
 * Constructor
 * param
 *		dataset: pointer to the training set from which we want to induct the tree predictor
 */
DTree::DTree(DataHandler * dataset)
{


    trainSet = dataset;
    nodeIds = 0;
    root = new Node(this, trainSet, nodeIds);
    nodeIds++;

}

/* ********************************************************
 * Destructor
 */
DTree::~DTree()
{
    delete root;
    //delete minmax;
	//delete trainSet;///TODO: pointer cannot be freed here

//    cerr<<"tree deleted !"<<endl;
}

/* ********************************************************
 * Accessor to the root node.
 */
Node *  	DTree::getRoot(){ return root; }

/* ********************************************************
 * Accessor to the training set.
 */
DataHandler *  	DTree::getTrainSet(){ return trainSet; }

/* ********************************************************
 * Add a new node to the structure by giving the parent node and the data subset of the new node
 * param
 *		par: pointer to the parent node
 *		subset: pointer to the data subset handler from which the new node has to be build
 */
void		DTree::addNode(Node * par, DataHandler * subset)
{
	Node * n = new Node(this,subset,nodeIds,par,par->getLvl()+1);
	nodeIds++;
	par->addChild(n);
}

/* ********************************************************
 * Add a new leaf to the structure by giving the parent node, the data subset and the decision of the leaf
 * param
 *		par: pointer to the parent node
 *		subset: pointer to the data subset handler from which the new node has to be build
 *		predict: the class to be predicted by the leaf
 */
void		DTree::addLeaf(Node * par, DataHandler * subset, int predict)
{
	Node * l = new Node(this,subset,nodeIds,par,par->getLvl()+1,predict);
	nodeIds++;
	par->addChild(l);
}

/* ********************************************************
 * Predict the class of the given instance
 * param
 *		inst: pointer to the instance of which we want to predic the class
 */
u_int		DTree::predict(Instance * inst)
{
	return root->predict(inst);
}

/* ********************************************************
 * Test the tree predictor with the given test set.
 * param
 *		testSet: point to the testSet from which we want to test the tree predictor
 */
Result * 	DTree::test(DataHandler * testSet)
{
	Result * res = new Result(testSet,stats.timeTrain);
	for(v_inst_it it=testSet->begin();it!=testSet->end();it++)
	{
		u_int trueClass = testSet->getClass(*it);
		u_int predictClass = predict(*it);
		res->maj_confMat(trueClass, predictClass);
	}

	return res;
}

/* ********************************************************
 * Recursive method that allow to compute the statistics of the tree structure after it has already been built
 * param
 *		node: the current node, reached during the recursive process
 */
void 		DTree::computeStats(Node * node)
{
	stats.nbNodes++;
	if(node->is_leaf()) {



	    stats.nbLeaves++;

	    u_int c=node->getPrediction();
	    if(c==1) stats.nbLeavesOutlier++;
	    if(c==0) stats.nbLeavesTarget++;


	    //cerr<<"node is leaf:"<<node->getId()<<"/"<<stats.nbLeaves<<"/"<<stats.nbLeavesTarget<<"/"<<stats.nbLeavesOutlier<<endl;

	}
	if(stats.nbLevels <= node->getLvl()) stats.nbLevels = node->getLvl()+1;

	for(u_int i=0;i<node->getNbChildren();i++)
		computeStats(node->getChild(i));

	stats.bNodes = true;
	stats.bLeaves = true;
	stats.bLeavesTarget = true;
	stats.bLeavesOutlier = true;
	stats.bLevels = true;
}


/* *********************************************************
*permet de stocker dans un fichier texte la liste et la caractéristique de chacun des noeuds de l'arbre permettant une représentation graphique (en 2D ou 3D) du partitionnement final
*Recursive method
@param node from wich to build the tree (default:root)
@param nom the name of the file in which to save the structure
*/
void DTree::computeStructure(Node *node, string nom){

	if(node->is_leaf()){
		ofstream log_nodes(nom.c_str(),ios::out|ios::app);
		//log_nodes<<node->getId()<<"\t"<<node->getParent()->getId()<<"\t"<<node->getPrediction()<<"\tleaf"<<endl;

		if(node->is_root()){
//	cerr<<"node root:"<<endl;
		log_nodes<<node->getId()<<"\t"<<-1<<"\t"<<node->getPrediction()<<"\tleaf"<<endl;
}
else{

//	cerr<<"node root: false"<<endl;
		//log_nodes<<node->getId()<<"\t"<<node->getPrediction()<<"\tleaf"<<endl;
		log_nodes<<node->getId()<<"\t"<<node->getParent()->getId()<<"\t"<<node->getPrediction()<<"\tleaf"<<endl;
}

		log_nodes.close();
	}
	else{
			ofstream log_nodes(nom.c_str(),ios::out|ios::app);

			vector<_Split *> split=node->getSplitRule()->getSplits();
			if(split.size()>0){
				double split_sup=split.at(0)->getSup();
				double split_inf=split.at(0)->getInf();

				log_nodes<<node->getId()<<"\t"<<node->getSplitRule()->getAttId()<<"\t"<<split_inf<<"\t"<<split_sup<<"\t"<<node->getChild(0)->getId()<<"\t"<<node->getChild(1)->getId()<<endl;
			}
			log_nodes.close();

		for(u_int i=0;i<node->getNbChildren();i++){
			computeStructure(node->getChild(i),nom);
		}

	}

}



/* ********************************************************
 * Gives the number of nodes in the tree. This method automatically call the computeStats method if it has not been done before.
 */
u_int 		DTree::stat_getNbNodes()
{
	if(stats.bNodes == false)
		computeStats(root);

	return stats.nbNodes;
}

/* ********************************************************
 * Gives the number of leaf in the tree. This method automatically call the computeStats method if it has not been done before.
 */
u_int 		DTree::stat_getNbLeaves()
{
	if(stats.bLeaves == false)
		computeStats(root);

	return stats.nbLeaves;
}

/* ********************************************************
 * Gives the number of level in the tree structure. This method automatically call the computeStats method if it has not been done before.
 */
u_int 		DTree::stat_getNbLevels()
{
	if(stats.bLevels == false)
		computeStats(root);

	return stats.nbLevels;
}

u_int 		DTree::stat_getNbLeavesOutlier()
{
	if(stats.bLeavesOutlier == false)
		computeStats(root);

	return stats.nbLeavesOutlier;
}
u_int 		DTree::stat_getNbLeavesTarget()
{
	if(stats.bLeavesTarget == false)
		computeStats(root);

	return stats.nbLeavesTarget;
}

/* ********************************************************
 * Gives the time of the training process.
 */
double		DTree::stat_getTimeTrain()
{
	return stats.timeTrain;
}

/* ********************************************************
 *
 */
char    	DTree::stat_getLoadingBar(u_int n){ return stats.loadingBar[n%4]; }


/* ********************************************************
 * Set the time of the training process
 */
void		DTree::stat_setTimeTrain(double time)
{
	stats.timeTrain = time;
}


/* ********************************************************
 * Gives a printable string of the statistics of the tree.
 */
string 		DTree::statsToString()
{
	if(!stats.bLevels || !stats.bLeaves || !stats.bNodes) computeStats(root);

	string out;
	out += "\n Arbre de Décision construit en "; out += Utils::to_string((double) stats.timeTrain); out += " secondes";
	out += " \n Description : \n";
	out += "\t "; out += Utils::to_string((int)stats.nbNodes); out += " nodes \n";
	out += "\t "; out += Utils::to_string((int)stats.nbLeaves); out += " leaves \n";
	out += "\t "; out += Utils::to_string((int)stats.nbLevels); out += " levels \n";
	out += "\n\t for a training set of " ; out += Utils::to_string((int) trainSet->w_size()); out += " instances\n";
	return out;
}


/* ********************************************************
 * Gives a printable string of tree structure. If the number of nodes is more than 30, only statistics are returned in the string.
 */
string 		DTree::toString()
{
	string out = "";
	if(!stats.bLevels || !stats.bLeaves || !stats.bNodes) computeStats(root);
	if(stats.nbNodes < 30) out += (*root).toString();

	out += statsToString();

	return out;
}
