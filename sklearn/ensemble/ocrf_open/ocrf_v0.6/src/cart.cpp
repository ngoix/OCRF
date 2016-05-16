
#include "../include/cart.h"

/* ************************************************************************
 * Constructor
 * param :
 * 	d : display option. if true the consol will display information about the inducting progression
 * 	g : specify weather or not to use the gini index. If not, the entropy measure is used instead
 */
Cart::Cart(bool d, bool g)
: DTInduc()
{
	gin = g;
	prePrun = 1;
	disp = d;

	ln2 = log(2.0);
}

/* ************************************************************************
 * Destructor
 */
Cart::~Cart()
{}

/* ************************************************************************
 * launch the tree induction process.
 */
DTree * 	Cart::growTree(DataHandler * set)
{
	trainSet = set;

	if(trainSet == NULL)
	{
		if(disp)
		{
			string tmp = "\n\nERREUR : RndTree::growTree()";
			tmp += "\n\t Vous n'avez désigné aucune base de données pour l'apprentissage";
			Utils::print(tmp);
		}
		return NULL;
	}

	// grow an empty tree, i.e. with a single node (the root node)
	DTree * p_tree = new DTree(trainSet);

	// If the trainSet contains only data from one class, the root is transform into a leaf
	if(is_leaf(trainSet,0))
	{
		// the root node is transform into a leaf
		int predic = getPrediction(trainSet);
		p_tree->getRoot()->rootToLeaf(predic);
		p_tree->stat_setTimeTrain(0.0);
		return p_tree;
	}

	// initialization :
	// Have a 2D array to sort data indices according to each attribute
	u_int ** sortedInd = new u_int*[trainSet->dim()];
	for(u_int i=0;i<trainSet->dim();i++)
	{
		if(i != trainSet->getClassInd())
		{
			vector<double> vals;
			for(v_inst_it it=trainSet->begin();it!=trainSet->end();it++)
				vals.push_back((*it)->at(i));

			sortedInd[i] = Utils::sort(vals);
		}
	}

	double _time = ((double) clock());

	// launch tree growing from the tree root.
	int ret = growSubTree(p_tree->getRoot(),sortedInd);
	if(ret == -1)
	{
		string tmp = "\n\nERREUR : RndTree::growSubTree()";
		Utils::print(tmp);
	}

	_time = (((double) clock()) - _time) /CLOCKS_PER_SEC;
	if(_time>=0) p_tree->stat_setTimeTrain(_time);
	else p_tree->stat_setTimeTrain(0.0);

	for(u_int i=0;i<trainSet->dim();i++)
		if(i != trainSet->getClassInd()) delete[] sortedInd[i];
	delete[] sortedInd;

	return p_tree;
}


/* ************************************************************************
 * Recursive function to grow trees. Compute a splitting rule and make the partition
 * for the current node. finally make a recursive call with every child node created.
 * param :
 * 		node : the current node to be split
 * 		sortedInd : a 2D array of sorted index of instances.
 */
int 	Cart::growSubTree(Node * node, u_int ** sortedInd)
{
	DataHandler * data = node->getDataSet();

	// launch the random feature selection process and create the splitting rule with the chosen feature.
	Rule * rule = featSelection(node,sortedInd);

	// if no rule has been created, the branch induction is stopped.
	// (It means that no feature has managed to produce a split)
	if (rule == NULL)
	{
		// the node is transform into a leaf
		int predic = getPrediction(data);
		node->makeLeaf(predic);
	}
	else
	{
		u_int *** subsets = NULL;

		node->setRule(rule);
		subsets = partitionNode(node,rule,sortedInd);

		for(u_int j=0;j<node->getNbChildren();j++)
		{
			Node * n = node->getChild(j);
			if(!(n->is_leaf()))
				growSubTree(n,subsets[j]);

			for(u_int k=0;k<data->dim();k++)
				if(k != data->getClassInd()) delete[] subsets[j][k];
			delete[] subsets[j];
		}
		delete[] subsets;
	}

	return 0;
}

/* ************************************************************************
 * Function that implements the Random Feature Selection process.
 * param :
 * 		node : the current node to be split
 * 		sortedInd : an array of instance indices sorted by each attribute values.
 *
 * Return the Rule object for the split procedure that have been produced with the selected criterion criterion.
 */
Rule * 		Cart::featSelection(Node * node, u_int ** sortedInd)
{
	DataHandler * data = node->getDataSet();

	long double bestGain = 0.0;
	double bestSplit = 0.0;
	u_int bestAtt = data->getClassInd();
	bool found = false;

	double w_size = data->w_size();

	long double eval0;
	if(gin) eval0 = gini(data->getDistrib(),data->getNbClass(),w_size);
	else eval0 = entropy(data->getDistrib(),data->getNbClass(),w_size);

	for(u_int attIndex=0;attIndex<data->dim();attIndex++)
	{
		if(attIndex == data->getClassInd()) continue;

		double split;
		long double gain = evalAttribute(node,attIndex,sortedInd[attIndex],&split,eval0,w_size);
		if(gain > bestGain)
		{
			bestGain = gain;
			bestAtt = attIndex;
			bestSplit = split;
			found = true;
		}
	}

	if(!found) return NULL;

	u_int bestAttId = data->getAttribute(bestAtt)->getId();

	if(data->getAttribute(bestAtt)->is_nominal()) return new Rule(bestAttId,data->getAttribute(bestAtt)->getNbModal());
	else return new Rule(bestAttId,bestSplit);
}

/* ************************************************************************
 * Function that evaluates the quality of the current split
 * param :
 * 		node : the current node to be split
 * 		attInd : the index of the current feature to be evaluated as a splitting criterion
 * 		sortedInd : the 1D sorted instance index array of the corresponding attribute
 * 		splitPoint : a double pointer to receive the splitting point
 * 		eval0 : the evaluation measure of the current node's subset
 * 		w_size : the weighted size of the current node's subset
 *
 * return the value of the best split point evaluation, i.e. the best measure for the current criterion
 */
long double 	Cart::evalAttribute(Node * node, u_int attInd, u_int * sortedInd, double * splitPoint, double eval0, double w_size)
{
	DataHandler * data = node->getDataSet();
	u_int nbClass = data->getNbClass();
	bool ok = false;

	if(data->getAttribute(attInd)->is_nominal())
	{
    	/*********** Nominal Attribute **************/
    		return 0.0;
	}
	else
	{
    	/*********** Numeric Attribute **************/
		double * totEff = new double[2];
		totEff[0] = 0.0;
		totEff[1] = 0.0;
		double ** classEff = new double*[2];
		classEff[0] = new double[nbClass];
		classEff[1] = new double[nbClass];

		for(u_int i=0;i<nbClass;i++)
		{
			classEff[0][i] = 0.0;
			classEff[1][i] = data->getDistrib(i);
			totEff[1] += classEff[1][i];
		}

		double currSplit = trainSet->getInstance(sortedInd[0])->at(attInd);
		long double bestEval = -(numeric_limits<long double>::max());

		for(u_int i=0;i<data->size();i++)
		{
			Instance * inst = trainSet->getInstance(sortedInd[i]);
			double currVal = inst->at(attInd);
			u_int currClass = data->getClass(inst);
			double currWeight = data->getWeight(inst);

			if(currVal > currSplit)
			{
				long double currEval = eval0 - eval(w_size,nbClass,classEff,totEff,(u_int)2);
				if((currEval > 0) && (currEval > bestEval))
				{
					ok = true;
					bestEval = currEval;
					(*splitPoint) = (currVal + currSplit) / 2.0;
				}
			}

			currSplit = currVal;
			classEff[0][currClass] += currWeight;
			classEff[1][currClass] -= currWeight;
			totEff[0] += currWeight;
			totEff[1] -= currWeight;
		}

		delete[] totEff;
		delete[] classEff[0];
		delete[] classEff[1];
 		delete[] classEff;

 		if(ok) return bestEval;
 		else return -1.0;
	}
}

/* ************************************************************************
 * Function that evaluates the quality of the current split
 * param :
 * 		n : the weighted size of the current node's subset
 * 		nbClass : the number of class possible values
 * 		distribs : a 2D array to memorize class distribution for each child node to be created
 * 		tots : an array of total size of each child node subset
 * 		nbSplit : the number of child node to be created
 */
long double 	Cart::eval(double n, u_int nbClass, double ** distribs, double * tots, u_int nbSplit)
{
	long double eval = 0.0;

	for(u_int i=0;i<nbSplit;i++)
	{
		if(tots[i] != 0.0)
		{
			long double i_t;
			if(gin) i_t = gini(distribs[i],nbClass,tots[i]);
			else i_t = entropy(distribs[i],nbClass,tots[i]);
			eval += ((tots[i]/n) * i_t);
		}
	}

	return eval;
}

/* ************************************************************************
 * Gini index
 * param :
 * 		nbClass : the number of class possible values
 * 		distribs : the class distributions of the current subset
 * 		tot : the size of the current subset
 */
long double 	Cart::gini(double * distrib, u_int nbClass, double tot)
{
	if(tot == 0.0) return 0.0;

	long double gini = 0.0;
	for(u_int i=0;i<nbClass;i++)
	{
		long double nj_n = distrib[i]/tot;
		gini += (nj_n * (1 - nj_n));
	}

	return gini;
}

/* ************************************************************************
 * Entropy measure
 * param :
 * 		nbClass : the number of class possible values
 * 		distribs : the class distributions of the current subset
 * 		tot : the size of the current subset
 */
long double 	Cart::entropy(double * distrib, u_int nbClass, double tot)
{
	if(tot == 0.0) return 0.0;

	long double entropy = 0.0;
	for(u_int i=0;i<nbClass;i++)
		if(distrib[i] != 0)
			entropy -= (distrib[i]*log(distrib[i]));

	return ((entropy+(tot*log(tot)))/(tot*ln2));
}


/* ************************************************************************
 * Perform the partitionning process on the given node with the specified rule
 * param :
 * 		node : the current node to be split
 * 		rule : the rule to be used for the splitting process
 * 		sortedInd : the 1D sorted instance index array of the corresponding attribute
 */
u_int ***	Cart::partitionNode(Node * node, Rule * rule, u_int ** sortedInd)
{
	DataHandler * data = node->getDataSet();
	DTree * tree = node->getTree();
	u_int attInd = rule->getAttId();
	u_int nb = rule->getNbSplits();

	// get the position of the rule attribute
	for(u_int i=0;i<data->dim();i++)
	{
		u_int id = data->getAttribute(i)->getId();
		if(attInd == id)
		{
			attInd = i;
			break;
		}
	}

	// have an array of DataHandler objects to create data subsets for each
	// child node to be created.
	vector<DataHandler *> tab;
	for(u_int j=0;j<nb;j++)
		tab.push_back(new DataHandler(data->getDataSet(),data->getClassInd(),false));

	// have an array to memorize node ids of each data of the current node.
	map<u_int,u_int> nodeIds;

	// Partition the current dataset and memorize the child node id of each object
	for(u_int i=0;i<data->size();i++)
	{
		u_int id = sortedInd[attInd][i];
		Instance * inst = trainSet->getInstance(id);
		double v = inst->at(attInd);
		u_int nodeId = rule->evaluate(v);
		nodeIds[id] = nodeId;
		tab[nodeId]->addInstance(inst,data->getWeight(inst));
	}

	// have an array of data subsets for the child nodes to be created. Those subsets are to be used
	// for partitionning the array of sorted indices
	u_int *** subsets = new u_int**[nb];

	for(u_int i=0;i<nb;i++)
	{
		// create child nodes
		if(is_leaf(tab[i],(node->getLvl())+1))
		{
			u_int predict = getPrediction(tab[i]);
			if(predict == (u_int) -1) { cout << "ERREUR get predic" << endl; return NULL; }
			tree->addLeaf(node,tab[i],predict);
		}
		else
		{
			tree->addNode(node,tab[i]);
		}

		// allocate subsets memory arrays.
		subsets[i] = new u_int*[data->dim()];
		u_int size = tab[i]->size();
		for(u_int j=0;j<data->dim();j++)
			if(j != data->getClassInd()) subsets[i][j] = new u_int[size];

	}

	u_int * inds = new u_int[nb];
	for(u_int j=0;j<data->dim();j++)
	{
		if(j != data->getClassInd())
		{
			for(u_int l=0;l<nb;l++) inds[l] = 0;

			for(u_int k=0;k<data->size();k++)
			{
				u_int id = sortedInd[j][k];
				u_int nodeId = nodeIds[id];
				subsets[nodeId][j][inds[nodeId]] = id;
				inds[nodeId]++;
			}
		}
	}

	delete[] inds;

	return subsets;
}

/* *******************************************************
 * Decide weather or not a subset belong to a leaf or an internal node.
 * param :
 * 		set : the given subset to be tested
 * 		lvl : the level of its node in the tree.
 */
bool 		Cart::is_leaf(DataHandler * set, u_int lvl)
{
	// pre-pruning : a node is a leaf if its subset contains less than PREPRUN instances
	if(((int)(set->size())) <= prePrun) return true;

	// Else, it can be a leaf if all its instances represent the same class
	u_int compt = 0;
	for(u_int i=0;i<set->getNbClass();i++)
	{
		if(set->getDistrib(i) > 0.0) compt++;
		if(compt>1) return false;
	}
	return true;
}

/* *******************************************************
 * Compute the most represented class value in the subset
 */
int		Cart::getPrediction(DataHandler * set)
{
	int predic = 0;
	bool ok = false;
	double dist = -(numeric_limits<double>::max());

	for(u_int i=0;i<set->getNbClass();i++)
	{
		if(dist < (set->getDistrib(i)))
		{
			ok = true;
			predic = i;
			dist = set->getDistrib(i);
		}
	}

	if(ok) return predic;
	else return -1;
}
