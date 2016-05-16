
#include "../include/f_dtree.h"

F_DTree::F_DTree(DTree * tree)
{

listsubspaceTree=NULL;
///list_dim=std::vector<int>();
rejectOutOfBounds=tree->getRejetNodes();
minmax=tree->getMinMax();
root_node=tree->getRoot();



	for(v_inst_it it=tree->getTrainSet()->begin();it!=tree->getTrainSet()->end();it++){
//		bag.push_back((*it)->getId());
		bag.push_back((*it)->getOriginalId());
	}

	nbNodes = tree->stat_getNbNodes();
	nbLeaves = tree->stat_getNbLeaves();
	nbLeavesTarget =  tree->stat_getNbLeavesTarget();
	nbLeavesOutlier = tree->stat_getNbLeavesOutlier();
	nbLevels = tree->stat_getNbLevels();

	if(tree->getRoot()->is_leaf())
		root = new F_Node(tree->getRoot()->getPrediction(),0);
	else
	{
		root = new F_Node(tree->getRoot()->getSplitRule(),0);
		createFDTree(tree->getRoot(),root);
	}

}

F_DTree::F_DTree()
{
	nbNodes = 0;
	nbLeaves = 0;
	nbLeavesTarget = 0;
	nbLeavesOutlier = 0;
	nbLevels = 0;

bag.clear();
	root = NULL;


	if(listsubspaceTree!=NULL){
		delete [] listsubspaceTree;
	}

}

int F_DTree::getNbNode(){

return nbNodes;
}

int F_DTree::getNbLeaves(){

	return nbLeaves;
}

int F_DTree::getNbLeavesTarget()
{
	return nbLeavesTarget;
}

int F_DTree::getNbLeavesOutlier()
{
	return nbLeavesOutlier;
}

int F_DTree::getNbLevels()
{
	return nbLevels;
}



F_DTree::F_DTree(string filename){

	ifstream file(filename.c_str(),ios_base::in);
	if(file.is_open()){

		string line;
		getline(file,line,'\n');
		vector<string> tokens;
		Utils::tokenize(line, tokens, " ");

		for(u_int i=0;i<tokens.size();i++){
			bag.push_back((u_int) Utils::from_string(tokens[i]));
		}

		getline(file,line,'\n');
		nbNodes = (int) Utils::from_string(line);


		map<int,F_Node *> nodes;
		map<int,int> lefts;
		map<int,int> rights;
		map<int,u_int> predictions;

		for(u_int i=0;i<nbNodes;i++){

			getline(file,line,'\n');

			vector<string> tokens;
			Utils::tokenize(line, tokens, " ");

			int id = (int) Utils::from_string(tokens[0]);
			if(tokens.size() == 5)
			{
				int attId = (int) Utils::from_string(tokens[1]);
				double split = Utils::from_string(tokens[2]);
				Rule * rule = new Rule(attId,split);
				F_Node * node = new F_Node(rule,id);
				nodes[id] = node;
				delete rule;

				lefts[id] = ((int) Utils::from_string(tokens[3]));
				rights[id] = ((int) Utils::from_string(tokens[4]));
			}
			else
			{
				predictions[id] = ((u_int) Utils::from_string(tokens[1]));
				F_Node * node = new F_Node(predictions[id],id);
				nodes[id] = node;
			}
		}

		for(u_int i=0;i<nbNodes;i++)
		{
			if(predictions.count(i) == 0)
			{
				int left = lefts[i];
				int right = rights[i];
				nodes[i]->children.push_back(nodes[left]);
				nodes[i]->children.push_back(nodes[right]);
			}
		}
		root = nodes[0];


	}
}
/*
void F_DTree::recursiveDelete(F_Node * n){

if(n==NULL){
return;
}
else{

	for(u_int i=0;i<n->getNbChildren();i++){
		F_node * child = n->getChild(i);

	}

}

}
*/

F_DTree::~F_DTree()
{

bag.clear();
//root=NULL;
delete root;
//delete trainSet;
}


void F_DTree::setListDim(vector<int> list){
	list_dim=list;
}

void F_DTree::setListSubspace(u_int* list){
	listsubspaceTree=list;
}

void 	F_DTree::createFDTree(Node * node, F_Node * f_node){



	for(u_int i=0;i<node->getNbChildren();i++){
		Node * child = node->getChild(i);

		F_Node * f_n = NULL;



		if(child->is_leaf()){
			f_n = new F_Node(child->getPrediction(),child->getId());

		}
		else{
			f_n = new F_Node(child->getSplitRule(),child->getId());




		}



		f_node->children.push_back(f_n);

		if(!child->is_leaf()) createFDTree(child,f_n);

	}



}


vector<u_int>	F_DTree::getBag(){

return bag;

}

u_int	F_DTree::predict(Instance * inst){

if(rejectOutOfBounds){

vector<double> v=inst->getVectSimple();

int nbD=v.size();


bool active_ss=(listsubspaceTree!=NULL);//or !list_dim.empty()
int found=0;
string bar="";
for(int i=0;i<nbD-1;i++){




	int d=i;

if(active_ss){
	d=list_dim[i];//match original attributes ID from RSM
}

double val=inst->at(i);
double min_val=minmax[0][d];
double max_val=minmax[1][d];


if(val>max_val || val<min_val) {

found++;
/*
stringstream str;
str<<found<<"/"<<nbD-1;
bar="("+str.str()+")-";
*/

}//0:outlier
else{
	break;//found one dimension
}


}

//if(found>0) cout<<bar;

if(found>=nbD-1){//outside bounds for all dimensions
	//cout<<"\n\nfound !"<<endl;
	//cin.get();
	return OUTLIER;
}

//if(found>0) cout<<endl;

}

return recursPredict(inst, root);
}

u_int 	F_DTree::recursPredict(Instance * inst, F_Node * node){




if(node->splitRule == NULL){



	return node->prediction;

}


u_int childInd = node->splitRule->evaluate(inst->at(node->splitRule->getAttId()));


return recursPredict(inst,node->children[childInd]);

}

void 	F_DTree::save(string filename, int id)
{
	ofstream file(filename.c_str());
	if(file.is_open())
	{
		for(u_int i=0;i<bag.size();i++)
		{
			file << Utils::to_string((int) bag[i]);
			file << " ";
		}
		file << "\n";
		file << Utils::to_string((int)nbNodes);
		file << "\n";
		file << (*root).fileToString().c_str();
		file.flush();
	}
	file.close();
}



