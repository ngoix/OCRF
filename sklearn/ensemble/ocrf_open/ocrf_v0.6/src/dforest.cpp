#include "../include/dforest.h"

/* *************************************************************
 * Constructor
 * param :
 * 		set : pointer to the training set
 */
DForest::DForest(DataHandler * set) {
	trainSet = set;
	minmax = trainSet->computeMinMax();

	rsm_predict = false;
}

//DForest::DForest(DataHandler * set,bool rsm_bool){
//	trainSet = set;
//	rsm=rsm_bool;
//}

/* *************************************************************
 * Constructor
 * param :
 * 		filename: string path to a .forest file that contains the forest to be loaded.
 */
DForest::DForest(string filename) {

	rsm_predict = false;
	minmax = nullptr;
	ifstream f(filename.c_str(), ios_base::in);

	if (f.is_open()) {
		string line;
		getline(f, line, '\n');

		/*DataHandler * handler*/
		trainSet = Arff::load(line.c_str());

		getline(f, line, '\n');
		getline(f, line, '\n');
		int nbTrees = Utils::from_string(line);

		for (int i = 0; i < nbTrees; i++) {
			getline(f, line, '\n');

			F_DTree* ftree=new F_DTree(line);
			forest.push_back(ftree);

			ftree->save("results/test",-1);

		}

	}

}

/* *************************************************************
 * Destructor
 */
DForest::~DForest() {

	delete[] minmax[0];
	delete[] minmax[1];
	delete[] minmax;

	for (vector<F_DTree *>::iterator it = forest.begin(); it != forest.end();
			it++) {

		delete *it; //*it;

	}
	forest.clear();

}

/* *************************************************************
 * return a pointer to the training set
 */
DataHandler * DForest::getDataHandler() {
	return trainSet;
}

/* *************************************************************
 * return the number of trees in the forest
 */
u_int DForest::getNbTrees() {
	return forest.size();
}

/* *************************************************************
 * return a trees of the forest
 */
F_DTree * DForest::getTree(u_int i) {
	return forest[i];
}

/* *************************************************************
 * add a decision tree to the forest
 * param :
 * 		tree : the Decision Tree to be added
 */
void DForest::addTree(F_DTree * tree) {
	forest.push_back(tree);
}

/* *************************************************************
 * Remove a decision tree from the forest
 * param :
 * 		ind : an index to the tree to remove.
 */
void DForest::removeTree(u_int ind) {
	//...
}

/* *************************************************************
 * Computes the margin function according to the given test dataset
 * param :
 *		inst : a pointer to the instance on which the margin function is computed
 */
double DForest::margin(Instance * inst) {
	int truClass = trainSet->getClass(inst);
	double A = 0.0, B = 0.0;

	vector<double> tabRes;
	for (u_int i = 0; i < trainSet->getNbClass(); i++)
		tabRes.push_back(0.0);

	for (u_int i = 0; i < forest.size(); i++) {
		//cerr<<"predict tree "<<i<<":";
		u_int res = forest[i]->predict(inst);
		tabRes[res]++;
		//cerr<<"classe:"<<res<<endl;
		//cin.get();

	}

	u_int * ind = Utils::sort(tabRes);
	int max1 = ind[trainSet->getNbClass() - 1];
	int max2 = ind[trainSet->getNbClass() - 2];

	A = tabRes[truClass];
	if (max1 == truClass)
		B = tabRes[max2];
	else
		B = tabRes[max1];

	A /= forest.size();
	B /= forest.size();

	delete[] ind;
	tabRes.clear();

	return (A - B);
}

/* *************************************************************
 * Computes the strength of the forest according to the given test dataset.
 * param :
 *		testset : the test set to be used for computing the strength
 */
double DForest::strength(DataHandler * testset) {
	double mgs = 0.0;
	for (int k = 0; k < (int) testset->size(); k++) {
		Instance * inst = testset->getInstance(k);
		mgs += margin(inst);
	}
	mgs /= testset->size();
	return mgs;
}

/* *************************************************************
 * Computes out-of-bag estimates of the strength of the forest.
 */
double DForest::strength() {
	double str = 0.0;
	for (u_int k = 0; k < trainSet->size(); k++) {
		Instance * inst = trainSet->getInstance(k);
		int truClass = trainSet->getClass(inst);
		double A = 0.0, B = 0.0;
		int nbTr = 0;
		vector<double> tabRes;
		for (u_int i = 0; i < trainSet->getNbClass(); i++)
			tabRes.push_back(0.0);

		for (u_int i = 0; i < forest.size(); i++) {
			vector<u_int> bag = forest[i]->getBag();
			if (find(bag.begin(), bag.end(), inst->getId()) == bag.end()) {
				nbTr++;
				u_int predictClass = forest[i]->predict(inst);
				tabRes[predictClass]++;
			}
		}
		u_int * ind = Utils::sort(tabRes);
		int max1 = ind[trainSet->getNbClass() - 1];
		int max2 = ind[trainSet->getNbClass() - 2];

		A = tabRes[truClass];
		if (max1 == truClass)
			B = tabRes[max2];
		else
			B = tabRes[max1];

		A /= nbTr;
		B /= nbTr;

		delete ind;

		str += (A - B);
	}

	str /= trainSet->size();
	return str;
}

double DForest::correlation(DataHandler * testset) {
//	double corr = 0.0;
///double rmg_plus=0;
///double mean_rmg_plus=0;
///double rmg_uncorrect_max=0;
	vector<double> rmg_theta_1;
	vector<double> rmg_theta_2;
	double sigma_theta_1 = 0;
	double sigma_theta_2 = 0;
	double sigma = 0;
	double rho_final = 0;

///double s_mean_1=0;
///double s_mean_2=0;

///double s2_theta_1=0;
///double s2_theta_2=0;

	int iter = 0;
	int nb = (int) testset->size();
	int nbTree = forest.size();
	for (int t1 = 0; t1 < nbTree; t1++) {
		for (int t2 = t1 + 1; t2 < nbTree; t2++) {
//calcul de la marge brute

			double s_mean_1 = 0;
			double s_mean_2 = 0;

			double s2_theta_1 = 0;
			double s2_theta_2 = 0;

			for (int j = 0; j < nb; j++) {

				Instance* inst = testset->getInstance(j);
				int trueclass = inst->getClass();

				int pred1 = forest.at(t1)->predict(inst);
				int pred2 = forest.at(t2)->predict(inst);

				double rmg_val = 0;
				if (trueclass == pred1)
					rmg_val = 1;
				else
					rmg_val = -1;
				rmg_theta_1.push_back(rmg_val);
				s_mean_1 += rmg_val / nb;
				s2_theta_1 += rmg_val * rmg_val / nb;

				rmg_val = 0;
				if (trueclass == pred2)
					rmg_val = 1;
				else
					rmg_val = -1;
				rmg_theta_2.push_back(rmg_val);
				s_mean_2 += rmg_val / nb;
				s2_theta_2 += rmg_val * rmg_val / nb;

			}
			sigma_theta_1 = sqrt(s2_theta_1 - s_mean_1 * s_mean_1);
			sigma_theta_2 = sqrt(s2_theta_2 - s_mean_2 * s_mean_2);

			for (int j = 0; j < nb; j++) {
				sigma += (rmg_theta_1.at(j) - s_mean_1)
						* (rmg_theta_2.at(j) - s_mean_2) / nb;
			}
			rmg_theta_1.clear();
			rmg_theta_2.clear();

			if (sigma_theta_1 * sigma_theta_2 == 0)
				return -2;		//error

			double rho = sigma / (sigma_theta_1 * sigma_theta_2);
			sigma = 0;
			rho_final += 2 * rho / (nbTree * nbTree - nbTree);

			iter++;

		}
	}
	return rho_final;
}

double DForest::correlation() {

	cerr << "WARNING - NO VALUE ADDED YET" << endl;
	return -1;
}

/* *************************************************************
 * Computes the Out-Of-Bag Estimates
 */
Result * DForest::getOOBestimates() {
	Result * res = new Result(trainSet, stats.timeTrain);
	for (u_int k = 0; k < trainSet->size(); k++) {
		Instance * inst = trainSet->getInstance(k);
		u_int trueClass = trainSet->getClass(inst);
		int cpt = 0;

		u_int * tabRes = new u_int[trainSet->getNbClass()];
		for (u_int i = 0; i < trainSet->getNbClass(); i++)
			tabRes[i] = 0;
		for (u_int i = 0; i < forest.size(); i++) {
			vector<u_int> bag = forest[i]->getBag();
			if (find(bag.begin(), bag.end(), inst->getId()) == bag.end()) {
				cpt++;
				u_int predictClass = forest[i]->predict(inst);
				tabRes[predictClass]++;
			}
		}
		u_int max = 0;
		for (u_int i = 1; i < trainSet->getNbClass(); i++)
			if (tabRes[max] < tabRes[i])
				max = i;

		res->maj_confMat(trueClass, max);
	}
	return res;
}

Result * DForest::getOOBOCestimates(u_int ** listsubspace, bool rsmOk, int nbRSM) {
	///TODO:to move to OCFOREST_H

	rsm_predict=rsmOk;

	Result * res = new Result(trainSet, stats.timeTrain);
	for (u_int k = 0; k < trainSet->size(); k++) {
		Instance * inst = trainSet->getInstance(k);

		u_int trueClass = trainSet->getClass(inst);
		int cpt = 0;

		u_int * tabRes = new u_int[2];
		//for(u_int i=0;i<trainSet->getNbClass();i++) tabRes[i] = 0;
		for (u_int i = 0; i < forest.size(); i++) {
			vector<u_int> bag = forest[i]->getBag();

			if (find(bag.begin(), bag.end(), inst->getOriginalId()) == bag.end()) {
				cpt++;

				Instance * curinst;

				if (!rsm_predict) {
					curinst = new Instance(inst->getOriginalId(), inst->getVect());
				} else {			//onrecopie les attributs des dimensions du rsm
					int nbAttTrainSet = (int) trainSet->getNbAttributes();

					vector<double> vals(nbRSM+1);
					vector<int> vals_att(nbRSM+1);
					int compt = 0;
					for (int f = 0; f < nbAttTrainSet; f++) {

						if (Utils::contains(listsubspace[i], f, nbRSM)) {
							vals[compt] = inst->at(f);

							vals_att[compt] = f;

							compt++;

						}

					}
					vals[nbRSM] = inst->getClass();
					curinst = new Instance(inst->getOriginalId(), &vals);
				}
				u_int predictClass = forest[i]->predict(curinst);

				delete curinst;
				tabRes[predictClass]++;
			}
		}




		u_int max = 0;
		for (u_int i = 1; i < 2; i++){
			if (tabRes[max] < tabRes[i]){
				max = i;
			}
		}
		delete[] tabRes;

		res->maj_confMat(trueClass, max);
	}

	return res;
}

/* *************************************************************
 * Give the Decision Forest prediction for a given instance
 * param :
 * 		inst : the instance that we want to predict the class
 */
u_int DForest::predict(Instance * inst) {


	u_int * tabRes = new u_int[trainSet->getNbClass()];

	for (u_int i = 0; i < trainSet->getNbClass(); i++)
		tabRes[i] = 0;


	for (u_int i = 0; i < forest.size(); i++) {
		u_int res = forest[i]->predict(inst);
		tabRes[res]++;
	}

	u_int max = 0;
	for (u_int i = 1; i < trainSet->getNbClass(); i++)
		if (tabRes[max] < tabRes[i])
			max = i;

	delete tabRes;
	return max;
}

u_int DForest::predictOC(Instance * inst, u_int ** listsubspace, bool rsmOk,
		int nbRSM, string filename) {

	int nb = trainSet->getNbClass();

	u_int * tabRes = new u_int[nb];

	vector<double> vals(nbRSM + 1);
	vector<int> vals_att(nbRSM + 1);

	int compt;

	rsm_predict = rsmOk;

	for (int i = 0; i < (int) trainSet->getNbClass(); i++)
		tabRes[i] = 0;

	for (int i = 0; i < (int) forest.size(); i++) {

		Instance * curinst;

		if (!rsm_predict) {

			curinst = new Instance(inst->getId(), inst->getVect());

		} else {			//onrecopie les attributs des dimensions du rsm

			///mettre ici la liste des sous-esapces dans fdtree; couteux dans la boucle appelante a predictOC cependant
			///forest[i]->setListSubspace(listsubspace[i]);

			int nbAttTrainSet = (int) trainSet->getNbAttributes();

			compt = 0;
			for (int f = 0; f < nbAttTrainSet; f++) {

				if (Utils::contains(listsubspace[i], f, nbRSM)) {
					vals[compt] = inst->at(f);

					vals_att[compt] = f;

					compt++;

				}

			}

			vals[nbRSM] = inst->getClass();
			curinst = new Instance(inst->getId(), &vals);

			forest[i]->setListDim(vals_att);
			//forest[i]->setListSubspace(listsubspace[i]);

		}

		u_int res = forest[i]->predict(curinst);

		delete curinst;
		tabRes[res]++;

	}

	if (filename.compare("") != 0) {

		ofstream resultsDecision(filename.c_str(), ios::out | ios::app);
		resultsDecision << "\n" << inst->getId() << "\t" << inst->getClass()
				<< "\t";
		for (int j = 0; j < nb - 1; j++) {
			resultsDecision << tabRes[j] << "\t";
		}
		resultsDecision << tabRes[nb - 1];
		resultsDecision.close();
	}

	u_int max = 0;
	for (u_int i = 1; i < trainSet->getNbClass(); i++)
		if (tabRes[max] < tabRes[i])
			max = i;

	vals.clear();
	vals_att.clear();

	delete[] tabRes;

	return max;
}

/* *************************************************************
 * launch the test procedure for a given test set
 * param :
 * 		testSet : the test set to be used for the test procedure.
 */
Result * DForest::test(DataHandler * testSet) {
	Result * res = new Result(testSet, stats.timeTrain);

	for (v_inst_it it=testSet->begin();it!=testSet->end();it++)
	{
		cout << "test pour : "<<(*it)->toString()<<"\n";
		u_int trueClass = testSet->getClass(*it);
		u_int predictClass = predict(*it);
		cout << "resultat : "<<predictClass<< " pour : "<<trueClass<<"\n";
		res->maj_confMat(trueClass, predictClass);
	}

	return res;
}

Result * DForest::testOC(DataHandler * testSet, u_int ** listsubspace,
		bool rsmOk, int nbRSM, string filename) {

	Result * res = new Result(testSet, stats.timeTrain);

	int i = 0;
	for (v_inst_it it=testSet->begin();it!=testSet->end();it++)
	{

		u_int trueClass = testSet->getClass(*it);

		u_int predictClass = predictOC(*it,listsubspace,rsmOk,nbRSM,filename);

		res->maj_confMat(trueClass, predictClass);
/*
		if(trueClass!=predictClass){
			cout<<"["<<(*it)->getId()<<"/"<<(*it)->getOriginalId()<<":"<<trueClass<<"=>"<<predictClass<<"]"<<endl;
		}
*/


		i++;
	}

	return res;
}

int DForest::saveFile(string dest) {

	int nb = forest.size();

	for (int i = 0; i < nb; i++) {

		/*backup
		 *nodes
		 *rule
		 *prediction

		 */
//fileForest<<forest[i]->
		stringstream out;
		out << i;

		string dest_tree = dest + "/tree_" + out.str() + ".txt";

		Node * start_node = forest[i]->getRootNode();

		writeFileForest(dest_tree, start_node);	//starting node

	}

	return 0;
}

int DForest::writeFileForest(string dest_tree, Node * starting_node) {

	ofstream fileForest(dest_tree.c_str(), ios::out | ios::app);

	fileForest << "#parent\t" << starting_node->getId() << "\t#node_size\t"
			<< starting_node->getSize() << "\t";

	Rule * rule_split = starting_node->getSplitRule();

	fileForest << "#attribute\t" << rule_split->getAttId() << "\t#split\t"
			<< rule_split->getSupSplit() << "\t";

	if (!starting_node->is_leaf()) {

		fileForest << "#parent_split\t#left\t" << starting_node->getChild(0)->getId()
				<< "\t";
		fileForest << "#right\t" << starting_node->getChild(1)->getId() << "\n";

		fileForest.close();

		for (u_int j = 0; j < starting_node->getNbChildren(); j++) {
			Node * child = starting_node->getChild(j);
			writeFileForest(dest_tree, child);	//starting node
		}

	} else {
		/*
		 *prediction
		 */
		fileForest << "#leaf\t" << starting_node->getPrediction() << "\n";

	}

	fileForest.close();

	return 0;
}

double DForest::stat_getTimeTrain() {
	return stats.timeTrain;
}

double DForest::stat_getMeanDTTimeTrain() {
	return stats.meanDTTimeTrain;
}

char DForest::stat_getLoadingBar(u_int n) {
	return stats.loadingBar[n % 4];
}

void DForest::stat_setTimeTrain(double time) {
	stats.timeTrain = time;
}

void DForest::stat_setMeanDTTimeTrain(double time) {
	stats.meanDTTimeTrain = time;
}

void DForest::save(string filename) {

	cout << "Saving file ..." << endl;
	ofstream file(filename.c_str());
	if (file.is_open()) {

		if (trainSet->getFileName().empty()) {///Unable to retrieve original file path (inputstream or else)
			string fileDataBackup = filename;
			fileDataBackup += "_data.arff";
			Arff::save(fileDataBackup.c_str(), trainSet);
		}

		file << trainSet->getFileName() << endl;

		for (u_int i = 0; i < trainSet->size(); i++) {
			file << trainSet->getInstance(i)->getId();
			file << " ";
		}
		file << endl;

		file << Utils::to_string((int) forest.size());
		file << endl;

		for (u_int i = 0; i < forest.size(); i++) {
			string fileTreeName = filename;
			//fileTreeName.erase(fileTreeName.end() - 6, fileTreeName.end());///TODO:search for last occurrence of "."

			size_t pos=fileTreeName.rfind(".");
			if(pos!=string::npos){
			fileTreeName.erase(fileTreeName.begin()+pos, fileTreeName.end());///TODO:search for last occurrence of "."
			}
			else{

			}
			fileTreeName += Utils::to_string(i);
			fileTreeName += ".tree";
			forest[i]->save(fileTreeName, i);
			file << fileTreeName << "\n";
			file.flush();
		}

	} else {

		cerr << "Unable to save model" << endl;

	}
	file.close();

	cout << "Saving file OK." << endl;
}

string DForest::statsToString() {
	string out = "";
/////////////////////////////////////////////
	return out;
}

string DForest::toString() {
	string res = "";
	res += "Nombre d'arbres : ";
	res += Utils::to_string((int) forest.size());

	return res;
}

double DForest::getMeanNode() {
	int nbArbre = forest.size();
	unsigned long int totalnode = 0;
	for (int i = 0; i < nbArbre; i++) {
		totalnode += forest[i]->getNbNode();
	}
	return ((double) totalnode / (double) nbArbre);
}
