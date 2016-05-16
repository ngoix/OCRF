#include "../include/ocforest.h"

OCForest::OCForest(params_forest pForest, params_oneclass pOneclass,
		string dbName) :
		RFInduc(pForest.param_nbTrees, false){

	handltempAmont = nullptr;

	rsm			= pForest.param_rsm;
	bagging   	= pForest.param_bagging;
	nbRSM 		= pForest.param_KRSM;
	Ltree 		= pForest.param_nbTrees;
	K_RFS 		= pForest.param_KRFS;
	log_trees 	= pForest.param_logTrees;
	r 			= pForest.param_percentBagging;
	studyNodes 	= pForest.param_studyNodes;

	generateInOriginalDataSet 	= pOneclass.param_generateInOriginalDataSet;
	nbOutlier 					= pOneclass.param_nbOutlierData;
	alpha_domain 					= pOneclass.param_alpha;
	rejectOutOfBounds 			= pOneclass.param_rejectOutOfBounds;
	optimize_gen 				= pOneclass.param_optimize;
	histoOutlier 				= pOneclass.param_histogramOutlierData;
	boundsInterTarget 			= pOneclass.param_boundsInterTargetData;

	database_name = dbName;

	////TODO: init in class constructor
	OCRFStats.nbNodesTotal = 0;
	OCRFStats.nbLevelsTotal = 0;
	OCRFStats.nbLeavesTotal = 0;
	OCRFStats.nbLeavesTargetTotal = 0;
	OCRFStats.nbLeavesOutlierTotal = 0;

	OCRFStats.varNodesTotal = 0;
	OCRFStats.varLeavesTotal = 0;
	OCRFStats.varLeavesTargetTotal = 0;
	OCRFStats.varLeavesOutlierTotal = 0;
	OCRFStats.varLevelsTotal = 0;


	listsubspace = nullptr;

	if (rsm) {
		listsubspace = new u_int*[Ltree];
	} else {

	}

	srand((unsigned) time(0));///TODO:remove from generator

	generator = new OutlierGenerator(generateInOriginalDataSet, bagging, rsm, r, nbRSM,
			nbOutlier, alpha_domain, optimize_gen);

}

OCForest::~OCForest() {

	if (rsm) {

		for (int i = 0; i < Ltree; i++)
				{
			delete[] listsubspace[i];
		}
		delete[] listsubspace;
		listsubspace = nullptr;
	} else {
	}

	delete generator;
	generator = nullptr;

	if (generateInOriginalDataSet) {
		delete handltempAmont;
		handltempAmont=nullptr;
	}

}

int OCForest::getNbNodes() {
	return OCRFStats.nbNodesTotal;
}
int OCForest::getNbLeaves() {
	return OCRFStats.nbLeavesTotal;
}
int OCForest::getNbLeavesTarget() {
	return OCRFStats.nbLeavesTargetTotal;
}
int OCForest::getNbLeavesOutlier() {
	return OCRFStats.nbLeavesOutlierTotal;
}
int OCForest::getNbLevels() {
	return OCRFStats.nbLevelsTotal;
}

double OCForest::getVarNbNodes() {
	return OCRFStats.varNodesTotal;
}
double OCForest::getVarNbLeaves() {
	return OCRFStats.varLeavesTotal;
}
double OCForest::getVarNbLeavesTarget() {
	return OCRFStats.varLeavesTargetTotal;
}
double OCForest::getVarNbLeavesOutlier() {
	return OCRFStats.varLeavesOutlierTotal;
}
double OCForest::getVarNbLevels() {
	return OCRFStats.varLevelsTotal;
}

DForest * OCForest::growForest(DataHandler * set) {

	double _time = ((double) clock());
	_time = _time / CLOCKS_PER_SEC;
	double averageTime = 0.0;
	//unsigned long int nbNodestot=0;
	double meanNodesTotal = 0;
	double mean2NodesTotal = 0;

	double meanLeavesTotal = 0;
	double mean2LeavesTotal = 0;

	double meanLeavesTargetTotal = 0;
	double mean2LeavesTargetTotal = 0;

	double meanLeavesOutlierTotal = 0;
	double mean2LeavesOutlierTotal = 0;

	double meanLevelsTotal = 0;
	double mean2LevelsTotal = 0;


	trainSet = set;
	for (u_int ins = 0; ins < trainSet->size(); ins++) {
		Instance* inst = trainSet->getInstance(ins);
		inst->setOriginalId(inst->getId());
	}

	if (generateInOriginalDataSet) {

		DataHandler* dataPostBag = new DataHandler(set->getDataSet(),
				set->getClassInd(), false);
		u_int size = set->size();
		for (u_int i = 0; i < size; i++)
				{
			Instance * inst = set->getInstance(i);
			dataPostBag->addInstance(inst);
		}
		handltempAmont = generator->generateOutlierData(dataPostBag,
				histoOutlier, boundsInterTarget);
		delete dataPostBag;
	}
	DForest * res = new DForest(trainSet);
	///float t_test_0_final=clock();//time
	time_t start_final, end_final;
	///double dif_final;
	time(&start_final);

	bool study_node = false;
	if (set->getNbAttributes() <= 3)
		study_node = true;

	for (int i = 0; i < Ltree; i++) {

		time_t start, end;
		///double dif;
		time(&start);
		///float t_test_0=clock();//time

		u_int ** tabAtt = new u_int*[1];

		DataHandler* handltemp = NULL;

		if (!generateInOriginalDataSet) ///Generate outlier data in RSM projected bootstrap; retrieved subspaces in tabAtt
		{

			handltemp = generator->generateProcess(trainSet, tabAtt,
					histoOutlier, boundsInterTarget); //TODO: need optimization of this process

		} else { ///just get boostrap set as outlier data aldready generated in original dataset; bagging is used by default

			handltemp = generator->bagg(handltempAmont);

		}
		if (rsm) {
			listsubspace[i] = (*tabAtt); ///for each tree
		}

		if (log_trees) { ///visualization
			stringstream ss;
			string dbNameTemp = database_name;

			string dir_out = (string) DATA_RESULTS_ROOT + "/" + dbNameTemp
					+ "_outlier";
			string ssName;
			ss << i;
			ssName = dir_out + "/data_outlier_" + ss.str() + ".txt";
			ss.str("");
			mkdir(dir_out.c_str(), 01777);
			ofstream file_dataOutlier(ssName.c_str(), ios::out | ios::app);

			int nb = handltemp->size();

			for (int j = 0; j < nb; j++) {
				Instance * curinst = handltemp->getInstance(j);
				vector<double> temp = curinst->getVectSimple();
				u_int no = 0;
				file_dataOutlier << curinst->getId() << ",";
				for (no = 0; no < temp.size() - 2; no++) {
					file_dataOutlier << temp.at(no) << ",";
				}

				file_dataOutlier << temp.at(no) << "," << curinst->getClass()
						<< endl;
				temp.clear();
			}
			file_dataOutlier.close();
		}

		RndTree* rndtreeInduc = new RndTree(K_RFS, false);
		DTree * tree = rndtreeInduc->growTree(handltemp);

		tree->setConstraintRoot(rejectOutOfBounds, res->getMinMax());

		averageTime += tree->stat_getTimeTrain();

		F_DTree * ftree = new F_DTree(tree);

		if (rsm) {

			vector<int> vals_att(listsubspace[i],listsubspace[i]+nbRSM);

			/*
int nbAttTrainSet = (int) trainSet->getNbAttributes();
int compt = 0;
			vector<int> vals_att(nbRSM+1);
			for (int f = 0; f < nbAttTrainSet; f++) {
				if (Utils::contains(listsubspace[i], f, nbRSM)) {
					///vals[compt] = inst->at(f);
					vals_att[compt] = f;
					compt++;
				}
			}
*/

			ftree->setListDim(vals_att);
			ftree->setListSubspace(listsubspace[i]);

		}

		OCRFStats.nbNodesTotal += ftree->getNbNode();
		OCRFStats.nbLeavesTotal += ftree->getNbLeaves();
		OCRFStats.nbLeavesTargetTotal += ftree->getNbLeavesTarget();
		OCRFStats.nbLeavesOutlierTotal += ftree->getNbLeavesOutlier();
		OCRFStats.nbLevelsTotal += ftree->getNbLevels();

//		cerr<<"\n nbNode "<<i<<":"<<ftree->getNbNode();
//		cerr<<"\n getNbLeaves "<<i<<":"<< ftree->getNbLeaves();
//		cerr<<"\n getNbLeavesTarget "<<i<<":"<< ftree->getNbLeavesTarget();
//		cerr<<"\n getNbLeavesOutlier "<<i<<":"<< ftree->getNbLeavesOutlier();
//		cerr<<"\n getNbLevels "<<i<<":"<< ftree->getNbLevels();
//stats
		meanNodesTotal += ftree->getNbNode() / Ltree;
		mean2NodesTotal += pow(ftree->getNbNode(), 2) / Ltree;

		meanLeavesTotal += ftree->getNbLeaves() / Ltree;
		mean2LeavesTotal += pow(ftree->getNbLeaves(), 2) / Ltree;

		meanLeavesTargetTotal += ftree->getNbLeavesTarget() / Ltree;
		mean2LeavesTargetTotal += pow(ftree->getNbLeavesTarget(), 2) / Ltree;

		meanLeavesOutlierTotal += ftree->getNbLeavesOutlier() / Ltree;
		mean2LeavesOutlierTotal += pow(ftree->getNbLeavesOutlier(), 2) / Ltree;

		meanLevelsTotal += ftree->getNbLevels() / Ltree;
		mean2LevelsTotal += pow(ftree->getNbLevels(), 2) / Ltree;

		if (study_node) {

			int foo = Utils::randInt(1000000);//TODO: backup file association for better log
			stringstream ss1;
			string nom = "nodes_study/log/" + database_name;
			mkdir(nom.c_str(), 01777);
			ss1 << nom << "/" << foo << "_log_nodes_" << i << ".txt";
//string truc="res/log_nodes_"+i+".txt";
			ofstream log_nodes_backup("nodes_study/log/log_backup.txt",
					ios::out | ios::app);
			log_nodes_backup << ss1.str() << endl;
			log_nodes_backup.close();
			tree->computeStructure(tree->getRoot(), ss1.str());
		}

		res->addTree(ftree);

		delete tree;
		delete rndtreeInduc;
		delete[] tabAtt;
		delete handltemp;

		time(&end);
		///dif = difftime (end,start);

		///float t_test_1=clock();//time
		///float time_test=(t_test_1-t_test_0)/(CLOCKS_PER_SEC);

	}

	time(&end_final);
	///dif_final = difftime (end_final,start_final);

	///float t_test_1_final=clock();//time
	///float time_test_final=(t_test_1_final-t_test_0_final)/(CLOCKS_PER_SEC);

	OCRFStats.varNodesTotal = sqrt(mean2NodesTotal - pow(meanNodesTotal, 2));
	OCRFStats.varLeavesTotal = sqrt(mean2LeavesTotal - pow(meanLeavesTotal, 2));
	OCRFStats.varLeavesTargetTotal = sqrt(
			mean2LeavesTargetTotal - pow(meanLeavesTargetTotal, 2));
	OCRFStats.varLeavesOutlierTotal = sqrt(
			mean2LeavesOutlierTotal - pow(meanLeavesOutlierTotal, 2));
	OCRFStats.varLevelsTotal = sqrt(mean2LevelsTotal - pow(meanLevelsTotal, 2));

	averageTime = averageTime / Ltree;

	_time = (((double) clock()) / CLOCKS_PER_SEC) - _time;
	res->stat_setTimeTrain(_time);
	res->stat_setMeanDTTimeTrain(averageTime);

	string tmp = Utils::to_string(_time);
	tmp += " seconds";
	if (disp)
		Utils::print(tmp);
	return res;

}

u_int ** OCForest::getlistsubspace() {
	return listsubspace;
}

void OCForest::displayListSubspace() {
	for (int i = 0; i < Ltree; i++) {
		cout << "Sub-space for tree l_" << i << " : ";
		for (int j = 0; j < nbRSM; j++) {
			cout << listsubspace[i][j] << " ";
		}
		cout << "\n";
	}
}

