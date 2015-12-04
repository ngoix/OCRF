#include "../include/utils.h"
#include "../include/arff.h"
#include "../include/outliergenerator.h"
#include "../include/instance.h"
#include "../include/ocforest.h"

#include <fstream>
#include <string>
#include <math.h>
#include <sstream>
#include <iostream>
#include <cstdlib>

using namespace std;

void show_help() {
	cout << "###################################" << endl;
	cout << "\nUsage:" << endl;
	cout << "./ocrf -db database_name [-options]" << endl;
	cout << "\n\nOptions:" << endl;

	cout << "-db \t database name" << endl;
	cout << "-dimension: \tnumber of dimensions" << endl;
	cout << "-strat \t Folder hierarchy for learning process : strat_i/fold_j/learning_set.arff|test_set.arff" << endl;
	cout << "-fold \t Folder hierarchy for learning process : strat_i/fold_j/learning_set.arff|test_set.arff" << endl;
	cout <<"-path_learning"<<endl;
	cout <<"-path_test\n"<<endl;

	cout << "-method \tlearning method for OCRF; "
					"0 (default) for outlier generation in bagging+rsm set (projected bootstrap set);"
					"1 for outlier generation before the induction of the forest (i.e. befor bagging in Forest-RI), no RSM is applied;"
					"default with bagging+rsm\n" << endl;

	cout
			<< "-beta \t Factor controlling the number of outlier data generated according to the number of target data (e.g. 10 for 10x number of target data)\n"
			<< endl;
	cout << "-alpha \t Factor controlling the extension of the outlier domain used for outlier generation according to the volume of the hyperbox surrounding the target data\n" << endl;
	cout << "-rejectOutOfBounds \t Data outside target bounds are considered as outlier data" << endl;
	cout << "-optimize \t 0 for uniform distribution for outlier data; 1 for biased roulette-wheel distribution" << endl;

	cout << "-krsm \t Number of dimensions for the Random Subscpace Method (RSM)" << endl;
	cout << "-krfs \t Number of features randomly selected at each node during the induction of the tree" << endl;
 	cout << "###################################" << endl;
}


int main(int argc, char **argv) {

	srand(time(nullptr));

	string db; ///<Name of the database
	int nbAttr; ///<Number of attributes from the original feature space
	int iterMyStrat = 0; //strat number;
	int iterMyFold = 0; //number of folds

	bool param_rejectOutOfBounds = false; ///< true to reject instantly outliers data that are outside the domain of the target data (false is recommended); if true ensures less flexible out-of-class generalization;
	bool optimize_gen = true; ///< true for optimization of outliers generation in each subspace
///bool optimize_amont=false;///< true for outliers generation from the original learning set
	int param_method = 0;///< 0:bag+rsm then outlier generation; 1:outlier genertaion in original set then bagging (no rsm)
	int rejectOutOfBounds = 0; //
	bool param_generateInOriginalDataSet = false; //classical approach
	int optimize_ok = -1;
	int param_beta=10;
	double param_alpha = 1.2;//TODO:beware, not same notation in OCRF PR article in which $\gamma=1+\alpha$

	int nbRSM_selected = -1;//default
	int nbRFS_selected = -1;//default
	int nbRFS = 0;
	int nbRSM = 0; //(int)nbCarac/2;
	bool bagging = true; //true;
	bool rsm = true;//modified hereafter, depending on outlier generation method (original set,bagging,bagging+rsm)
	int nbTree = 100;
	int nbBagging = 100; //percentage for bagging

	bool param_studyNodes=true;


	string param_alpha_str = "";
	string param_beta_str = "";

	string param_nbTree_str = "";
	string param_studyNodes_str = "";

	string path_learning_str = "";
	string path_test_str = "";


	stringstream msg_param;

	if (argc == 1) { //show help
		show_help();
		exit(0);
	}

	for (int i = 1; i < argc; i += 2) {

		string temp = (string) argv[i];

		if (temp.compare("-h") == 0 || temp.compare("--help") == 0) {

			show_help();
			exit(0);
		}

///TODO:create class for parameters sharing
		/**Dataset parameters*/
		if (temp.compare("-db") == 0) {
			db = argv[i + 1];
			msg_param << "-db:" << db << endl;
		}
		if (temp.compare("-dimension") == 0) {///TODO:not required
			nbAttr = atoi(argv[i + 1]);
			msg_param << "-dimension:" << nbAttr << endl;
		}

		/**Learning process parameters*/
		if (temp.compare("-strat") == 0) {
			iterMyStrat = atoi(argv[i + 1]);
			msg_param << "-strat:" << iterMyStrat << endl;
		}
		if (temp.compare("-fold") == 0) {
			iterMyFold = atoi(argv[i + 1]);
			msg_param << "-fold:" << iterMyFold << endl;
		}

		if (temp.compare("-path_learning") == 0) {
			path_learning_str = argv[i + 1];
			msg_param << "-path_learning:" << path_learning_str << endl;
		}
		if (temp.compare("-path_test") == 0) {
			path_test_str = argv[i + 1];
			msg_param << "-path_test:" << path_test_str << endl;
		}

		/**Parameters for OCRF*/
		if (temp.compare("-beta") == 0) {
			param_beta_str = argv[i + 1];
			param_beta = atoi(argv[i + 1]);
			msg_param << "-beta:" << param_beta << endl;
		}
		if (temp.compare("-alpha") == 0) {
			param_alpha_str = argv[i + 1];
			msg_param << "-alpha:" << param_alpha_str << endl;
		}
		if (temp.compare("-krsm") == 0) {
			nbRSM_selected = atoi(argv[i + 1]);
			msg_param << "-krsm:" << nbRSM_selected << endl;
		}
		if (temp.compare("-krfs") == 0) {
			nbRFS_selected = atoi(argv[i + 1]);
			msg_param << "-krfs:" << nbRFS_selected << endl;
		}
		if (temp.compare("-nbTree") == 0) {
			param_nbTree_str = argv[i + 1];
			msg_param << "-nbTree:" << param_nbTree_str << endl;
		}

		/**Optimization parameters*/
		if (temp.compare("-method") == 0) {
			param_method = atoi(argv[i + 1]);
			msg_param << "-method:" << param_method << endl;
		}
		if (temp.compare("-rejectOutOfBounds") == 0) {
			rejectOutOfBounds = atoi(argv[i + 1]);
			msg_param << "-rejectOutOfBounds:" << rejectOutOfBounds << endl;
		}
		if (temp.compare("-optimize") == 0) {
			optimize_ok = atoi(argv[i + 1]);
			msg_param << "-optimize:" << optimize_ok << endl;
		}

		if (temp.compare("-studyNodes") == 0) {
			param_studyNodes = atoi(argv[i + 1]);
			msg_param << "-studyNodes:" << param_studyNodes << endl;
		}

	}

	cerr << "parameters:\n" << msg_param.str() << endl;

	if (rejectOutOfBounds == 1)
		param_rejectOutOfBounds = true;

	if (optimize_ok == 0)
		optimize_gen = false;
	if (optimize_ok == 1)
		optimize_gen = true; //TODO:1=biased roulette-wheel;0:uniform;2=calculate histograms in global LS
///if(optimize_ok==2) optimize_amont=true;





		if (fromString(param_alpha_str, param_alpha)>0) {
			Utils::flog << "##########error casting fromString:" << __LINE__
					<< endl;
		}

		if (fromString(param_beta_str, param_beta)>0) {
			Utils::flog << "##########error casting fromString:" << __LINE__
					<< endl;
		}
		if (fromString(param_nbTree_str, nbTree)>0) {
			Utils::flog << "##########error casting fromString:" << __LINE__
					<< endl;
		}

		//debugv(nbTree_str,nbTree);



		if (param_method == 0) { //bagging+rsm
//	cerr<<"meth:bag rsm"<<endl;cin.get();
			param_generateInOriginalDataSet = false; //method ede ref
			bagging = true; //true (default);
			rsm = true;//true (default)
		} else if (param_method == 1) { //outlier generation in global learning set, without RSM, standard induction (bagging + RFS)

			param_generateInOriginalDataSet = true;
			bagging = true;
			rsm = false;

		}

		string res_root = (string) DATA_RESULTS_ROOT;
		string sFoldRes = res_root + "/" + db;
		string sFoldModel=sFoldRes + "/model";
		string model_filename=sFoldModel + "/model.forest";
		mkdir(res_root.c_str(), 01777);
		mkdir(sFoldRes.c_str(), 01777);
		mkdir(sFoldModel.c_str(), 01777);


//string sout="./resultats/res/"+db+"_outlier";
//	mkdir(sout.c_str(),01777);

//	mkdir("./resultats/res/outlier",01777);

		string dataAppPath = ""; ///(string)DATA_ROOT+"/"+db+"/strats/strat_"+s1+"/fold_"+s2+"/app.arff";
		if (path_learning_str.compare("") != 0) {
			dataAppPath = path_learning_str;
		}

		DataHandler * handlerBaseOrig = Arff::load(dataAppPath.c_str()); //base d'apprentissage

		int nbFeatures = handlerBaseOrig->getNbAttributes();
		int nbTargetApp = handlerBaseOrig->getDistrib(TARGET); //nbTarget disponible en Apprentissage
		int nbOutlierApp = handlerBaseOrig->getDistrib(OUTLIER); //nbOutlier=0 dans noptre problematique oneclass

		if (nbRSM_selected != -1) {
			///manual values
			nbRSM = nbRSM_selected;//TODO:raise exception if not valid as manual: user has to provide valid value (lower than dimension)
		} else {

			nbRSM = nbFeatures; ///without RSM

			if (rsm) { ///with RSM activated
					   ///Other possible values for nbRSM (int)sqrt(nbCarac) or nbCarac/2 [HoRSM98]
				if (nbFeatures > 10) {
					nbRSM = 10;		//TODO:default value hardcoded; use default configuration file
				} else {
					nbRSM = nbFeatures;
				}
			}

		}

		if (nbRFS_selected != -1) {		///manual value
			nbRFS = (int) ((double) nbRFS_selected / 100.0 * nbRSM);
		} else {		///automatic value
			nbRFS = (int) sqrt(nbRSM);		///Standard value in literature
		}

		unsigned long int param_nbOutlierData = (int) nbTargetApp * param_beta;		//beta=1

		vector<vector<int> > histoOutlier;
		vector<vector<double> > boundsInterTarget;

		if (optimize_gen) {
			OutlierGenerator outg;
			outg.rouletteWheel(handlerBaseOrig, histoOutlier, boundsInterTarget,
					param_alpha);
		} else {
			///other optimization; other method is uniform
		}

		bool log_trees = false;		///TODO

		/*
		 if(iterStrat==0 && iterFold==0) {
		 log_trees=true;
		 }
		 log_trees=false;
		 */

		params_forest pForest;
		params_oneclass pOneclass;

		pForest.param_nbTrees=nbTree;
		pForest.param_KRFS=nbRFS;
		pForest.param_percentBagging=nbBagging;
		pForest.param_KRSM=nbRSM;
		pForest.param_bagging=bagging;
		pForest.param_rsm=rsm;
		pForest.param_logTrees=log_trees;
		pForest.param_studyNodes=param_studyNodes;

		pOneclass.param_generateInOriginalDataSet=param_generateInOriginalDataSet;
		pOneclass.param_nbOutlierData=param_nbOutlierData;
		pOneclass.param_alpha=param_alpha;
		pOneclass.param_rejectOutOfBounds=param_rejectOutOfBounds;
		pOneclass.param_optimize=optimize_gen;
		pOneclass.param_method=param_method;
		pOneclass.param_histogramOutlierData=histoOutlier;
		pOneclass.param_boundsInterTargetData=boundsInterTarget;


///OCFOREST INDUCTION
		OCForest * ocrf = new OCForest(pForest,pOneclass,db);
		DForest * forest = ocrf->growForest(handlerBaseOrig);
///END OCFOREST INDUCTION

		///ERROR ESTIMATE FOR TARGET CLASS - OUT-OF-BAG
		Result* resOOB=forest->getOOBOCestimates(ocrf->getlistsubspace(),rsm,nbRSM);
		cout<<"OOB-"<<resOOB->toString()<<endl;

		///LOG RESULTS INIT
		stringstream varStream;
		string refStr;
		varStream << param_method;
		refStr = varStream.str();
		varStream.str("");
		string rejectOutOfBounds_str;
		varStream << rejectOutOfBounds;
		rejectOutOfBounds_str = varStream.str();
		varStream.str("");
		string optimize_okStr;
		varStream << optimize_ok;
		optimize_okStr = varStream.str();
		varStream.str("");
		varStream.clear();

		string krsmStr;
		varStream << nbRSM;
		krsmStr = varStream.str();
		varStream.str("");
		string param_rsm = "_krsm_" + krsmStr;
		string param_method_str = "_method_" + refStr;
		string param_reject_root = "_rejetbornes_" + rejectOutOfBounds_str;
		string params_str = db + param_method_str + param_rsm + "_beta_" + param_beta_str
				+ "_alpha_" +param_alpha_str + param_reject_root;

		string logResMinMaxOutlier = sFoldRes + "/results_" + params_str
				+ "_minmaxoutlier.txt";
		string logResMinMaxTarget = sFoldRes + "/results_" + params_str
				+ "_minmaxtarget.txt";
		string logRes = sFoldRes + "/results_" + params_str + ".txt";
		string logResDecision = sFoldRes + "/results_decision_" + params_str
				+ ".txt";

		////END INIT

		///TESTING THE CLASSIFIER
		string dataTestPath = "";///(string)DATA_ROOT+"/"+db+"/strats/strat_"+s1+"/fold_"+s2+"/test.arff";
		if (path_test_str.compare("") != 0) {
			dataTestPath = path_test_str;
		}
		DataHandler * handlerTest = Arff::load(dataTestPath.c_str());//base de test

		int nbTargetTest = handlerTest->getDistrib(TARGET);
		int nbOutlierTest = handlerTest->getDistrib(OUTLIER);

		Result * res = forest->testOC(handlerTest,
				ocrf->getlistsubspace(), rsm, nbRSM, logResDecision);

		u_int ** confmat = res->getconfmat();
		cout<<res->toString()<<endl;

		///END TESTING CLASSIFIER

		///LOG RESULTS INIT
		ofstream file_minmaxoutlier(logResMinMaxOutlier.c_str(),
				ios::out | ios::app);
		ofstream file_minmaxtarget(logResMinMaxTarget.c_str(),
				ios::out | ios::app);
		ofstream file_results(logRes.c_str(), ios::out | ios::app);
		ofstream file_resultsDecision(logResDecision.c_str(),
				ios::out | ios::app);

		file_resultsDecision << "\n" + path_test_str;		//<<endl;
		file_resultsDecision.close();

		stringstream st;
		st << "#M#" << nbFeatures << "#nbTree#" << nbTree;

		st << "#alpha#" << param_alpha << "#beta#" << param_beta
				<< "#nbOutlierGen#" << param_nbOutlierData;

		st << "#nbTargetApp#" << nbTargetApp << "#nbOutlierApp#" << nbOutlierApp
				<< "#nbTargetTest#" << nbTargetTest << "#nbOutlierTest#"
				<< nbOutlierTest;
		st << "#bagging#" << bagging << "#RSM#" << rsm << "#K_RSM#" << nbRSM
				<< "#K_RFS#" << nbRFS;

		st << "#time_train#" << (double) forest->stat_getMeanDTTimeTrain();
		st << "#nbNodes#" << ocrf->getNbNodes() << "#VarnbNodesMean#"
				<< (double) ocrf->getVarNbNodes();
		st << "#nbLevels#" << (double) ocrf->getNbLevels()
				<< "#VarnbLevelsMean#"
				<< (double) ocrf->getVarNbLevels();
		st << "#nbLeaves#" << (double) ocrf->getNbLeaves()
				<< "#VarnbLeavesMean#"
				<< (double) ocrf->getVarNbLeaves();
		st << "#nbLeavesTarget#" << (double) ocrf->getNbLeavesTarget()
				<< "#VarnbLeavesTargetMean#"
				<< (double) ocrf->getVarNbLeavesTarget();
		st << "#nbLeavesOutlier#" << (double) ocrf->getNbLeavesOutlier()
				<< "#VarnbLeavesOutlierMean#"
				<< (double) ocrf->getVarNbLeavesOutlier();

		st << "#reco_weighted#" << res->getRecoPond() << "#reco#"
				<< res->getRecoRate();

		st << "#TP#" << confmat[TARGET][TARGET]
		   << "#FN#" << confmat[TARGET][OUTLIER]
				                   << "#FP#"
				<< confmat[OUTLIER][TARGET]
				                    << "#TN#"
				<< confmat[OUTLIER][OUTLIER] << endl;

		file_results << st.str();
		file_results.close();

		cout << st.str() << endl;
		Utils::flog << st.str();

		bool DEBUG_MINMAX = false;

		if (DEBUG_MINMAX) {
			double** minmax_outlier = handlerTest->computeMinMaxClass(OUTLIER);
			double** minmax_target = handlerTest->computeMinMaxClass(TARGET);

			for (int i = 0; i < nbFeatures; i++) {
				file_minmaxtarget << minmax_target[0][i] << "\t";
				file_minmaxoutlier << minmax_outlier[0][i] << "\t";
			}
			for (int i = 0; i < nbFeatures; i++) {
				file_minmaxtarget << minmax_target[1][i] << "\t";
				file_minmaxoutlier << minmax_outlier[1][i] << "\t";
			}

			file_minmaxtarget << "\n";
			file_minmaxoutlier << "\n";

			delete[] minmax_outlier[0];
			delete[] minmax_outlier[1];
			delete[] minmax_outlier;
			delete[] minmax_target[0];
			delete[] minmax_target[1];
			delete[] minmax_target;

		}
		file_minmaxtarget.close();
		file_minmaxoutlier.close();
///END LOG

		delete handlerBaseOrig;
		delete handlerTest;

		delete ocrf;
		delete forest;
		delete res;


}

