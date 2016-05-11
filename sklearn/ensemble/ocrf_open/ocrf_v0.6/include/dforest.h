#ifndef DFOREST_H_
#define DFOREST_H_

#include "../include/dtree.h"
#include "../include/f_dtree.h"
#include "../include/datahandler.h"
#include "../include/arff.h"
#include "../include/utils.h"


/********************************************************************
*
*   Name: class _DFStats
*
*   Description:  private class for internal use only
*
*********************************************************************/
class _DFStats
{
	friend class DForest;

	private:
		double		timeTrain;
		double		meanDTTimeTrain;
		const char * 	loadingBar;

				_DFStats()
				{
					timeTrain = 0.0;
					meanDTTimeTrain = 0.0;
					loadingBar = "|/-\\";
				}
};

/********************************************************************
*
*   Name:           class DForest
*
*   Description:  Allow to represent and to handle a Decision Forest.
* 		contains a vector of Decision Tree (F_DTree object), and a pointer
* 		to a DataHandler object
*
*********************************************************************/
class DForest
{

	private :



		vector<F_DTree *>	forest;

		DataHandler * 	trainSet;
		_DFStats 		stats;

double ** minmax;
		bool rsm_predict;/**< Default value is false*/

	public :
					DForest():rsm_predict(false){
						minmax=nullptr;
						trainSet=nullptr;
					}
					DForest(DataHandler * set);
					DForest(string filename);
		virtual			~DForest();

		void			addTree(F_DTree * tree);
		void			removeTree(u_int ind);

        void             affnbNode();
		DataHandler *		getDataHandler();
		u_int			getNbTrees();
		F_DTree *		getTree(u_int ind);
		Result *		getOOBestimates();
		Result *		getOOBOCestimates(u_int ** listsubspace,bool rsmOk,int nbRSM);
		u_int			predict(Instance * inst);
		u_int			predictOC(Instance * inst, u_int ** listsubspace, bool rsmOk,int nbRSM,string fich);
		Result *		test(DataHandler * testSet);
		Result *        testOC(DataHandler * testSet, u_int ** listsubspace,bool rsmOk,int nbRSM,string fich);
		double			margin(Instance * inst);
		double			correlation(DataHandler * testset);
		double			correlation();
		double 			strength(DataHandler * testset);
		double 			strength();

		double			stat_getTimeTrain();
		double		stat_getMeanDTTimeTrain();
		char    		stat_getLoadingBar(u_int n);
		void 			stat_setTimeTrain(double time);
		void			stat_setMeanDTTimeTrain(double time);
		void			save(string filename);

		string 			toString();
		string 			statsToString();

		//stats++
		double 			getMeanNode();

//		double** getMinMaxVal();
		double** getMinMax(){return minmax;}

		int saveFile(string dest);
		int writeFileForest(string dest_tree,Node * starting_node);
		//#niveaux/profnodeur
		//#leaf target
		//#leaf outlier
};


#endif /*DFOREST_H_*/
