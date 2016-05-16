#include "../include/outliergenerator.h"
#include "../include/bagging.h"
#include "../include/utils.h"

OutlierGenerator::OutlierGenerator() {

	vectorsubspace = nullptr;
	/*
	 optimize_gen=optimize_gen_temp;
	 amontGen=amontGenTemp;
	 bag = baggy;
	 rsm = randsm;
	 bagsize = bsize;
	 nbsubspace = nbsub;
	 nbOutlier = nbO;
	 taillezone = tz;

	 vecteursubspace = new u_int[nbsub];


	 srand((unsigned)time(0));//initialisation de rand

	 if(bsize<=0 || bsize>100) bagsize = 100;
	 */

}

OutlierGenerator::OutlierGenerator(bool amontGenTemp, bool baggy, bool randsm,
		double bsize, int nbsub, int nbO, double param_alpha,
		bool optimize_gen_temp) {

	optimize_gen = optimize_gen_temp;
	amontGen = amontGenTemp;
	bag = baggy;
	rsm = randsm;
	bagsize = bsize;
	nbsubspace = nbsub;
	nbOutlier = nbO;
	alpha_mod = param_alpha-1;
	vectorsubspace = new int[nbsub];

	srand((unsigned) time(0));///TODO:remove from generator

	if (bsize <= 0 || bsize > 100)
		bagsize = 100;

}

OutlierGenerator::~OutlierGenerator() {

	if (vectorsubspace != nullptr) {
		delete[] vectorsubspace;
	}

	for (u_int i = 0; i < data.size(); i++)
		data.erase(data.begin() + i);
	data.clear();

	listRSM.clear();
	//delete newData;
}

DataHandler * OutlierGenerator::generateProcess(DataHandler * base,
		u_int **tabRSM, const vector<vector<int> >& histoOutlierTemp,
		const vector<vector<double> >& boundsInterTargetTemp) {

	histoOutlier = histoOutlierTemp;
	boundsInterTarget = boundsInterTargetTemp;

	DataHandler * dataPostBag = nullptr;
	DataHandler * dataOutliergen = nullptr;

	if (bag) {//bagging
		dataPostBag = bagg(base);
	} else {

		if (!amontGen) {//no bagging
			dataPostBag = new DataHandler(base->getDataSet(),
					base->getClassInd(), true);

			u_int size = base->size();

			for (u_int i = 0; i < size; i++) {
				dataPostBag->addInstance(base->getInstance(i));
			}

		}
	}

	DataHandler * datafinal = dataPostBag;
	if (rsm) {
		datafinal = randomsubspacemethod(dataPostBag, tabRSM);
	} else {

	}

	/*
	 generation of outlier in RSM projected bootstrap sample
	 */
	if (!amontGen) {
		dataOutliergen = generateOutlierData(datafinal);

	}

	delete datafinal;
if(rsm){
	delete dataPostBag;
}

	return dataOutliergen;
}

DataHandler * OutlierGenerator::generateOutlierData(DataHandler * base,
		const vector<vector<int> >& histoOutlierTemp,
		const vector<vector<double> >& boundsInterTargetTemp) {

	histoOutlier = histoOutlierTemp;
	boundsInterTarget = boundsInterTargetTemp;

	return generateOutlierData(base);
}

DataHandler * OutlierGenerator::generateOutlierData(DataHandler * base) {

	data = base->getDataSet()->getData();

	u_int nbCarac = base->getNbAttributes();


	DataSet * newData = new DataSet();

	for (u_int i = 0; i < base->size(); i++) {

		Instance* inst = base->getInstance(i);    //TODO: save Id
		u_int origId = inst->getOriginalId();

		vector<double> vect = inst->getVectSimple();
		for (int w = 0; w < base->getWeight(inst->getId()); w++) {
			newData->addInstance(&vect, origId);
		}
		vect.clear();
	}

	u_int id = base->size();

	time_t start, end;
///double dif;
	time(&start);
///float t_test_0_gen=clock();//time

	int nbNull = 0;

	double **minmaxval = base->computeMinMax();

	for (int i = 0; i < nbOutlier; i++) {

		Instance * curinst = genereInstance((u_int) (id + i),
				base->getNbAttributes(), minmaxval, alpha_mod,
				base->getClassInd());

//Instance * curinst = genereInstanceBoule((u_int)(id+i),base->getNbAttributes(),minmaxval,taillezone,base->getClassInd());

		if (curinst != NULL) {
			newData->addInstance(curinst->getVect());
		} else {    //continue
			nbNull++;

		}
//        handler->addInstance(curinst);

		delete curinst;
	}

	time(&end);
///dif = difftime(end,start);

//ofstream time_calc("./resultats/res/time.txt",ios::out|ios::app);

///float t_test_1_gen=clock();//time
///float time_test_gen=(t_test_1_gen-t_test_0_gen)/(CLOCKS_PER_SEC);

//time_calc<<nbOutlier<<"\t"<<dif<<"\t"<<time_test_gen<<endl;
//time_calc.close();

	delete[] minmaxval[0];
	delete[] minmaxval[1];
	delete[] minmaxval;

	//************************************************************************************************
	// attributes added
	for (u_int c = 0; c < nbCarac; c++) {
		newData->addAttribute(Utils::to_string((int) c), NUMERIC);
//        handler->getDataSet()->addAttribute(Utils::to_string(c),NUMERIC);
	}
	vector<string> modal(2);
	modal[TARGET] = "target";
	modal[OUTLIER] = "outlier";
	newData->addAttribute("class", NOMINAL, &modal);
//handler->getDataSet()->addAttribute("class",NOMINAL,&modal);

	//************************************************************************************************
	// init of the handler

	DataHandler * handler = new DataHandler(newData, base->getClassInd(), true);



//delete newData;
	return handler;
//    return new DataHandler(newData,base->getClassInd(),true);
}

/**
 * @brief use regular gaussian distribution to produce uniform distribution in a ball
 * TODO: integrate other version of OCRF
 */
Instance * OutlierGenerator::genereInstanceBoule(u_int id, u_int nbCarac,
		double ** minmaxval, double taillezone, u_int placeCarac) {

	throw "Deprecated !";

	return nullptr;
	/*
	 Instance * inst;
	 vector<double> vals(nbCarac+1);
	 vector<double> valsOutlier(nbCarac+1);
	 double largeurborne;
	 double curval;
	 double ranval;

	 ofstream fichGauss("./resultats/gauss.txt",ios::out|ios::app);
	 ofstream fichBoule("./resultats/boule.txt",ios::out|ios::app);
	 double rayon =1.0+taillezone;

	 double norme_vals=0;
	 for(u_int i=0;i<nbCarac;i++){
	 vals[i]=Utils::randgauss(0,1);
	 norme_vals+=vals[i]*vals[i];
	 fichGauss<<vals[i]<<"\t";
	 }
	 fichGauss<<"\n";
	 vals[nbCarac]=1;//outlier
	 valsOutlier[nbCarac]=1;//outlier

	 norme_vals=sqrt(norme_vals);

	 double z=alglib::chisquaredistribution((double)nbCarac, norme_vals*norme_vals);
	 double r1=pow(z,(double)1.0/nbCarac);
	 //cout<<"val r1##################:"<<r1<<endl;
	 for(u_int i=0;i<nbCarac;i++){

	 //double val=Utils::randBoule(,(int)nbCarac);

	 valsOutlier[i]=vals[i]/norme_vals*r1*rayon;
	 //cout<<"boule:"<<val<<endl;

	 fichBoule<<valsOutlier[i]<<"\t";
	 }
	 fichBoule<<"\n";


	 inst = new Instance(id,&valsOutlier);
	 vals.clear();
	 //distribution gaussienne
	 //somme des carres
	 //fonction de repartition de la loi normale
	 //normalisation


	 return inst;
	 */

}


DataHandler * OutlierGenerator::bagg(DataHandler * base) {

/*
	DataHandler * handlerfinal = new DataHandler(base->getDataSet(),
			base->getClassInd(), false);    //fully handled false

	u_int s = base->size();
	u_int size = (u_int) (((double) (s * bagsize)) / 100.0);

	for (u_int i = 0; i < size; i++) {

		int ind = Utils::randInt(s);
		Instance * inst = base->getInstance(ind);

		handlerfinal->addInstance(inst);
	}
	return handlerfinal;
*/

	Bagging bag_gen(bagsize);
	return bag_gen.generateBootstrap(base);


}

DataHandler * OutlierGenerator::randomsubspacemethod(DataHandler * handler,
		u_int ** tabRSM) {

	(*tabRSM) = Utils::samplingWithoutReplacement(nbsubspace, handler->getNbAttributes());

	DataSet * newDataRSM = new DataSet();
	int ind;
//	vector<double> vals(nbsubspace + 1);

	vector<int> listRSM_temp;
	int nbAtt=handler->getNbAttributes();
	for (u_int c = 0; c < (u_int) nbAtt; c++) {

		if (Utils::contains((*tabRSM), c, nbsubspace)) {
			listRSM_temp.push_back(c);
		}

	}
	listRSM = listRSM_temp;

	for (u_int i = 0; i < handler->size(); i++) {
		ind = 0;
		vector<double> vals(nbsubspace + 1);

		Instance * inst = handler->getInstance(i);

		for (u_int c = 0; c < handler->getNbAttributes(); c++) {
			if (Utils::contains((*tabRSM), c, nbsubspace)) {
				vals[ind] = inst->at(c);

				ind++;
			}
		}

		vals[nbsubspace] = inst->getClass();

		for(int w=0;w<(int)handler->getWeight(inst->getOriginalId());++w){
			newDataRSM->addInstance(&vals, inst->getOriginalId());
		}

	}

	for (int c = 0; c < nbsubspace; c++) {
		newDataRSM->addAttribute(Utils::to_string(c), NUMERIC);
	}

	vector<string> modal(2);
	modal[TARGET] = "target";
	modal[OUTLIER] = "outlier";
	newDataRSM->addAttribute("class", NOMINAL, &modal);

	DataHandler * handlerfinal = new DataHandler(newDataRSM, nbsubspace, true);

	return handlerfinal;
}

double OutlierGenerator::parzenKernelValue(double x, int d) {

	throw "Deprecated !";

//double h=1.0;//window

	double somme = 0;

///double z=0;//au lieu de sqrt(2*PI);

	int nb = data.size();

	double h = 1.0 / sqrt(nb);    //window

//double maxi=0.000000000000000000000001;
//double mini=0.000000000000000000000001;
////cerr<<"taille data:"<<nb<<endl;
//for(int i=0;i<nb;i++){
//double x_i=fabs(data[i].at(d));
//if(x_i>maxi) maxi=x_i;
////if(x_i<mini) mini=x_i;
//
//}

//for(int i=0;i<nb;i++){
//double x_i=data[i].at(d);
////z+=1.0*exp(-pow(((x_i-mini)/((maxi-mini)*h)),2)/2);
////z+=1.0*exp(-pow(((x_i)/((maxi)*h)),2)/2);
//z+=1.0*exp(-pow(((x_i)/((maxi)*h)),2)/2);
////cerr<<"x_i="<<x_i<<" z="<<z<<" maxi="<<maxi<<";";
//}
//
//cerr<<"\nz="<<z<<" | classique="<<sqrt(2*PI)<<endl;

	for (int i = 0; i < nb; i++) {
		double x_i = data[i].at(d);
//somme+=1.0/(z)*exp(-pow((x-x_i)/((maxi-mini)*h),2)/2);
//somme+=1.0/(z*h)*exp(-pow((x-x_i)/((maxi)*h),2)/2);
		somme += 1.0 / (sqrt(2 * PI) * h) * exp(-pow((x - x_i) / h, 2) / 2);
	}

	return somme / nb;

}

Instance * OutlierGenerator::optimizeGeneration(u_int placeCarac, u_int id,
		u_int nbCarac, double taillezone, double ** minmaxval) {

	vector<double> vals(nbCarac + 1);

	Instance * inst = NULL;

	double largeurborne = 0;    //mesure de l'intervalle
	double curval = 0;    //valeur de l'attribut
	double ranval = 0;    //random generator

	bool proba_estim = false;

	bool alea_biais = true;    //roue de la fortune biais�e
	bool roulette_wheel = true;
///bool alea_derivative=false;//generation proche des donn�es

	double seuil = 0.5;    //seuil sur proba_estim
	int iterMax = 10;    //nb iter max sur proba_estim
	int iter = 0;
///double eps=0.01;//precision

	int nb_partition = histoOutlier.at(0).size(); //50 (default);sqrt(nbData);10:valeur conseillee par aggarwal et al. [];\sqrt{n} a voir

//ranval = (double)(rand()%11)/10;
	ranval = ((double) Utils::randInt(100000)) / 100000.0;    //fraction de 1

	iter = 0;

	if (placeCarac != 0) {
		for (u_int c = 0; c < nbCarac; c++) {

///mapping with RSM
			int cRSM = c;
			if (rsm) {
				cRSM = listRSM[c];
			}

			//largeurborne = minmaxval[1][c] - minmaxval[0][c];
			largeurborne = boundsInterTarget.at(cRSM).at(nb_partition)
					- boundsInterTarget.at(cRSM).at(0);

			iter = 0;

			if (proba_estim) { //estimation de proba par sommation de gaussienne

				do {

//curval = ranval*(largeurborne+2*taillezone*largeurborne) + minmaxval[0][c] - taillezone*largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur
					curval = ranval * (1 + taillezone) * largeurborne
							+ minmaxval[0][c] - taillezone / 2 * largeurborne; //on genere dans l'enveloppe pour la dimension c une valeur
					//            curval = ranval;//*(largeurborne+2*taillezone*largeurborne);// + minmaxval[0][c] - taillezone*largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur aleatoire

					double proba = parzenKernelValue(curval, c);
					if (proba < seuil) {
						break;
					}
					iter++;

					ranval = ((double) Utils::randInt(100000)) / 100000.0;//fraction de 1

				} while (iter < iterMax);

				if (iter >= iterMax) {
//	curval = ranval*(largeurborne+2*taillezone*largeurborne) + minmaxval[0][c] - taillezone*largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur aleatoire
					curval = ranval * (1 + taillezone) * largeurborne
							+ minmaxval[0][c] - taillezone / 2 * largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur
					//cerr<<curval<<"/d="<<c<<endl;

				}

				vals[c] = curval;

			}

			if (alea_biais) {	//roue de la fortune biais�e

				int somme_effectifs = 0;
				for (int i = 0; i < nb_partition; i++) {
					somme_effectifs += histoOutlier.at(cRSM).at(i);
				}

//tirage aleatoire dans [0;somme_effectifs]
				ranval = ((double) Utils::randInt(100000)) / 100000.0;//fraction de 1
				int eff_alea = (int) ((float) ranval * somme_effectifs);//valeur aleatoire de la partition de generation outlier
//reperage dans effectifs l'intervalle correspondant

				if (roulette_wheel) {	//roulette-wheel selection

					int somme_temp = 0;
					int trouve = 0;
					int iter_int = 0;

					for (int j = 0; j < nb_partition; j++) {
//inversion
						somme_temp += histoOutlier.at(cRSM).at(j);
//somme_temp+=effectifs[j];

						if (somme_temp >= eff_alea) {
							trouve = 1;
							iter_int = j;

							break;
						}
//iter_int++;
					}

					if (trouve == 0) {

//curval = ranval*(1+taillezone)*largeurborne + minmaxval[0][c] - taillezone/2*largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur; ce qui n'arrive pas

//curval = boundsInterTarget.at(cRSM).at(0) +ranval*largeurborne;//generate outlier data when no location found
						curval = 0;	//TODO: place target value at origin of this axe if no location found

						/*
						 ofstream fich_random("random_outlier",ios::out|ios::app);
						 fich_random<<curval<<endl;
						 fich_random.close();
						 return NULL;
						 */

					} else {	//generation d'une valeur dans inter

//if(iter_int>0){
						ranval = ((double) Utils::randInt(100000)) / 100000.0;//fraction de 1
//double diff=inter[iter_int+1]-inter[iter_int];
						double diff = boundsInterTarget.at(cRSM).at(
								iter_int + 1)
								- boundsInterTarget.at(cRSM).at(iter_int);

						vals[c] = boundsInterTarget.at(cRSM).at(iter_int)
								+ ranval * diff;//inter[iter_int+1])/2;//a changer pr un alea
//vals[c]=inter[iter_int]+ranval*diff;//inter[iter_int+1])/2;//a changer pr un alea
					}

				}

			}

		}
	} else {
		cerr << "place  at beginning not yet implemented" << endl;
	}

	vals[placeCarac] = OUTLIER;	//0:outlier;1:target

	inst = new Instance(id, &vals);

	return inst;
}

Instance * OutlierGenerator::genereInstance(u_int id, u_int nbCarac,
		double ** minmaxval, double taillezone, u_int placeCarac) {
//Instance * OutlierGenerator::genereInstanceBoule(u_int id, u_int nbCarac, double ** minmaxval, double taillezone, u_int placeCarac) {
	vector<double> vals(nbCarac + 1);
	double largeurborne = 0;
	double curval = 0;
	double ranval = 0;

	Instance * inst = NULL;

	bool opti = optimize_gen;	//true;

	if (opti) {

		inst = optimizeGeneration(placeCarac, id, nbCarac, taillezone,
				minmaxval);
	} else {	//uniform generation

		if (placeCarac != 0) {
			for (u_int c = 0; c < nbCarac; c++) {
				largeurborne = minmaxval[1][c] - minmaxval[0][c];

//int nb0=0;
//for(int j=0;j<nbData;j++){
//
//	if(data[j].at(c))
//}

				ranval = ((double) Utils::randInt(100000)) / 100000.0;//fraction de 1

				//curval = ranval*(largeurborne+2*taillezone*largeurborne) + minmaxval[0][c] - taillezone*largeurborne;//on genere dans
				curval = ranval * (1 + taillezone) * largeurborne
						+ minmaxval[0][c] - taillezone / 2 * largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur
				vals[c] = curval;
			}
		} else {
			for (u_int c = 1; c < nbCarac + 1; c++) {
				largeurborne = minmaxval[1][c] - minmaxval[0][c];
				ranval = ((double) Utils::randInt(100000)) / 100000.0;
				//curval = ranval*(largeurborne+2*taillezone*largeurborne) + minmaxval[0][c] - taillezone*largeurborne;
				curval = ranval * (1 + taillezone) * largeurborne
						+ minmaxval[0][c] - taillezone / 2 * largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur
				vals[c] = curval;
			}
		}

		vals[placeCarac] = OUTLIER;	//0:outlier;1:target
//    for (int i=0;i<nbCarac+1;i++) {
//        cout << vals[i] << " ";
//    }

		inst = new Instance(id, &vals);
//vals.get_allocator().deallocate(&vals,sizeof(vals));
		//return (new Instance(id,&vals));

		//if(assert(!(&vals==0))) vals.clear();
	}

	return inst;

}

/** Biased Roulette Wheel implementation for 1D complementary histogram generation
 *
 */
void OutlierGenerator::rouletteWheel(DataHandler* handlerBaseOrig,
		vector<vector<int> >& histoOutlier,
		vector<vector<double> >& boundsInterTarget, double param_alpha) {

	int nb_partition = 50; //50: default; TODO: test with 10 [see aggarwal et al. ref?]; other values: see \sqrt{n}
	double largeurborne = 0;    //mesure de l'intervalle

	u_int nbData = handlerBaseOrig->size();
	u_int nbCarac = handlerBaseOrig->dim();
	double **minmaxval = handlerBaseOrig->computeMinMax();
	vector<Instance> data = handlerBaseOrig->getDataSet()->getData();
//int nbCarac=base->dim();

	vector<vector<int> > histoTargetTemp(nbCarac, vector<int>(nb_partition, 0));
	vector<vector<int> > histoOutlierTemp(nbCarac,
			vector<int>(nb_partition, 0));
	vector<vector<double> > boundsInterTargetTemp(nbCarac,
			vector<double>(nb_partition + 1, 0));

	for (u_int c = 0; c < nbCarac; c++) {
		if ((u_int) c == handlerBaseOrig->getClassInd())
			continue;
		largeurborne = minmaxval[1][c] - minmaxval[0][c];
///double largeurborne_alpha=largeurborne*(1+taillezone);
		double largeurborne_alpha = largeurborne * param_alpha;
///int volume_inter=largeurborne_alpha/nb_partition;

		for (int k = 0; k < nb_partition + 1; k++) {
			//curval = ranval*(1+taillezone)*largeurborne + minmaxval[0][c] - taillezone/2*largeurborne;//on genere dans l'enveloppe pour la dimension c une valeur
///	double inter_min=(double)k/(nb_partition)*largeurborne_alpha+minmaxval[0][c]-taillezone/2*largeurborne;
			double inter_min = (double) k / (nb_partition) * largeurborne_alpha
					+ minmaxval[0][c] - (param_alpha - 1) / 2 * largeurborne;

			//inter.at(k)=inter_min;
			boundsInterTargetTemp.at(c).at(k) = inter_min;
		}

		double max_val = 0;
///int somme_effectifs=0;
///int somme_effectifs_derivative=0;

//vector<double> temp=Utils::getMatCol(data);
		vector<double> temp;
		for (u_int i = 0; i < nbData; i++) {
			double val = data.at(i).at(c);
			temp.push_back(val);    //vecteur colonne c

		}
		std::sort(temp.begin(), temp.end());

		u_int k = 0;
		for (int j = 0; j < nb_partition; j++) {
			int eff = 0;
			int iter_nb = 0;

			double t = temp.at(0);
			double a1 = boundsInterTargetTemp.at(c).at(j);
			double a2 = boundsInterTargetTemp.at(c).at(j + 1);
			while (t < a1 && k < nbData) {
				iter_nb++;
				t = temp.at(k);
				k++;
			}

			while (t < a2 && k < nbData) {
				iter_nb++;
				t = temp.at(k);
				k++;
				eff++;
			}

			histoTargetTemp.at(c).at(j) = eff;

			float hist_cost = 1;  //1:default value for Hmax-cost*H; cost<Hmax/H
			histoOutlierTemp.at(c).at(j) = -hist_cost * eff; //TODO:added slack variable - chesner aout12
			if (eff > max_val)
				max_val = eff;
		}

		for (int j = 0; j < nb_partition; j++) {
			histoOutlierTemp.at(c).at(j) += max_val;
			float va = histoOutlierTemp.at(c).at(j);
			if (va < 0) {
				histoOutlierTemp.at(c).at(j) = 0; //no outliers to generate if value of bin zero or negative (ensures to not generate any outlier data)
				cout << "hist bin outlier negative" << va << "/" << max_val
						<< endl;
			}
		}

	}

	histoOutlier = histoOutlierTemp;
	boundsInterTarget = boundsInterTargetTemp;

	histoOutlierTemp.clear();
	histoTargetTemp.clear();
	boundsInterTargetTemp.clear();

	delete[] minmaxval[0];
	delete[] minmaxval[1];
	delete[] minmaxval;

}

