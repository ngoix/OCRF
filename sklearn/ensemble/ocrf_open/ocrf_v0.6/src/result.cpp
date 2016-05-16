
#include "../include/result.h"

Result::Result(DataHandler * set, double time) {

//    testSet = new DataHandler(*set);
    testSet = set;
    nbClass = testSet->getNbClass();
    timeTrain = time;

    confMat = new u_int*[nbClass];
    for (u_int i=0;i<nbClass;i++) {
        confMat[i] = new u_int[nbClass];
        for (u_int j=0;j<nbClass;j++)
            confMat[i][j] = 0;
    }

    correctClassif = 0;
    incorrectClassif = 0;
    total = 0;
    recoRate = 0.0;
    errorRate = 0.0;
    fmeasure = 0.0;
    standDeviat = 0.0;
    maxRecoRate = 0.0;
    minRecoRate = 0.0;
    fmeas = false;
}

Result::~Result() {

    for (u_int i=0;i<nbClass;i++)
        delete[] confMat[i];
    delete[] confMat;

}

void		Result::maj_confMat(u_int tru, u_int pred) {
    if (tru == pred) correctClassif++;
    else incorrectClassif++;
    total++;
    confMat[tru][pred]++;
//    recoRate = (((double) correctClassif)/((double) total))*100.0;
//    errorRate = (((double) incorrectClassif)/((double) total))*100.0;

    double* temp=new double[nbClass];
    double somme=0;
    double somme_diag=0;
    double somme_tot=0;

    recoRate=0;
    errorRate=0;
    for(u_int i=0;i<nbClass;i++){
    	somme=0;

			somme_diag+=confMat[i][i];

    	for(u_int j=0;j<nbClass;j++){
    		somme+=confMat[i][j];
    		somme_tot+=confMat[i][j];
    	}

    	if(somme!=0){
				temp[i]=(double)confMat[i][i]/somme*100;
				recoRate+=temp[i]/nbClass;
				errorRate=100-recoRate;
    	}

    }

    if(somme_tot!=0) recoPond=somme_diag/somme_tot*100;
    else recoPond=-1;

delete[] temp;
}

void		Result::setRecoRate(double rate) {
    recoRate = rate;
}

void		Result::setErrorRate(double rate) {
    errorRate = rate;
}

void		Result::setStandDeviat(double sd) {
    standDeviat = sd;
}

void		Result::setTimeTrain(double time) {
    timeTrain = time;
}

void 		Result::setMinRecoRate(double min) {
    minRecoRate = min;
}

void		Result::setMaxRecoRate(double max) {
    maxRecoRate = max;
}

void		Result::setFMeasure(double fm) {
    fmeas = true;
    fmeasure = fm;
}

double 		Result::getRecoRate() {
    return recoRate;
}

double		Result::getErrorRate() {
    return errorRate;
}

double 		Result::getStandDeviat() {
    return standDeviat;
}

double 		Result::getFMeasure() {
    if (!fmeas) {
        u_int ptcl[nbClass];
        u_int clpt[nbClass];
        u_int tot = 0;
        double f = 0.0;

        for (u_int i=0;i<nbClass;i++) {
            ptcl[i] = 0;
            clpt[i] = 0;
            for (u_int j=0;j<nbClass;j++) {
                tot += confMat[i][j];
                ptcl[i] += confMat[j][i];
                clpt[i] += confMat[i][j];
            }
        }

        for (u_int i=0;i<nbClass;i++) {
            double fmaxj = 0.0;
            for (u_int j=0;j<nbClass;j++) {
                double rec = ((double) confMat[i][j]) / ((double) clpt[i]);
                double prec = ((double) confMat[i][j]) / ((double) ptcl[j]);
                double fij = 0.0;
                if (rec!=0 || prec!=0) fij = (2*prec*rec) / (prec+rec);
                if (fij > fmaxj) fmaxj = fij;
            }
            f += (((double) clpt[i])/ ((double) tot)) * fmaxj;
        }
        fmeasure = f;
        fmeas = true;
    }
    return fmeasure;
}

double 		Result::getTimeTrain() {
    return timeTrain;
}

double		Result::getMinRecoRate() {
    return minRecoRate;
}

double		Result::getMaxRecoRate() {
    return maxRecoRate;
}

string 		Result::toString() {
    string out = "\n";
    out += "Test sample size : \t";
    out += Utils::to_string((int)testSet->size());
    out += "\nNombre d'instances bien classées : \t";
    out += Utils::to_string((int)correctClassif);
    out += " ( ";
//    out += Utils::to_string(getRecoRate());
    out += Utils::to_string(recoPond);
    out += "% )";
    out += "\nNombre d'instances mal classées : \t";
    out += Utils::to_string((int)incorrectClassif);
    out += " ( ";
//    out += Utils::to_string(getErrorRate());
    out += Utils::to_string(100-recoPond);
    out += "% )";
    //out += "\nF-mesure : \t";
    //out += Utils::to_string(getFMeasure());
    out += "\n\n";

    for (u_int i=0;i<nbClass;i++) {
        for (u_int j=0;j<nbClass;j++) {
            out += Utils::to_string((int)(confMat[i][j]));
            out += "\t";
        }

        out += "\n";
    }

    return out;
}

void Result::affconfmat() {
    int somme;
    cerr<<"Affichage conf_mat"<<endl;
    for (u_int i=0;i<nbClass;i++) {
        for (u_int j=0;j<nbClass;j++) {
            cout << confMat[i][j]<<" ";
        }
        cout << "\n";
    }
    for (u_int i=0;i<nbClass;i++) {
        somme =0;
        for (u_int j=0;j<nbClass;j++) {
            somme +=confMat[i][j];
        }
        cerr << "performance pour classe "<<i<<" : "<< (double)confMat[i][i]/(double)somme*100<<"\n";
    }
}

u_int ** Result::getconfmat() {
    return confMat;
}
