#ifndef RESULT_H_
#define RESULT_H_

#include "../include/datahandler.h"

class Result
{
	private :
		DataHandler * 		testSet;
		u_int 			nbClass;
		u_int **		confMat;
		u_int 			correctClassif;
		u_int			incorrectClassif;
		u_int			total;
		double 			recoRate;
		double 			recoPond;
		double 			fmeasure;
		double			errorRate;
		double 			standDeviat;
		double			maxRecoRate;
		double			minRecoRate;

		bool			fmeas;

		double			timeTrain;

	public :
						Result(DataHandler * set, double time);
		virtual			~Result();

		void			maj_confMat(u_int tru, u_int pred);
		void			setRecoRate(double rate);
		void			setErrorRate(double rate);
		void 			setStandDeviat(double sd);
		void			setTimeTrain(double time);
		void			setMinRecoRate(double min);
		void			setMaxRecoRate(double max);
		void			setFMeasure(double fm);

		double 			getRecoPond(){return recoPond;}
		double 			getRecoRate();
		double			getErrorRate();
		double 			getStandDeviat();
		double			getMinRecoRate();
		double			getMaxRecoRate();
		double 			getFMeasure();
		double 			getTimeTrain();

		string 			toString();
		void            affconfmat();
		u_int **        getconfmat();

};

#endif /*RESULT_H_*/
