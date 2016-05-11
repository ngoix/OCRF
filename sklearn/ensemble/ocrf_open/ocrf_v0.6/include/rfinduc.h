#ifndef RFINDUC_H_
#define RFINDUC_H_

#include "../include/dforest.h"
#include "../include/datahandler.h"

class RFInduc
{
	protected :
		DataHandler * 		trainSet;
		int 			L;
		double 			tim;
		bool 			disp;

	public :
					RFInduc(int l, bool d);
		virtual 		~RFInduc();

		virtual DForest * 	growForest(DataHandler * set) = 0; // this method has to be defined in every descendant class
};


#endif /*RFINDUC_H_*/
