#ifndef BAGGING_H_
#define BAGGING_H_

#include "../include/datahandler.h"

/********************************************************************
*
*   Name:           	class Bagging
*
*   Description:	The class for implementing the Bagging operator
*	It is linked to only one DataHandler.
*
*********************************************************************/
class Bagging
{
	private :
		double		bagSize;

	public :
				Bagging(double percent);

		DataHandler * 	generateBootstrap(DataHandler * handler);
		void 		setbagSize(int r);
};

#endif /*BAGGING_H_*/
