#include "../include/bagging.h"

Bagging::Bagging(double percent)
:bagSize(percent)
{
	if(percent<=0 || percent>100) bagSize = 100;
}

/* ********************************************************
 * return a boostrap sample of the DataHandler member, according to the bagging principle
 */
DataHandler * 	Bagging::generateBootstrap(DataHandler * handl)
{
	u_int s = handl->size();
	u_int size = (u_int) (((double) (s*bagSize))/100.0);

	DataHandler * handler = new DataHandler(handl->getDataSet(),handl->getClassInd(),false);

	vector<u_int> bag;
	for(int i=0;i<(int)size;i++)
	{
		int ind = Utils::randInt(s);
		Instance * inst = handl->getInstance(ind);
		handler->addInstance(inst);

	}

	return handler;
}

void 		Bagging::setbagSize(int r){ bagSize = r; }
