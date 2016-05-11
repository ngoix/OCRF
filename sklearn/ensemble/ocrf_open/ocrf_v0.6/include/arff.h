#ifndef ARFF_H_
#define ARFF_H_

#include "../include/datahandler.h"
#include "../include/rfexception.h"


/********************************************************************
*
*   Name:		class Arff
*
*   Description: 	this class is an Arff file reader and writer.
*	It allows to load a dataset from an arff formatted file as well
*	as to save a dataset into an arff file.
*	All the class methods are static, in order to avoid instantiation.
*
*********************************************************************/
class Arff
{
	private :
		static int 				readArffMeta(ifstream & arff, DataSet * dataset);
		static int 				readArffData(ifstream & arff, DataSet * dataset);

	public :
		static DataHandler *	load(const char * arffFile);
		static void  			save(const char * arffFile, DataHandler * data);
};

#endif /*ARFF_H_*/
