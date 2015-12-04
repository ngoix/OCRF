#ifndef DTINDUC_H_
#define DTINDUC_H_

#include "../include/dtree.h"
#include "../include/utils.h"

/********************************************************************
*
*   Name: class DTInduc
*
*   Description: virtual structure of a Decision Tree Induction algorithm
*		This class has to be extended
*
*********************************************************************/
class DTInduc
{
    protected :
        DataHandler * 	trainSet;    // the current Training Set

        double 		time;

    public:
        		DTInduc();
        virtual 	~DTInduc();

        virtual DTree * growTree(DataHandler * set) = 0; // this method has to be defined in every subclass

};

#endif /*DTINDUC_H_*/
