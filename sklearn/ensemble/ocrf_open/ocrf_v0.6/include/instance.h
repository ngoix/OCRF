#ifndef INSTANCE_H_
#define INSTANCE_H_

#include "../include/utils.h"

/********************************************************************
*
*   Name:           class Instance
*
*   Description:  	structure for a piece of Instance object.
*       This class is only defined for internal uses.
*       Members are defined as private in order to limitate access and instanciation.
*       It is allowed to be used only by DataSet, which is thus declared as a friend class.
*
*********************************************************************/
class Instance
{
	friend class 	DataSet;




	private:
    	vector<double> 	vect;
    	u_int 		id;
    	int originalId;//original instance id (e.g. from global learning set)

    public :
			Instance(u_int id, vector<double>* vals = NULL);

			// this copy constructor is needed to be public in order to use Instance objects in vector container.
			// When an object is inserted in a vector container, it is actually
			// duplicated in the vector structure, via the copy constructor.
			// The parameter is needed to be const. We thus have to use the const_cast operator
			// in the copy process to be able to push them back to Instance vectors.
			Instance(const Instance &cdat);
			virtual ~Instance();

 void    	add(double v);
 double 		at(u_int att);
 u_int		getId();
 u_int       getClass();
 string 		toString();

 vector<double> * getVect();
 vector<double> getVectSimple();

 void setVect(vector<double> temp){vect=temp;}

 void modClass(u_int newClass);

 void setOriginalId(int id_temp){originalId=id_temp;};
int getOriginalId(){return originalId;}

};

#endif /*INSTANCE_H_*/
