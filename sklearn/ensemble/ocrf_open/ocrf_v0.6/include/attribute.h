#ifndef ATTRIBUTE_H_
#define ATTRIBUTE_H_

#include "../include/utils.h"

enum attType {NUMERIC, NOMINAL};

/********************************************************************
*
*   Name:           class Attribute
*
*   Description:  	structure for attribute description.
*       Members are defined as private in order to limitate access and instanciation.
*       It is allowed to be used only by DataSet, which is thus declared as a friend class.
*
*********************************************************************/
class Attribute
{
	friend class 		DataSet;




	private:
		u_int		id;
		string 		name;
		attType 	type;
		vector<string> 	modal;

	public:
				Attribute(u_int id, string n, attType t, vector<string> * mod = NULL);
				~Attribute(){modal.clear();};

		bool		is_nominal();
		u_int		getNbModal();
		string		getModal(u_int i);
		vector<string>&	getModalVect();
		string		getName();
		u_int 		getId();

		string 		toString();
};

#endif /*ATTRIBUTE_H_*/
