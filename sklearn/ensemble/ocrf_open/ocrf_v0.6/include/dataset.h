#ifndef DATASET_H_
#define DATASET_H_

#include "../include/utils.h"
#include "../include/attribute.h"
#include "../include/instance.h"
/********************************************************************
*
*   Name:           class DataSet
*
*   Description:  	our DataSet structure.
*
*********************************************************************/
class DataSet
{






    protected:
        vector<Attribute> 	attributes;
        vector<Instance> 	data;

        u_int			idsInst;
	u_int			idsAtt;

    public:
	u_int 			nbRef;

	virtual			~DataSet();
                    DataSet(DataSet * d);
        			DataSet();

        int 		addAttribute(string n, attType t, vector<string> * modal = NULL);
	int			addAttribute(Attribute & a);
        int			addInstance(vector<double> * vals,int=-1);

        u_int			size();
        u_int			dim();
        Instance *		getInstance(u_int instInd);
        Attribute *		getAttribute(u_int attInd);
        double			getValue(u_int instInd, u_int attInd);
        int             getNBAttribute();
	u_int			getNbModal(u_int attInd);
	void            affbase();

	vector<Instance> getData(){return data;}
	void setData(vector<Instance> temp){data=temp;}

};

#endif /*DATASET_H_*/
