#ifndef RULE_H_
#define RULE_H_

#include "../include/utils.h"
#include "../include/instance.h"
/********************************************************************
*
*   Name:           class _Split
*
*   Description:  	Structure for representing one node split
*      	It corresponds to a unique link toward a children node.
*       Thus it can contain, a modality (or a vector of modality, but not implemented yet)
*       Or it can contain a range of values.
*
*********************************************************************/
class _Split
{
    friend class 		Rule;




    private:
	    vector<u_int> 	modalities;
	    double 		sup;
	    double 		inf;

				_Split(double i, double s):sup(s),inf(i){}
				_Split(u_int mod):sup(0),inf(0){ modalities.push_back(mod); }
				_Split(const _Split &splt)
				{
					_Split * split = const_cast<_Split*>(&splt);
					modalities = split->modalities;
					sup = split->sup;
					inf = split->inf;
				}

public:
				double getSup(){return sup;}
				double getInf(){return inf;}
};


/********************************************************************
*
*   Name:           class Rule
*
*   Description:  	Structure for representing one node splitting rule
*      	It is for the moment restricted to binary split for double valued attributes.
*
*********************************************************************/
class Rule
{
	private:
	    u_int 		attId;     // an index that refer to the splitting criterion attribute in the attributes vector
	    vector<_Split *> 	splits;
	    bool 		nominal;

    public:
						Rule(u_int id, double split);
						Rule(u_int id, u_int nbMod);
						Rule(Rule * rule);
		virtual		    ~Rule();

vector<_Split *> getSplits();

    	u_int 			evaluate(double v);

    	u_int 			getNbSplits();
    	u_int 			getAttId();
    	bool 			is_nominal();

		string 			fileToString();

    	string 			toString();

    	double getSupSplit(){return splits[0]->sup;}


};

#endif /*RULE_H_*/
