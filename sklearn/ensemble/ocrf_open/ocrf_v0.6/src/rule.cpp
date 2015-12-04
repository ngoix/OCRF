#include "../include/rule.h"

Rule::Rule(u_int id, double split)
{
    attId = id;
    nominal = false;
    splits.push_back(new _Split(-(numeric_limits<double>::max()),split));
    splits.push_back(new _Split(split,numeric_limits<double>::max()));
}

Rule::Rule(u_int id, u_int nbMod)
{
    attId = id;
    nominal = true;
    for(u_int i=0;i<nbMod;i++)
        splits.push_back(new _Split(i));
}

Rule::Rule(Rule* rule)
{
	attId = rule->attId;
	nominal = rule->nominal;
	for(vector<_Split *>::iterator it=rule->splits.begin();it!=rule->splits.end();it++)
	{
		const _Split * tmp = (const _Split *) (*it);
		_Split * sp = new _Split(*tmp);
		splits.push_back(sp);
	}
	//splits = rule->splits;
}

Rule::~Rule()
{
    for(vector<_Split *>::iterator it=splits.begin();it!=splits.end();it++)
        delete (*it);
}


u_int 		Rule::evaluate(double v)
{
    if(nominal)
    {
    		u_int ind = 0;
	    for(vector<_Split *>::iterator it=splits.begin();it!=splits.end();it++)
	    {
	        for(v_u_int::iterator it2=(*it)->modalities.begin();it2!=(*it)->modalities.end();it2++)
	            if((u_int)v == (*it2)) return ind;

	        ind++;
	    }
	    return 1000;
    }
    else
    {
	    int ind = 0;
    		///double precision=0.0;
	    for(vector<_Split *>::iterator it=splits.begin();it!=splits.end();it++)
	    {
	        if(((*it)->sup > v) && ((*it)->inf <= v))
	            return ind;
	        ind++;
	    }
	    //cout<<"return 0:"<<endl;
	    return -1;
    }
}

vector<_Split *>	Rule::getSplits(){ return splits; }
u_int 		Rule::getNbSplits(){ return splits.size(); }
u_int 		Rule::getAttId(){ return attId; }
bool 		Rule::is_nominal(){ return nominal; }

string 		Rule::fileToString()
{
	string out = Utils::to_string((int) attId);
	out += " ";
	out += Utils::to_string(splits[0]->sup);

	return out;
}

string 		Rule::toString()
{
    string out;
    out = Utils::to_string((int)(attId));
    if(nominal)
    {
        out += " (Nominal)/ ";
        out += Utils::to_string((int)(splits.size()));
        out += " nodes";
    }
    else
    {
        out += " (Numerical)/ ";
        out += Utils::to_string((splits.front())->sup);
    }
    return out;
}
