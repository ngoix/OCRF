#include "../include/attribute.h"

/* ********************************************************
 * Constructor
 * param
 *		ii : an id integer
 * 		n : the name of the attribute
 * 		t : the type (NUMERIC or NOMINAL)
 * 		mod : a vector of modalities for nominal attributes (=NULL for numeric attribute)
 */
Attribute::Attribute(u_int ii, string n, attType t, vector<string> * mod)
{
	id = ii;
	name = n;
	type = t;
	if(mod != NULL) {
	     modal = (*mod);
	}
}

/* ********************************************************
 * return TRUE if the attribute is nominal
 */
bool	Attribute::is_nominal(){ return type == NOMINAL; }

/* ********************************************************
 * return the number of modalities.
 * As the vector "modal" exist in any case, but can be empty for numeric attribute, the use of the size function is always valid.
 */
u_int	Attribute::getNbModal(){ return modal.size(); }

/* ********************************************************
 * param :
 * 		i : an index for the modality vector
 *
 * return the string modality for the given index
 */
string	Attribute::getModal(u_int i){ return modal.at(i); }

/* ********************************************************
 * param :
 * 		i : an index for the modality vector
 *
 * return the modalities vector for the given index
 */
vector<string> & Attribute::getModalVect(){ return modal; }

/* ********************************************************
 * return the name of the attribute.
 */
string 	Attribute::getName(){ return name; }

/* ********************************************************
 * return the id of the attribute.
 */
u_int 	Attribute::getId(){ return id; }



string	Attribute::toString()
{
	string out = "(";
	out += Utils::to_string((int) id);
	out += ") ";
	out += name;
	out += " ";
	out += Utils::to_string((int) type);

	for(u_int i=0;i<modal.size();i++)
	{
		out += " ";
		out += modal[i];
	}

	return out;
}

