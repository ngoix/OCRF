#ifndef F_NODE_H_
#define F_NODE_H_

#include "../include/rule.h"

class F_Node;

class F_Node
{
	friend class 			F_DTree;





	private :
		int 				id;
		Rule *				splitRule;
		vector<F_Node *>	children;
		u_int 				prediction;

							F_Node(Rule * rule, int id);
							F_Node(u_int predict, int id);


		string 				fileToString();
		string 				toString();

	public :
		virtual 			~F_Node();

		u_int				getAttInd();
		Rule * 		getSplitRule();
		int		getId();


		F_Node *  	getChild(u_int i);

};

#endif /*F_NODE_H_*/
