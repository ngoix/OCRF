
#include "../include/f_node.h"


F_Node::F_Node(Rule * rule, int i)
{
	id = i;
	splitRule = new Rule(rule);
	prediction = -1;
}

F_Node::F_Node(u_int predict, int i)
{
	id = i;
	prediction = predict;
	splitRule = NULL;
	//delete splitRule;
}

F_Node::~F_Node()
{
	//if(splitRule != NULL)
	delete splitRule;
	if(!children.empty()) for(u_int i=0;i<children.size();i++)
		delete children[i];
}

string 	F_Node::fileToString()
{
	string out = Utils::to_string((int) id);
	out += " ";
	if(splitRule != NULL)
	{
		out += splitRule->fileToString();
		out += " ";

		for(u_int i=0;i<children.size();i++) {out += Utils::to_string((int) children[i]->id); out += " ";}
		out += "\n";
		for(vector<F_Node*>::iterator it=children.begin();it!=children.end();it++)
			out += (*it)->fileToString();
	}
	else
	{
		out += Utils::to_string((int) prediction);
		out += "\n";
	}

	return out;
}

Rule * 		F_Node::getSplitRule(){ return splitRule; }
int	F_Node::getId(){ return id; }

u_int			F_Node::getAttInd()
{
	return splitRule->getAttId();
}

F_Node *  	F_Node::getChild(u_int i){ return children[i]; }

string 		F_Node::toString()
{
	string out;
	/*if(type == LEAF) out += "[LEAF] ";
	if(type == ROOT || type == ROOTLEAF) out += "[ROOT] ";
	if(type == NODE) out += "[NODE] ";
	*/
	out += Utils::to_string((int) id) ;
	out += " : ";
	if(splitRule != NULL)
		out += splitRule->toString();
	else
	{
		out += "class ";
		out += Utils::to_string((int) prediction);
	}

	if(!children.empty())
	{
		out += "\n";
		for(vector<F_Node*>::iterator it=children.begin();it!=children.end();it++)
			out += (*it)->toString();
	}
	return out;
}


