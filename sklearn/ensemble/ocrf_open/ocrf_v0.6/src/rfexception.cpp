
#include "../include/rfexception.h"


RFException::RFException()
{
	prefix = "ERROR -";
	message = "\n\t";
}

RFException::RFException(const RFException &rfexc)
{
	prefix = rfexc.prefix;
	message = rfexc.message;
}

RFException::~RFException()throw() {}

string RFException::getMsg()
{
	string res = prefix;
	res += message;
	return res;
}

ARFFFileException::ARFFFileException()
:RFException()
{
	prefix += " ARFF File Exception ";
	message += " File openning error ";
}


ARFFFormatException::ARFFFormatException(const char * msg)
:RFException()
{
	prefix += " ARFF Format Exception ";
	message += msg;
}


ScriptFormatException::ScriptFormatException(const char * msg)
:RFException()
{
	prefix += " Script Format Exception ";
	message += msg;
}
