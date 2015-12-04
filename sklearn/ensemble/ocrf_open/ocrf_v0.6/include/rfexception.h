#ifndef RFEXCEPTION_H_
#define RFEXCEPTION_H_

#include "../include/utils.h"

class RFException : public exception
{
	protected:
		string 		prefix;
		string 		message;

	public:
				RFException();
				RFException(const RFException &);
		virtual		~RFException() throw();

		string		getMsg();
};

class ARFFFileException : public RFException
{
	public:
		ARFFFileException();
};

class ARFFFormatException : public RFException
{
	public:
		ARFFFormatException(const char * msg);
};

class ScriptFormatException : public RFException
{
	public :
		ScriptFormatException(const char * msg);
};

#endif /*RFEXCEPTION_H_*/
