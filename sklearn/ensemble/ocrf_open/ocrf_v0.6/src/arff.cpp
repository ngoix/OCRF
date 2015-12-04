
#include "../include/arff.h"


/* **********************************************
 * Load arff Data from the arff file pointed by "arffFile"
 */
DataHandler * Arff::load(const char * arffFile)
{
	string filename(arffFile);
	// creation of an empty dataset


	DataSet * dataset = new DataSet();
	try
	{

		// file openning
		ifstream arff(arffFile, ios_base::in);

		if(arff.is_open())
		{
			string line;



			// ignore comment lines : read the stream till finding a line that is not empty
			// or that does not begin by a '%' (comment character)
			do {getline(arff,line,'\n');}
			while(line.empty() || line.find("%") == 0);


			// read arff formatted data if a '@relation' keyword is found.
			// For any error find in the format of arff data, the loading process is stopped
			if((line.find("@relation") == 0) || (line.find("@Relation") == 0))
			{


				readArffMeta(arff, dataset);

				readArffData(arff, dataset);
			}
			else throw ARFFFormatException("@relation keyword expected here");

		}
		else {

			throw ARFFFileException();

		}

		arff.close();
	}
	catch(ARFFFileException &ex){ throw ex; }
	catch(ARFFFormatException &ex){ throw ex; }
	catch(RFException &ex){ throw ex; }
	catch(exception &ex){ throw ex; }

	// by default, the class position in the attributes vector is set to the last element.
	u_int cl = 0;
	if(dataset->getAttribute(dataset->dim()-1)->is_nominal()) cl = (dataset->dim()-1);

	DataHandler * handler = new DataHandler(dataset,cl);
	//delete dataset;

	handler->setFileName(filename);

	return handler;
}


/* **********************************************
 * This function reads arff information about attributes, to save them in our structure.
 * param :
 * 		arff : the input stream
 *		data : a pointer to the final dataset structure
 */
int Arff::readArffMeta(ifstream & arff, DataSet * dataset)
{

	//cout<<"debug readArffMeta arff 1"<<endl;

	string line = "";
	bool noAtt = true;
	bool attFound = false;

	// The only valid lines here, are those that begin by a '%' (comment) or by '@attribute'.
	// Empty lines or also ignored.
	do
	{
		attFound = false;
		getline(arff,line,'\n');
//cout<<"debug readArffMeta arff 2:"<<line<<endl;
		if((line.find("@attribute") == 0) || (line.find("@Attribute") == 0))
		{
			noAtt = false;
			attFound = true;
			line.erase(0,11);		// erase the '@attribute' keyword

//cout<<"debug readArffMeta arff 3"<<endl;
			// have the position of the '{' character in the line.
			// if the position returned is equal to 'npos', meaning that the character has not been found,
			// the attribute is numeric.
			size_t pos = line.find_first_of('{');
			if(pos == string::npos)
			{	// no '{' have been found : this a numerical attribute
				pos = line.find_first_of(' ');
				string name = line.substr(0,pos);	// just keep the name
				dataset->addAttribute(name,NUMERIC);
				//cout<<"debug readArffMeta arff 4"<<endl;
			}
			else
			{	// this a nominal attribute
				//cout<<"debug readArffMeta arff 5"<<endl;
				string name = line.substr(0,pos-1);
				line.erase(0,pos);			// keep the name and erase it from the string

				// have a vector to store every modality of the nominal attribute.
				vector<string> tokens;
				string tmp = "";
				for(u_int i=1;i<line.size();i++)
				{
					if(line[i]==' ' || line[i] == '\'') continue;
					if(line[i]=='}')
					{
						tokens.push_back(tmp);
						break;
					}
					if(line[i]==',')
					{
						tokens.push_back(tmp);
						tmp = "";
					}
					else tmp += line[i];
				}

				dataset->addAttribute(name,NOMINAL,&tokens);
			}
		}
	}
	while(line.empty() || line.find("%") == 0 || attFound);

//cout<<"debug readArffMeta arff 6"<<endl;

	if(noAtt) throw ARFFFormatException("no @attribute declaration found");

//cout<<"debug readArffMeta arff 7"<<endl;
	if((line.find("@data") != 0) && (line.find("@Data") != 0)) throw ARFFFormatException("@Data keyword expected here");
//cout<<"debug readArffMeta arff 8"<<endl;
	return 0;
}


/* **********************************************
 * This function reads arff information about data to save them in our structure.
 * param :
 * 		arff : the input stream
 *		data : a pointer to the final dataset structure
 */
int Arff::readArffData(ifstream & arff, DataSet * dataset)
{
	string line;
	do
	{
		getline(arff,line,'\n');
		if(!line.empty() && !(line.find("%") == 0))
		{	// empty lines are ignored

			vector<string> tokens;
			Utils::tokenize(line, tokens, ",");

			vector<double> vals;

			u_int i = 0;
			for(vector<string>::iterator it=tokens.begin();it!=tokens.end();it++)
			{

				double v;
				if(dataset->getAttribute(i)->is_nominal())
				{
					for(u_int j=0;j<dataset->getAttribute(i)->getNbModal();j++) {
						string temp = (string)(*it);
						string temp2 = (string)dataset->getAttribute(i)->getModal(j);
						//temp = temp.substr(0,temp2.length());
						//cerr<<temp<<";";
						if(temp.compare(temp2) == 0) {
							//v = (double) j;


							if(temp.compare("outlier")==0){
								v= OUTLIER;
							}
							if(temp.compare("target")==0){
								v= TARGET;
							}


						}
					}
				}
				else {
					 v = Utils::from_string(*it);
				}

				vals.push_back(v);
				i++;
			}

			dataset->addInstance(&vals);
		}
	}
	while(!arff.eof());
	return 0;
}




/* **********************************************
*  This function save data into an arff format file
*/
void Arff::save(const char * arffFile, DataHandler * datahandler)
{
	string filename(arffFile);
	DataSet * dataset = datahandler->getDataSet();
	ofstream file(arffFile);
	file << "@relation " << arffFile << "\n";

	for(u_int i=0;i<dataset->dim();i++)
	{
		file << "@attribute ";
		file << dataset->getAttribute(i)->getName() << " ";

		if(dataset->getAttribute(i)->is_nominal())
		{
			file << "{";
			file << dataset->getAttribute(i)->getModal(0);

			for(u_int j=1;j<dataset->getAttribute(i)->getNbModal();j++)
			{
				file << ",";
				file << dataset->getAttribute(i)->getModal(j);
			}

			file << "}\n";
		}
		else
		file << "real" << "\n";

	}


	file << "@data\n";
	for(u_int k=0;k<dataset->size();k++)
	{
		for(u_int l=0;l<dataset->dim()-1;l++)
		{
			if(dataset->getAttribute(l)->is_nominal())
				file << dataset->getAttribute(l)->getModal((u_int)(dataset->getValue(k,l))) << ",";
			else
				file << dataset->getValue(k,l) << ",";
		}

		if(dataset->getAttribute(dataset->dim()-1)->is_nominal())
			file << dataset->getAttribute(dataset->dim()-1)->getModal((u_int)(dataset->getValue(k,dataset->dim()-1)));
		else
			file << dataset->getValue(k,dataset->dim()-1);

		file << "\n";
	}
	file.close();
	datahandler->setFileName(filename);
}
