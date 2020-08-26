#include "TFPrediction.h"
#include "FFNNConfig.h"
#include "Timer.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h>


typedef std::vector<std::vector<Numeric>> NNInputData;
NNInputData getNNInputdata(std::string sfilepath,ConfigInt nSize, ConfigInt nBatchSize=0)
{
	std::string sLine;
	NNInputData result;
	std::ifstream datafile(sfilepath);
	ConfigInt iBatchCount = 0;
	Numeric data=0;
	if (nBatchSize > 0)
	{
		std::vector<Numeric> vBatch(nSize * nBatchSize);
		while (std::getline(datafile, sLine))
		{
			std::stringstream ssLine(sLine);
			for (ConfigInt i = 0; i < nSize; i++)
			{
				ssLine >> data;
				vBatch[iBatchCount] = data;
			}
			iBatchCount++;
			if (iBatchCount == nBatchSize)
			{
				result.push_back(vBatch);
				iBatchCount = 0;
			}
		}
		if (iBatchCount > 0)
		{
			vBatch.resize(iBatchCount);
			result.push_back(vBatch);
		}
	}
	else
	{
		std::vector<Numeric> vBatch;
		while (std::getline(datafile, sLine))
		{
			std::stringstream ssLine(sLine);
			for (ConfigInt i = 0; i < nSize; i++)
			{
				ssLine >> data;
				vBatch.push_back(data);
			}
		}
		result.push_back(vBatch);
	}
	return result;

}



int main()
{
	
	FFNNConfiguration ffnn_config;
	try
	{
		CTimer tm;
		std::string data_path("./data/");
		std::string xml_path = data_path + std::string("FFNNConfig.xml");
		std::string input_path("../data/input_HH.txt");
		XmlTree xml_tree;
		std::map<std::string, std::tuple<FFNNConfiguration::PairNN, TFPrediction<Numeric>, NNInputData> > mapPairTypeNN;
		std::map <std::string, NNInputData> mapPairTypeInput;
		boost::property_tree::xml_parser::read_xml(xml_path, xml_tree);
		ffnn_config = read_FFNNConfiguration(xml_tree.get_child("FFNNConfiguration"));
		for (auto& pair : ffnn_config.m_PairNNs)
		{
			auto pb_file_path = data_path + ffnn_config.m_sNNFolder + "/" + pair.m_sFile;
			auto input_data_path = pb_file_path + ".data.txt";
			auto nn = TFPrediction<Numeric>(pb_file_path, pair.m_sInputNode,  pair.m_sOutputNode);
			auto nninputdata = getNNInputdata(input_data_path,ffnn_config.m_nNNInputSize,1);
			//auto pairtype = pair.m_sType1.append("_").append(pair.m_sType2);
			//mapPairTypeNN[pairtype] = std::make_tuple(pair,nn, nninputdata);
			std::vector<Numeric> output;
            tm.start();
			for ( auto data_batch : nninputdata)
			{
				if (output.size() < data_batch.size())
					output.resize(data_batch.size());
                    
				if (!data_batch.empty())
				{
					nn.predict(ffnn_config.m_nNNInputSize, data_batch.size(), data_batch.data(), output.data());
				}
				else
				{
					std::cout << "Error! Input data array seems to be empty!!!" << "\n";
					return 0;
				}
			}
            std::cout<< pb_file_path << " prediction time: " << tm.stop_microseconds() << "\n";
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		std::cout << e.what();
	}
	catch (std::exception& ex)
	{
		std::cout << ex.what();
	}
	catch (...)
	{
		std::cout << "Unknown exception has been thrown";
	}

	return 0;
}

