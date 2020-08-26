#pragma once 
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/property_tree/xml_parser.hpp>


typedef double Numeric;
typedef std::string ConfigString;
typedef double ConfigReal;
typedef int ConfigInt;
typedef boost::property_tree::basic_ptree<std::string, std::string> XmlTree;

struct FFNNConfiguration
{
	struct PairNN
	{
		struct InputNN
		{
			ConfigInt m_nId;
			ConfigString m_sName;
			ConfigString m_sNormType;
			ConfigReal m_fMean;
			ConfigReal m_fDisp;
		};

		ConfigString m_sType1;
		ConfigString m_sType2;
		ConfigString m_sFile;
		ConfigString m_sInputNode;
		ConfigString m_sOutputNode;
		std::vector<InputNN> m_InputNNs;
	};

	ConfigString m_sNNFolder;
	ConfigInt  m_nNNInputSize;
	std::vector<PairNN> m_PairNNs;
};
template<typename T> void get_attr(T& Val, const std::string& sName, const XmlTree& xml_tree)
{
	std::string s_sAttrName("<xmlattr>.");
	Val = xml_tree.get<T>(s_sAttrName.append(sName));
}
template<typename T> void get_seq(std::vector<T>& Val, const std::string& sName, const XmlTree& xml_tree, T(*fnRead)(const XmlTree& xml_tree))
{
	XmlTree::const_assoc_iterator ait;
	for (ait = xml_tree.find(sName); ait != xml_tree.not_found(); ++ait)
	{
		if (sName == ait->first)
		{
			Val.push_back(fnRead(ait->second));
		}
	}
}

FFNNConfiguration::PairNN::InputNN read_InputNN(const XmlTree& xml_tree)
{
	FFNNConfiguration::PairNN::InputNN input_nn;
	get_attr(input_nn.m_nId, "Id", xml_tree);
	get_attr(input_nn.m_sName, "Name", xml_tree);
	get_attr(input_nn.m_sNormType, "NormType", xml_tree);
	get_attr(input_nn.m_fMean, "Mean", xml_tree);
	get_attr(input_nn.m_fDisp, "Disp", xml_tree);
	return input_nn;
}

FFNNConfiguration::PairNN read_PairNN(const XmlTree& xml_tree)
{
	FFNNConfiguration::PairNN pair_nn;
	get_attr(pair_nn.m_sFile, "File", xml_tree);
	get_attr(pair_nn.m_sType1, "Type1", xml_tree);
	get_attr(pair_nn.m_sType2, "Type2", xml_tree);
	get_attr(pair_nn.m_sInputNode, "InputNode", xml_tree);
	get_attr(pair_nn.m_sOutputNode, "OutputNode", xml_tree);
	get_seq(pair_nn.m_InputNNs, "InputNN", xml_tree, read_InputNN);
	return pair_nn;
}

FFNNConfiguration read_FFNNConfiguration(const XmlTree& xml_tree)
{
	FFNNConfiguration ffnn_config;
	get_attr(ffnn_config.m_sNNFolder, "NNFolder", xml_tree);
	get_attr(ffnn_config.m_nNNInputSize, "NNInputSize", xml_tree);
	get_seq(ffnn_config.m_PairNNs, "PairNN", xml_tree, read_PairNN);
	return ffnn_config;
}


