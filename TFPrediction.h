#ifndef _TFPREDICTION_H_
#define _TFPREDICTION_H_

#include <tensorflow/c/c_api.h>
#include <memory>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cstring>

enum R_Code { OK, TF_FAILED, READ_PB_FAILED, GRAPH_OPERATION_NOT_FOUND, GRAPH_DIMMENSIONS_NOT_FOUND, GRAPH_DATA_TYPE_MISMATCH, TENSOR_FAILED, DIMMENSION_MISMATCH };

template<typename T>
class TFPrediction
{	
public:

	TFPrediction(std::string pbPath, std::string inputNode, std::string outputNode);
	~TFPrediction();

	int getDimmension();

	R_Code predict(int dimmension, int points, T* input, T* output);
	R_Code calculateGradient(int dimmension, int points, T* input, T* output);

	TF_Code get_TF_Error_Code();
	std::string get_TF_Error_Message();

	R_Code get_Error_Code();

private:

	TF_Buffer* readPB(const std::string filename);
	void set_Error_Code(R_Code code);

private:

	//Model Data
	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;
	std::string g_inputNode;
	std::string g_outputNode;
	TF_Output gradient;

	R_Code m_errorCode;

private:
	
	template<typename TT>
	class Tensor
	{
	public:
		Tensor(TFPrediction* tfp, std::string nodeName, TT* data, int data_size);
		~Tensor();
		void set_data(TFPrediction* tfp, TT* new_data, int size);
		static int deduce_type();
	public:
		TF_Tensor* val;
		TF_Output op;
		TF_DataType dataType;
		std::vector<int64_t> shape;
	};
};

template<typename T>
TFPrediction<T>::TFPrediction(std::string pbPath, std::string inputNode, std::string outputNode): m_errorCode(OK)
{
	this->status = TF_NewStatus();
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}
	
	this->graph = TF_NewGraph();
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	this->session = TF_NewSession(this->graph, sess_opts, this->status);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	TF_DeleteSessionOptions(sess_opts);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	TF_Buffer* def = readPB(pbPath);
	if (def == nullptr)
	{
		std::cout << "Failed to load input pb file '" << pbPath << "'. Terminated...\n";
		set_Error_Code(READ_PB_FAILED);
		return;
	}

	TF_Graph* g = this->graph;
	TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	TF_GraphImportGraphDef(g, def, graph_opts, this->status);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	TF_DeleteImportGraphDefOptions(graph_opts);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	TF_DeleteBuffer(def);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	g_inputNode = inputNode;
	g_outputNode = outputNode;

	TF_Output grad_output;
	TF_Output grad_input;
	TF_Operation* input_op  = TF_GraphOperationByName(this->graph, inputNode.c_str());
	TF_Operation* output_op = TF_GraphOperationByName(this->graph, outputNode.c_str());
	grad_input  = TF_Output{ input_op, 0 };
	grad_output = TF_Output{ output_op, 0 };

	TF_AddGradients(this->graph, &grad_output, 1, &grad_input, 1, nullptr, this->status, &this->gradient);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return;
	}

	/*
	//Iterate throw graph nodes, print node names
	const char* name = nullptr;
	size_t pos = 0;
	TF_Operation* oper;
	while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr)
	{
		name = TF_OperationName(oper);
		std::cout << name << "\n";
	}
	*/
}	

template<typename T>
TFPrediction<T>::~TFPrediction()
{
	if(this->session && this->status)
		TF_DeleteSession(this->session, this->status);

	if(this->graph)
		TF_DeleteGraph(this->graph);

	if(this->status)
		TF_DeleteStatus(this->status);
}	
	
template<typename T>
R_Code TFPrediction<T>::calculateGradient(int dimmension, int points, T* input, T* output)
{	
	Tensor<T> inputTensor(this, g_inputNode, input, dimmension*points);
	if (get_Error_Code() != OK)
	{
		return get_Error_Code();
	}

	TF_Tensor* ov;
	TF_SessionRun(this->session, nullptr, &inputTensor.op, &inputTensor.val, 1, &this->gradient, &ov, 1, nullptr, 0, nullptr, this->status);
	if (get_TF_Error_Code() != TF_OK)
	{	
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}	
		
	auto raw_data = TF_TensorData(ov);
	if (get_TF_Error_Code() != TF_OK)
	{	
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}	
		
	size_t size = TF_TensorByteSize(ov) / TF_DataTypeSize(TF_TensorType(ov));
	if (get_TF_Error_Code() != TF_OK)
	{	
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}	
		
	if (size != dimmension*points)
	{	
		set_Error_Code(DIMMENSION_MISMATCH);
		return DIMMENSION_MISMATCH;
	}	
	
	const auto T_data = static_cast<T*>(raw_data);
	auto result = std::vector<T>(T_data, T_data + size);
	for (int i = 0; i < dimmension*points; i++)
	{
		output[i] = result[i];
	}
	
	TF_DeleteTensor(ov);
	if (get_TF_Error_Code() != TF_OK)
	{	
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}	
		
	return OK;
}		

template<typename T>
R_Code TFPrediction<T>::predict(int dimmension, int points, T* input, T* output)
{
	Tensor<T> inputTensor(this, g_inputNode, input, dimmension * points);
	if (get_Error_Code() != OK)
	{
		return get_Error_Code();
	}

	Tensor<T> outputTensor(this, g_outputNode, nullptr, 0);
	if (get_Error_Code() != OK)
	{
		return get_Error_Code();
	}

	TF_Tensor* ov;
	TF_SessionRun(this->session, nullptr, &inputTensor.op, &inputTensor.val, 1, &outputTensor.op, &ov, 1, nullptr, 0, nullptr, this->status);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}

	auto raw_data = TF_TensorData(ov);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}

	size_t size = TF_TensorByteSize(ov) / TF_DataTypeSize(TF_TensorType(ov));
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}

	if (size != points)
	{
		set_Error_Code(DIMMENSION_MISMATCH);
		return DIMMENSION_MISMATCH;
	}
	
	const auto T_data = static_cast<T*>(raw_data);
	auto result = std::vector<T>(T_data, T_data + size);
	for (int i = 0; i < points; i++)
	{
		output[i] = result[i];
	}

	TF_DeleteTensor(ov);
	if (get_TF_Error_Code() != TF_OK)
	{
		set_Error_Code(TF_FAILED);
		return TF_FAILED;
	}

	return OK;
}	
	
template<typename T>
int TFPrediction<T>::getDimmension()
{	
	Tensor<T> inputTensor(this, g_inputNode, nullptr, 0);
	if (get_Error_Code() != OK)
	{
		return -1;
	}

	return (int)*std::max_element(inputTensor.shape.begin(), inputTensor.shape.end());
}
	
template<typename T>
R_Code TFPrediction<T>::get_Error_Code()
{
	return m_errorCode;
}

template<typename T>
void TFPrediction<T>::set_Error_Code(R_Code code)
{
	m_errorCode = code;
}

template<typename T>
TF_Code TFPrediction<T>::get_TF_Error_Code()
{
	return TF_GetCode(this->status);
}

template<typename T>
std::string TFPrediction<T>::get_TF_Error_Message()
{
	if (TF_GetCode(this->status) != TF_OK)
	{
		return std::string(TF_Message(status));
	}
	else
	{
		return std::string("");
	}
}

template<typename T>
TF_Buffer* TFPrediction<T>::readPB(const std::string filename)
{
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		return nullptr;
	}

	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	auto data = new char[size];
	file.seekg(0, std::ios::beg);
	file.read(data, size);

	if (!file || !size) {
		return nullptr;
	}

	TF_Buffer* buffer = TF_NewBufferFromString(data, size);
	file.close();
	delete[] data;
	return buffer;
}

template<typename T> template<typename TT>
TFPrediction<T>::Tensor<TT>::Tensor(TFPrediction* tfp, std::string nodeName, TT* data, int data_size)
{		
	this->op.oper = TF_GraphOperationByName(tfp->graph, nodeName.c_str());
	this->op.index = 0;

	if (tfp->get_TF_Error_Code() != TF_OK)
	{
		tfp->set_Error_Code(TF_FAILED);
		return;
	}

	if (this->op.oper == nullptr)
	{
		tfp->set_Error_Code(GRAPH_OPERATION_NOT_FOUND);
		std::cout << "Graph operation '"<< nodeName <<"' not found. Terminated...\n";
		return;
	}

	int n_dims = TF_GraphGetTensorNumDims(tfp->graph, this->op, tfp->status);
	if (tfp->get_TF_Error_Code() != TF_OK)
	{
		tfp->set_Error_Code(TF_FAILED);
		return;
	}

	if (n_dims <= 0)
	{
		tfp->set_Error_Code(GRAPH_DIMMENSIONS_NOT_FOUND);
		std::cout << "Graph dimmensions not determined. Node name = '" << nodeName << "'. Terminated...\n";
		return;
	}

	this->dataType = TF_OperationOutputType(this->op);
	if (tfp->get_TF_Error_Code() != TF_OK)
	{
		tfp->set_Error_Code(TF_FAILED);
		return;
	}

	if (deduce_type() != (int)this->dataType)
	{
		tfp->set_Error_Code(GRAPH_DATA_TYPE_MISMATCH);
		std::cout << "Graph data type mismatch. Node name = '" << nodeName << "'. Terminated...\n";
		return;
	}

	auto* dims = new int64_t[n_dims];
	TF_GraphGetTensorShape(tfp->graph, this->op, dims, n_dims, tfp->status);
	if (tfp->get_TF_Error_Code() != TF_OK)
	{
		tfp->set_Error_Code(TF_FAILED);
		return;
	}

	this->shape = std::vector<int64_t>(dims, dims + n_dims);
	delete[] dims;

	this->val = nullptr;

	if (data != nullptr && data_size != 0)
	{
		this->set_data(tfp, data, data_size);
	}
}

template<typename T> template<typename TT>
void TFPrediction<T>::Tensor<TT>::set_data(TFPrediction* tfp, TT* new_data, int size)
{
	auto d = [](void* ddata, size_t, void*) { free(static_cast<T*>(ddata)); };

	std::unique_ptr<std::vector<int64_t>> actual_shape;
	actual_shape = std::make_unique<decltype(actual_shape)::element_type>(this->shape.begin(), this->shape.end());

	auto exp_size = std::abs(std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int64_t>()));
	std::replace_if(actual_shape->begin(), actual_shape->end(), [](int64_t r) {return r == -1; }, size / exp_size);

	auto t_size = sizeof(TT) * size;
	void* data = malloc(t_size);
	memcpy(data, new_data, t_size);
	this->val = TF_NewTensor(this->dataType, actual_shape->data(), actual_shape->size(), data, t_size, d, nullptr);
	if (tfp->get_TF_Error_Code() != TF_OK)
	{
		tfp->set_Error_Code(TF_FAILED);
		return;
	}	
		
	if (this->val == nullptr)
	{	
		tfp->set_Error_Code(TENSOR_FAILED);
		return;
	}	
}	

template<typename T> template<typename TT>
int TFPrediction<T>::Tensor<TT>::deduce_type()
{
	if (std::is_same<T, float>::value)
		return TF_FLOAT;
	if (std::is_same<T, double>::value)
		return TF_DOUBLE;
	if (std::is_same<T, int32_t >::value)
		return TF_INT32;
	if (std::is_same<T, uint8_t>::value)
		return TF_UINT8;
	if (std::is_same<T, int16_t>::value)
		return TF_INT16;
	if (std::is_same<T, int8_t>::value)
		return TF_INT8;
	if (std::is_same<T, int64_t>::value)
		return TF_INT64;
	if (std::is_same<T, uint16_t>::value)
		return TF_UINT16;
	if (std::is_same<T, uint32_t>::value)
		return TF_UINT32;
	if (std::is_same<T, uint64_t>::value)
		return TF_UINT64;

	return 0;
}

template<typename T> template<typename TT>
TFPrediction<T>::Tensor<TT>::~Tensor() {
	if (this->val != nullptr) {
		TF_DeleteTensor(this->val);
	}
}

#endif