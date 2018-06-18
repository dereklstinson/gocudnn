package gocudnn

//Flags contains all flag structs in one struct.  Hopefully helpful in sending flags
type Flags struct {
	ActivationMode      ActivationModeFlag
	BatchNormMode       BatchNormModeFlag
	ConvolutionMode     ConvolutionModeFlag
	ConvBwdDataPref     ConvBwdDataPrefFlag
	ConvBwdDataAlgo     ConvBwdDataAlgoFlag
	ConvBwdFilterPref   ConvBwdFilterPrefFlag
	ConvBwdFiltAlgo     ConvBwdFiltAlgoFlag
	ConvolutionFwdPref  ConvolutionFwdPrefFlag
	CTCLossAlgo         CTCLossAlgoFlag
	LRNmode             LRNmodeFlag
	DivNormMode         DivNormModeFlag
	MemcpyKind          MemcpyKindFlag
	OpTensor            OpTensorFlag
	PoolingMode         PoolingModeFlag
	ReduceTensorOp      ReduceTensorOpFlag
	ReduceTensorIndices ReduceTensorIndicesFlag
	IndiciesType        IndiciesTypeFlag
	RNNMode             RNNModeFlag
	DirectionMode       DirectionModeFlag
	RNNInputMode        RNNInputModeFlag
	RNNAlgo             RNNAlgoFlag
	SoftMaxAlgorithm    SoftMaxAlgorithmFlag
	SoftMaxMode         SoftMaxModeFlag
	DataType            DataTypeFlag
	MathType            MathTypeFlag
	PropagationNAN      PropagationNANFlag
	Determinism         DeterminismFlag
	TensorFormat        TensorFormatFlag
}
