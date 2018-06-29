package gocudnn

/*
#include <cublas_v2.h>
*/
import "C"

//CublasFlags holds all the flags for cublas
type CublasFlags struct {
	FillMode    CublasFillModeFlag
	DiagType    CublasDiagTypeFlag
	SideMode    CublasSideModeFlag
	Operation   CublasOperationFlag
	PointerMode CublasPointerModeFlag
	AtomicsMode CublasAtomicsModeFlag
	GemmAlgo    CublasGemmAlgoFlag
	Math        CublasMathFlag
}

//CublasFillModeFlag is a nill struct to pass CublasFillMode
type CublasFillModeFlag struct {
}

//CublasFillMode are the flags for the different fill modes of cublas
type CublasFillMode C.cublasFillMode_t

func (blas CublasFillMode) c() C.cublasFillMode_t { return C.cublasFillMode_t(blas) }

//Lower is CublasFillMode Flag
func (blas CublasFillModeFlag) Lower() CublasFillMode { return CublasFillMode(C.CUBLAS_FILL_MODE_LOWER) }

//Upper is CublasFillMode Flag
func (blas CublasFillModeFlag) Upper() CublasFillMode { return CublasFillMode(C.CUBLAS_FILL_MODE_UPPER) }

//CublasDiagTypeFlag is a nil struct used to pass CublasDIagType flags
type CublasDiagTypeFlag struct {
}

//CublasDiagType is used on the go surface for cublas flags
type CublasDiagType C.cublasDiagType_t

func (blas CublasDiagType) c() C.cublasDiagType_t { return C.cublasDiagType_t(blas) }

//NonUnit is a flag for CublasDiagType
func (blas CublasDiagTypeFlag) NonUnit() CublasDiagType { return CublasDiagType(C.CUBLAS_DIAG_NON_UNIT) }

//Unit is a flag for CublasDiagType
func (blas CublasDiagTypeFlag) Unit() CublasDiagType { return CublasDiagType(C.CUBLAS_DIAG_UNIT) }

//CublasSideModeFlag is a nil struct that is used to pass CublasSideMode flags as methods
type CublasSideModeFlag struct {
}

//CublasSideMode is a go type for cublasSideMode_t
type CublasSideMode C.cublasSideMode_t

func (blas CublasSideMode) c() C.cublasSideMode_t { return C.cublasSideMode_t(blas) }

//Left is a flag for CublasSideMode
func (blas CublasSideModeFlag) Left() CublasSideMode { return CublasSideMode(C.CUBLAS_SIDE_LEFT) }

//Right is a flag for CublasSideMode
func (blas CublasSideModeFlag) Right() CublasSideMode { return CublasSideMode(C.CUBLAS_SIDE_RIGHT) }

//CublasOperationFlag is a nil struct that is used to pass CublasOperation Flags through methods
type CublasOperationFlag struct {
}

//CublasOperation is a go type for C.cublasOperations_t
type CublasOperation C.cublasOperation_t

func (blas CublasOperation) c() C.cublasOperation_t { return C.cublasOperation_t(blas) }

//N is a flag for CublasOperation
func (blas CublasOperationFlag) N() CublasOperation { return CublasOperation(C.CUBLAS_OP_N) }

//T is a flag for CublasOperation
func (blas CublasOperationFlag) T() CublasOperation { return CublasOperation(C.CUBLAS_OP_T) }

//C is a flag for CublasOperation
func (blas CublasOperationFlag) C() CublasOperation { return CublasOperation(C.CUBLAS_OP_C) }

//CublasPointerModeFlag is a nil struct that is used to pass CublasPointerMode flags through methods
type CublasPointerModeFlag struct {
}

//CublasPointerMode is a go type for cublasPointerMode_t
type CublasPointerMode C.cublasPointerMode_t

func (blas CublasPointerMode) c() C.cublasPointerMode_t { return C.cublasPointerMode_t(blas) }

//Host is a flag for CublasPointerMode
func (blas CublasPointerModeFlag) Host() CublasPointerMode {
	return CublasPointerMode(C.CUBLAS_POINTER_MODE_HOST)
}

//Device is a flag for CublasPointerMode
func (blas CublasPointerModeFlag) Device() CublasPointerMode {
	return CublasPointerMode(C.CUBLAS_POINTER_MODE_DEVICE)
}

//CublasAtomicsModeFlag is a nil struct used to pass CublasAtmoicsMode flags through methods
type CublasAtomicsModeFlag struct {
}

//CublasAtomicsMode is a go type for C.cublasAtomicsMode_t
type CublasAtomicsMode C.cublasAtomicsMode_t

func (blas CublasAtomicsMode) c() C.cublasAtomicsMode_t { return C.cublasAtomicsMode_t(blas) }

//NotAllowed is a flag for CublasAtomicsMode
func (blas CublasAtomicsModeFlag) NotAllowed() CublasAtomicsMode {
	return CublasAtomicsMode(C.CUBLAS_ATOMICS_NOT_ALLOWED)
}

//Allowed is a flag for CublasAtomicsMode
func (blas CublasAtomicsModeFlag) Allowed() CublasAtomicsMode {
	return CublasAtomicsMode(C.CUBLAS_ATOMICS_ALLOWED)
}

//CublasGemmAlgoFlag is a nil struct used for passing CublasGemmAlgo Algos through structs
type CublasGemmAlgoFlag struct {
}

//CublasGemmAlgo is a go type for  C.cublasGemmAlgo_t
type CublasGemmAlgo C.cublasGemmAlgo_t

func (blas CublasGemmAlgo) c() C.cublasGemmAlgo_t { return C.cublasGemmAlgo_t(blas) }

//Default is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Default() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_DEFAULT) }

//Algo0 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo0() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO0) }

//Algo1 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo1() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO1) }

//Algo2 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo2() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO2) }

//Algo3 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo3() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO3) }

//Algo4 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo4() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO4) }

//Algo5 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo5() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO5) }

//Algo6 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo6() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO6) }

//Algo7 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo7() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO7) }

//Algo8 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo8() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO8) }

//Algo9 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo9() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO9) }

//Algo10 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo10() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO10) }

//Algo11 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo11() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO11) }

//Algo12 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo12() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO12) }

//Algo13 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo13() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO13) }

//Algo14 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo14() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO14) }

//Algo15 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo15() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO15) }

//Algo16 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo16() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO16) }

//Algo17 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo17() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO17) }

//Algo18 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo18() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO18) }

//Algo19 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo19() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO19) }

//Algo20 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo20() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO20) }

//Algo21 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo21() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO21) }

//Algo22 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo22() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO22) }

//Algo23 is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) Algo23() CublasGemmAlgo { return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO23) }

//TensorOpDefault is a flag for CublasGemAlgo
func (blas CublasGemmAlgoFlag) TensorOpDefault() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
}

//TensorOpAlgo0  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo0() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO0_TENSOR_OP)
}

//TensorOpAlgo1  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo1() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO1_TENSOR_OP)
}

//TensorOpAlgo2  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo2() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO2_TENSOR_OP)
}

//TensorOpAlgo3  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo3() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO3_TENSOR_OP)
}

//TensorOpAlgo4  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo4() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO4_TENSOR_OP)
}

//TensorOpAlgo5  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo5() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO5_TENSOR_OP)
}

//TensorOpAlgo6  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo6() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO6_TENSOR_OP)
}

//TensorOpAlgo7  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo7() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO7_TENSOR_OP)
}

//TensorOpAlgo8  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo8() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO8_TENSOR_OP)
}

//TensorOpAlgo9  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo9() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO9_TENSOR_OP)
}

//TensorOpAlgo10  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo10() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO10_TENSOR_OP)
}

//TensorOpAlgo11  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo11() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO11_TENSOR_OP)
}

//TensorOpAlgo12  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo12() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO12_TENSOR_OP)
}

//TensorOpAlgo13  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo13() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO13_TENSOR_OP)
}

//TensorOpAlgo14  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo14() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO14_TENSOR_OP)
}

//TensorOpAlgo15  is a flag for CublasGemmAlgo
func (blas CublasGemmAlgoFlag) TensorOpAlgo15() CublasGemmAlgo {
	return CublasGemmAlgo(C.CUBLAS_GEMM_ALGO15_TENSOR_OP)
}

//CublasMathFlag is a nil struct to pass flags for CublasMath
type CublasMathFlag struct {
}

//CublasMath is the type that hold C.cublasMath_t
type CublasMath C.cublasMath_t

//c is a flag for CublasMath
func (blas CublasMath) c() C.cublasMath_t { return C.cublasMath_t(blas) }

//Default is a flag for CublasMath
func (blas CublasMathFlag) Default() CublasMath { return CublasMath(C.CUBLAS_DEFAULT_MATH) }

//TensorOp is a flag for CublasMath
func (blas CublasMathFlag) TensorOp() CublasMath { return CublasMath(C.CUBLAS_TENSOR_OP_MATH) }
