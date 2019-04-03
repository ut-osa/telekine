#ifndef __VGPU_TENSORFLOW_TYPES_H__
#define __VGPU_TENSORFLOW_TYPES_H__

/* Tensorflow types: tensorflow/c/c_api.h */

// --------------------------------------------------------------------------
// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The enum values here are identical to corresponding values in types.proto.
typedef enum TF_DataType {
  TF_FLOAT = 1,
  TF_DOUBLE = 2,
  TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
  TF_UINT8 = 4,
  TF_INT16 = 5,
  TF_INT8 = 6,
  TF_STRING = 7,
  TF_COMPLEX64 = 8,  // Single-precision complex
  TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
  TF_INT64 = 9,
  TF_BOOL = 10,
  TF_QINT8 = 11,     // Quantized int8
  TF_QUINT8 = 12,    // Quantized uint8
  TF_QINT32 = 13,    // Quantized int32
  TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  TF_QINT16 = 15,    // Quantized int16
  TF_QUINT16 = 16,   // Quantized uint16
  TF_UINT16 = 17,
  TF_COMPLEX128 = 18,  // Double-precision complex
  TF_HALF = 19,
  TF_RESOURCE = 20,
  TF_VARIANT = 21,
  TF_UINT32 = 22,
  TF_UINT64 = 23,
} TF_DataType;

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
typedef enum TF_Code {
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_UNAUTHENTICATED = 16,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15,
} TF_Code;

// --------------------------------------------------------------------------
// TF_Status holds error information.  It either has an OK code, or
// else an error code with an associated error message.
typedef struct TF_Status TF_Status;

// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// The format for TF_STRING tensors is:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   The string length (as a varint), followed by the contents of the string
//   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
//   facilitate this encoding.
typedef struct TF_Tensor TF_Tensor;

// --------------------------------------------------------------------------
// TF_SessionOptions holds options that can be passed during session creation.
typedef struct TF_SessionOptions TF_SessionOptions;

// Represents a computation graph.  Graphs may be shared between sessions.
// Graphs are thread-safe when used as directed below.
typedef struct TF_Graph TF_Graph;

// TF_ImportGraphDefOptions holds options that can be passed to
// TF_GraphImportGraphDef.
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;

// Operation being built. The underlying graph must outlive this.
typedef struct TF_OperationDescription TF_OperationDescription;

// Operation that has been added to the graph. Valid until the graph is
// deleted -- in particular adding a new operation to the graph does not
// invalidate old TF_Operation* pointers.
typedef struct TF_Operation TF_Operation;

// Represents a specific input of an operation.
typedef struct TF_Input {
  TF_Operation* oper;
  int index;  // The index of the input within oper.
} TF_Input;

// Represents a specific output of an operation.
typedef struct TF_Output {
  TF_Operation* oper;
  int index;  // The index of the output within oper.
} TF_Output;

// TF_Function is a grouping of operations with defined inputs and outputs.
// Once created and added to graphs, functions can be invoked by creating an
// operation whose operation type matches the function name.
typedef struct TF_Function TF_Function;

// Function definition options. TODO(iga): Define and implement
typedef struct TF_FunctionOptions TF_FunctionOptions;

// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

typedef struct TF_Session TF_Session;

#endif
