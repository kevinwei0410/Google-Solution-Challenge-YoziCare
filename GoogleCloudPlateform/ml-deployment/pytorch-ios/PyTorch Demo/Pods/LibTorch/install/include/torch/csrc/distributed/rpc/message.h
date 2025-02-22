#pragma once

#include <torch/csrc/utils/future.h>
#include <torch/types.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// An enum denoting common RPC errors to allow specific error handling for them.
enum RPCErrorType {
  UNKNOWN_ERROR = 0, /* Indicates that error type could not be parsed */
  TIMEOUT = 1, /* Indicates that the RPC has timed out */
  INTENTIONAL_FAILURE = 2 /* Deliberate failure, such as those injected by
                             FaultyProcessGroupAgent for testing */
};

// The enum values are bitwise ORed with MessageType
// They are bit flags starting from 0x100 and should have
// value such as 0x100, 0x200, 0x400, 0x800, 0xF00, etc.
enum MessageTypeFlags {
  REQUEST_TYPE = 0x100,
  RESPONSE_TYPE = 0x200,
};

// Message types must have values between 0 to 255
enum MessageType {
  // messages for dist.rpc on builtin operators
  SCRIPT_CALL = 0 | MessageTypeFlags::REQUEST_TYPE,
  SCRIPT_RET = 1 | MessageTypeFlags::RESPONSE_TYPE,

  // messages for dist.rpc on Python UDF
  PYTHON_CALL = 2 | MessageTypeFlags::REQUEST_TYPE,
  PYTHON_RET = 3 | MessageTypeFlags::RESPONSE_TYPE,

  // messages for dist.remote on builtin operators and Python UDF
  SCRIPT_REMOTE_CALL =
      4 | MessageTypeFlags::REQUEST_TYPE, // A remote call on a builtin operator
  PYTHON_REMOTE_CALL =
      5 | MessageTypeFlags::REQUEST_TYPE, // A remote call on a Python UDF
  REMOTE_RET =
      6 | MessageTypeFlags::RESPONSE_TYPE, // Response for remote calls for UDF,
                                           // builtin, or script

  // RRef related internal messages
  SCRIPT_RREF_FETCH_CALL =
      7 | MessageTypeFlags::REQUEST_TYPE, // A UserRRef<IValue> fetches value
                                          // from owner
  PYTHON_RREF_FETCH_CALL =
      8 | MessageTypeFlags::REQUEST_TYPE, // A UserRRef<py::object> fetches
                                          // value from owner
  SCRIPT_RREF_FETCH_RET =
      9 | MessageTypeFlags::RESPONSE_TYPE, // An OwnerRRef sends ivalue to user
  PYTHON_RREF_FETCH_RET = 10 |
      MessageTypeFlags::RESPONSE_TYPE, // An OwnerRRef sends py::object to user
  RREF_USER_DELETE = 11 |
      MessageTypeFlags::REQUEST_TYPE, // A UserRRef tells the owner to deref
  RREF_FORK_REQUEST =
      12 | MessageTypeFlags::REQUEST_TYPE, // A child UserRRef tells the owner
                                           // about itself
  RREF_CHILD_ACCEPT =
      13 | MessageTypeFlags::REQUEST_TYPE, // A child UserRRef tells parent that
                                           // owner knows it
  RREF_ACK =
      14 | MessageTypeFlags::RESPONSE_TYPE, // ACK to internal RRef messages

  // Messages with autograd info
  FORWARD_AUTOGRAD_REQ = 15 | MessageTypeFlags::REQUEST_TYPE,
  FORWARD_AUTOGRAD_RESP = 16 | MessageTypeFlags::RESPONSE_TYPE,

  // Messages to propagate gradients on the backward pass.
  BACKWARD_AUTOGRAD_REQ = 17 | MessageTypeFlags::REQUEST_TYPE,
  BACKWARD_AUTOGRAD_RESP = 18 | MessageTypeFlags::RESPONSE_TYPE,

  // Messages to tell workers to clean up their autograd context.
  CLEANUP_AUTOGRAD_CONTEXT_REQ = 19 | MessageTypeFlags::REQUEST_TYPE,
  CLEANUP_AUTOGRAD_CONTEXT_RESP = 20 | MessageTypeFlags::RESPONSE_TYPE,

  // Messages that tell workers to run requests with profiling enabled.
  RUN_WITH_PROFILING_REQ = 21 | MessageTypeFlags::REQUEST_TYPE,
  RUN_WITH_PROFILING_RESP = 22 | MessageTypeFlags::RESPONSE_TYPE,

  // Messages to support RRef.backward().
  RREF_BACKWARD_REQ = 23 | MessageTypeFlags::REQUEST_TYPE,
  RREF_BACKWARD_RESP = 24 | MessageTypeFlags::RESPONSE_TYPE,

  // Other internal message types
  EXCEPTION = 55 | MessageTypeFlags::RESPONSE_TYPE,
  UNKNOWN = 60
};

// A message to be sent/received by an RpcAgent.
//
// A Message object contains 4 fields:
//    payload (std::vector<char>): a binary chunk of data.
//    tensors (std::vector<torch::Tensor>): all tensors. Tensor data are not
//        included in the payload, and it is up to the RpcAgent implementation
//        to determine how to serialize them. This design is helpful for
//        communicating super large tensors where serializing all the data at
//        once leads to excessively large memory footprint. An implementation
//        can then serialize and send tensors chunck-by-chunk, in the streaming
//        fashion.
//    type (MessageType): type of the message.
//    id (int64_t): message id, this is used by ProcessGroupAgent to match
//                  request and response. Other implementation can ignore it
//                  if they have their own ways to do matching.
//
// Layers above ``RpcAgent`` only converts ScriptCall, ScriptResp, PythonCall,
// and PythonResp into a Message, and it is up to the RpcAgent
// implementation to determine how to serialize a message.
class TORCH_API Message final : public torch::CustomClassHolder {
 public:
  Message();

  Message(
      std::vector<char>&& payload,
      std::vector<torch::Tensor>&& tensors,
      MessageType type);

  Message(
      std::vector<char>&& payload,
      std::vector<torch::Tensor>&& tensors,
      MessageType type,
      int64_t id);

  Message(const Message& other);
  Message(Message&& other) noexcept;
  Message& operator=(Message const& rhs) &;
  Message& operator=(Message&& rhs) &;
  void swap(Message& rhs) noexcept;

  // Destructively retrieves the payload.
  std::vector<char>&& movePayload() &&;
  std::vector<torch::Tensor>&& moveTensors() &&;

  std::vector<char>& payload();
  const std::vector<char>& payload() const;
  std::vector<torch::Tensor>& tensors();
  const std::vector<torch::Tensor>& tensors() const;
  MessageType type() const;

  bool isRequest() const;
  bool isResponse() const;
  bool isShutdown() const;

  // id is an optional field to match request/response. If an RpcAgent
  // implementation is able to do the matching without using this id, it can be
  // dropped during message serialization.
  int64_t id() const;
  void setId(int64_t id);

 private:
  std::vector<char> payload_;
  std::vector<torch::Tensor> tensors_;
  MessageType type_ = MessageType::UNKNOWN;
  int64_t id_ = -1;
};

// Create a response Message of type Exception.
// The exception string representation will be used as the message's payload.
// A message ID corresponding to the request that resulted in this response can
// be provided for matching requests/responses.
TORCH_API Message createExceptionResponse(const std::exception& e, int64_t id);

// Create a response Message of type Exception.
// The passed in string representation will be used as the message's payload.
// A message ID corresponding to the request that resulted in this response can
// be provided for matching requests/responses.
TORCH_API Message
createExceptionResponse(const std::string& exceptionStr, int64_t id);

using JitFuture = c10::ivalue::Future;

} // namespace rpc
} // namespace distributed
} // namespace torch
