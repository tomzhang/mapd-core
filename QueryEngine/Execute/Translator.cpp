#include "Translator.h"

#include <glog/logging.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cstdint>


FetchInt64Col::FetchInt64Col(const int64_t col_id,
                             const std::shared_ptr<Decoder> decoder)
  : col_id_{col_id}, decoder_{decoder} {}

llvm::Value* FetchInt64Col::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  // only generate the decoding code once; if a column has been previously
  // fetch in the generated IR, we'll reuse it
  auto it = fetch_cache_.find(col_id_);
  if (it != fetch_cache_.end()) {
    return it->second;
  }
  auto& in_arg_list = func->getArgumentList();
  CHECK_EQ(in_arg_list.size(), 2);
  auto& byte_stream_arg = in_arg_list.front();
  auto& pos_arg = in_arg_list.back();
  auto& context = llvm::getGlobalContext();
  auto it_ok = fetch_cache_.insert(std::make_pair(
      col_id_,
      decoder_->codegenDecode(
        &byte_stream_arg,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), col_id_),
        &pos_arg,
        ir_builder,
        module)));
  CHECK(it_ok.second);
  return it_ok.first->second;
}

std::unordered_map<int64_t, llvm::Value*> FetchInt64Col::fetch_cache_;

ImmInt64::ImmInt64(const int64_t val) : val_{val} {}

llvm::Value* ImmInt64::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto& context = llvm::getGlobalContext();
  return llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), val_);
}

OpGt::OpGt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpGt::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSGT(lhs, rhs);
}

OpLt::OpLt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpLt::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSLT(lhs, rhs);
}

OpGte::OpGte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpGte::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSGE(lhs, rhs);
}

OpLte::OpLte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpLte::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpSLE(lhs, rhs);
}

OpNeq::OpNeq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpNeq::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpNE(lhs, rhs);
}

OpEq::OpEq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpEq::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateICmpEQ(lhs, rhs);
}

OpAdd::OpAdd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpAdd::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateAdd(lhs, rhs);
}

OpSub::OpSub(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpSub::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateSub(lhs, rhs);
}

OpMul::OpMul(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpMul::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateMul(lhs, rhs);
}

OpDiv::OpDiv(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpDiv::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateSDiv(lhs, rhs);
}

OpAnd::OpAnd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpAnd::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateAnd(lhs, rhs);
}

OpOr::OpOr(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs)
  : lhs_{lhs}, rhs_{rhs} {}

llvm::Value* OpOr::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto lhs = lhs_->codegen(func, ir_builder, module);
  auto rhs = rhs_->codegen(func, ir_builder, module);
  return ir_builder.CreateOr(lhs, rhs);
}

OpNot::OpNot(std::shared_ptr<AstNode> op) : op_{op} {}

llvm::Value* OpNot::codegen(
    llvm::Function* func,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  return ir_builder.CreateNot(op_->codegen(func, ir_builder, module));
}

extern int _binary_RuntimeFunctions_ll_size;
extern int _binary_RuntimeFunctions_ll_start;
extern int _binary_RuntimeFunctions_ll_end;

AggQueryCodeGenerator::AggQueryCodeGenerator(
    std::shared_ptr<AstNode> filter,
    const std::string& query_template_name,
    const std::string& filter_placeholder_name) {
  auto& context = llvm::getGlobalContext();
  // read the LLIR embedded as ELF binary data
  auto llir_size = reinterpret_cast<size_t>(&_binary_RuntimeFunctions_ll_size);
  auto llir_data_start = reinterpret_cast<const char*>(&_binary_RuntimeFunctions_ll_start);
  auto llir_data_end = reinterpret_cast<const char*>(&_binary_RuntimeFunctions_ll_end);
  CHECK_EQ(llir_data_end - llir_data_start, llir_size);
  std::string llir_data(llir_data_start, llir_size);
  auto llir_mb = llvm::MemoryBuffer::getMemBuffer(llir_data, "", true);
  llvm::SMDiagnostic err;
  auto module = llvm::ParseIR(llir_mb, err, context);
  CHECK(module);
  // generate the filter
  auto ft = llvm::FunctionType::get(
    llvm::Type::getInt64Ty(context),
    std::vector<llvm::Type*> {
      llvm::PointerType::get(llvm::Type::getInt8PtrTy(context), 0),
      llvm::Type::getInt32Ty(context)
    },
    false);
  auto filter_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "filter", module);
  // will trigger inlining of the filter to avoid call overhead and register spills / fills
  filter_func->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);
  auto& arg_list = filter_func->getArgumentList();
  // the filter function has two arguments: the (compressed) columns and the current row index
  CHECK_EQ(arg_list.size(), 2);

  llvm::IRBuilder<> ir_builder(context);
  auto bb = llvm::BasicBlock::Create(context, "entry", filter_func);
  ir_builder.SetInsertPoint(bb);
  ir_builder.CreateRet(filter->codegen(filter_func, ir_builder, module));

  auto query_func = module->getFunction(query_template_name);
  CHECK(query_func);

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    CHECK_EQ(std::string(filter_call.getCalledFunction()->getName()), filter_placeholder_name);
    std::vector<llvm::Value*> args {
      filter_call.getArgOperand(0),
      filter_call.getArgOperand(1)
    };
    llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(filter_func, args, ""));
    break;
  }

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);

  std::string err_str;
  llvm::EngineBuilder eb(module);
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::JIT);
  llvm::TargetOptions to;
  to.EnableFastISel = true;
  eb.setTargetOptions(to);
  execution_engine_ = eb.create();
  CHECK(execution_engine_);

  // honor the always inline attribute for the runtime functions and the filter
  llvm::legacy::PassManager pass_manager;
  pass_manager.add(llvm::createAlwaysInlinerPass());
  pass_manager.run(*module);

  query_native_code_ = execution_engine_->getPointerToFunction(query_func);
}

AggQueryCodeGenerator::~AggQueryCodeGenerator() {
  // looks like ExecutionEngine owns everything (IR, native code etc.)
  delete execution_engine_;
}
