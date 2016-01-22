#include "QueryTemplateGenerator.h"

#include <glog/logging.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>

// This file was pretty much auto-generated by running:
//      llc -march=cpp RuntimeFunctions.ll
// and formatting the results to be more readable.

namespace {

llvm::Function* pos_start_helper(llvm::Module* mod, const std::string& name) {
  using namespace llvm;

  std::vector<Type*> FuncTy_7_args;
  FunctionType* FuncTy_7 = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_7_args,
      /*isVarArg=*/false);

  auto func_pos_start = mod->getFunction(name);
  if (!func_pos_start) {
    func_pos_start = Function::Create(
        /*Type=*/FuncTy_7,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/name,
        mod);  // (external, no body)
    func_pos_start->setCallingConv(CallingConv::C);
  }

  AttributeSet func_pos_start_PAL;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pos_start_PAL = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_pos_start->setAttributes(func_pos_start_PAL);

  return func_pos_start;
}

llvm::Function* pos_start(llvm::Module* mod) {
  return pos_start_helper(mod, "pos_start");
}

llvm::Function* group_buff_idx(llvm::Module* mod) {
  return pos_start_helper(mod, "group_buff_idx");
}

llvm::Function* pos_step(llvm::Module* mod) {
  using namespace llvm;

  std::vector<Type*> FuncTy_7_args;
  FunctionType* FuncTy_7 = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_7_args,
      /*isVarArg=*/false);

  auto func_pos_step = mod->getFunction("pos_step");
  if (!func_pos_step) {
    func_pos_step = Function::Create(
        /*Type=*/FuncTy_7,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"pos_step",
        mod);  // (external, no body)
    func_pos_step->setCallingConv(CallingConv::C);
  }

  AttributeSet func_pos_step_PAL;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_pos_step_PAL = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_pos_step->setAttributes(func_pos_step_PAL);

  return func_pos_step;
}

llvm::Function* init_group_by_buffer(llvm::Module* mod) {
  using namespace llvm;

  auto i64_type = IntegerType::get(mod->getContext(), 64);
  auto pi64_type = PointerType::get(i64_type, 0);
  auto i32_type = IntegerType::get(mod->getContext(), 32);

  std::vector<Type*> init_group_by_buffer_args{pi64_type, pi64_type, i32_type, i32_type, i32_type};

  auto init_group_by_buffer_type =
      FunctionType::get(Type::getVoidTy(mod->getContext()), init_group_by_buffer_args, false);

  auto func_init_group_by_buffer = mod->getFunction("init_group_by_buffer");
  if (!func_init_group_by_buffer) {
    func_init_group_by_buffer =
        Function::Create(init_group_by_buffer_type, GlobalValue::ExternalLinkage, "init_group_by_buffer", mod);
    func_init_group_by_buffer->setCallingConv(CallingConv::C);
  }

  AttributeSet func_init_group_by_buffer_PAL;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);
    func_init_group_by_buffer_PAL = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_init_group_by_buffer->setAttributes(func_init_group_by_buffer_PAL);

  return func_init_group_by_buffer;
}

llvm::Function* row_process(llvm::Module* mod,
                            const size_t aggr_col_count,
                            const bool is_nested,
                            const bool hoist_literals) {
  using namespace llvm;

  std::vector<Type*> FuncTy_5_args;
  PointerType* PointerTy_6 = PointerType::get(IntegerType::get(mod->getContext(), 64), 0);

  if (aggr_col_count) {
    for (size_t i = 0; i < aggr_col_count; ++i) {
      FuncTy_5_args.push_back(PointerTy_6);
    }
  } else {                                 // group by query
    FuncTy_5_args.push_back(PointerTy_6);  // groups buffer
    FuncTy_5_args.push_back(PointerTy_6);  // small groups buffer
    FuncTy_5_args.push_back(PointerTy_6);  // max matched
  }

  FuncTy_5_args.push_back(PointerTy_6);  // aggregate init values

  FuncTy_5_args.push_back(IntegerType::get(mod->getContext(), 64));
  FuncTy_5_args.push_back(IntegerType::get(mod->getContext(), 64));
  FuncTy_5_args.push_back(PointerTy_6);
  if (hoist_literals) {
    FuncTy_5_args.push_back(PointerType::get(IntegerType::get(mod->getContext(), 8), 0));
  }
  FunctionType* FuncTy_5 = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_5_args,
      /*isVarArg=*/false);

  auto row_process_name = unique_name("row_process", is_nested);
  auto func_row_process = mod->getFunction(row_process_name);

  if (!func_row_process) {
    func_row_process = Function::Create(
        /*Type=*/FuncTy_5,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/row_process_name,
        mod);  // (external, no body)
    func_row_process->setCallingConv(CallingConv::C);

    AttributeSet func_row_process_PAL;
    {
      SmallVector<AttributeSet, 4> Attrs;
      AttributeSet PAS;
      {
        AttrBuilder B;
        PAS = AttributeSet::get(mod->getContext(), ~0U, B);
      }

      Attrs.push_back(PAS);
      func_row_process_PAL = AttributeSet::get(mod->getContext(), Attrs);
    }
    func_row_process->setAttributes(func_row_process_PAL);
  }

  return func_row_process;
}

}  // namespace

llvm::Function* query_template(llvm::Module* mod,
                               const size_t aggr_col_count,
                               const bool is_nested,
                               const bool hoist_literals) {
  using namespace llvm;

  auto func_pos_start = pos_start(mod);
  CHECK(func_pos_start);
  auto func_pos_step = pos_step(mod);
  CHECK(func_pos_step);
  auto func_group_buff_idx = group_buff_idx(mod);
  CHECK(func_group_buff_idx);
  auto func_row_process = row_process(mod, aggr_col_count, is_nested, hoist_literals);
  CHECK(func_row_process);

  PointerType* PointerTy_1 = PointerType::get(IntegerType::get(mod->getContext(), 8), 0);
  PointerType* PointerTy_6 = PointerType::get(IntegerType::get(mod->getContext(), 64), 0);
  PointerType* PointerTy_9 = PointerType::get(PointerTy_1, 0);

  std::vector<Type*> FuncTy_8_args;
  FuncTy_8_args.push_back(PointerTy_9);
  if (hoist_literals) {
    FuncTy_8_args.push_back(PointerTy_1);
  }
  FuncTy_8_args.push_back(PointerTy_6);
  FuncTy_8_args.push_back(PointerTy_6);
  FuncTy_8_args.push_back(PointerTy_6);

  FuncTy_8_args.push_back(PointerTy_6);
  PointerType* PointerTy_10 = PointerType::get(PointerTy_6, 0);
  FuncTy_8_args.push_back(PointerTy_10);
  FuncTy_8_args.push_back(PointerTy_10);
  FuncTy_8_args.push_back(IntegerType::get(mod->getContext(), 32));
  FuncTy_8_args.push_back(IntegerType::get(mod->getContext(), 64));
  FuncTy_8_args.push_back(PointerType::get(IntegerType::get(mod->getContext(), 32), 0));

  FunctionType* FuncTy_8 = FunctionType::get(
      /*Result=*/Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_8_args,
      /*isVarArg=*/false);

  auto query_template_name = unique_name("query_template", is_nested);
  auto func_query_template = mod->getFunction(query_template_name);
  CHECK(!func_query_template);

  func_query_template = Function::Create(
      /*Type=*/FuncTy_8,
      /*Linkage=*/GlobalValue::ExternalLinkage,
      /*Name=*/query_template_name,
      mod);
  func_query_template->setCallingConv(CallingConv::C);

  AttributeSet func_query_template_PAL;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 1U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 2U, B);
    }

    Attrs.push_back(PAS);

    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      Attrs.push_back(AttributeSet::get(mod->getContext(), 3U, B));
    }

    {
      AttrBuilder B;
      B.addAttribute(Attribute::NoCapture);
      Attrs.push_back(AttributeSet::get(mod->getContext(), 4U, B));
    }

    Attrs.push_back(PAS);

    func_query_template_PAL = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_query_template->setAttributes(func_query_template_PAL);

  Function::arg_iterator args = func_query_template->arg_begin();
  Value* ptr_byte_stream_119 = args++;
  ptr_byte_stream_119->setName("byte_stream");
  Value* literals{nullptr};
  if (hoist_literals) {
    literals = args++;
    literals->setName("literals");
  }
  Value* ptr_row_count_ptr = args++;
  ptr_row_count_ptr->setName("row_count_ptr");
  Value* frag_row_off_ptr = args++;
  frag_row_off_ptr->setName("frag_row_off_ptr");
  Value* ptr_max_matched_ptr = args++;
  ptr_max_matched_ptr->setName("max_matched_ptr");
  Value* ptr_agg_init_val = args++;
  ptr_agg_init_val->setName("agg_init_val");
  Value* ptr_out = args++;
  ptr_out->setName("out");
  Value* ptr_unused = args++;
  ptr_unused->setName("unused");
  Value* frag_idx = args++;
  frag_idx->setName("frag_idx");
  Value* ptr_join_hash_table = args++;
  ptr_join_hash_table->setName("join_hash_table");
  Value* ptr_error_code = args++;
  ptr_error_code->setName("error_code");

  BasicBlock* label_120 = BasicBlock::Create(mod->getContext(), "", func_query_template, 0);
  BasicBlock* label__lr_ph = BasicBlock::Create(mod->getContext(), ".lr.ph", func_query_template, 0);
  BasicBlock* label_121 = BasicBlock::Create(mod->getContext(), "", func_query_template, 0);
  BasicBlock* label___crit_edge = BasicBlock::Create(mod->getContext(), "._crit_edge", func_query_template, 0);
  BasicBlock* label_122 = BasicBlock::Create(mod->getContext(), "", func_query_template, 0);

  // Block  (label_120)
  std::vector<Value*> ptr_result_vec;
  for (size_t i = 0; i < aggr_col_count; ++i) {
    auto ptr_result = new AllocaInst(IntegerType::get(mod->getContext(), 64), "result", label_120);
    ptr_result->setAlignment(8);
    ptr_result_vec.push_back(ptr_result);
  }

  LoadInst* int64_123 = new LoadInst(ptr_row_count_ptr, "", false, label_120);
  int64_123->setAlignment(8);
  LoadInst* frag_row_off = new LoadInst(frag_row_off_ptr, "", false, label_120);
  frag_row_off->setAlignment(8);

  std::vector<Value*> int64_124_vec;
  for (size_t i = 0; i < aggr_col_count; ++i) {
    auto idx_lv = ConstantInt::get(IntegerType::get(mod->getContext(), 32), i);
    auto agg_init_gep = GetElementPtrInst::CreateInBounds(ptr_agg_init_val, idx_lv, "", label_120);
    auto int64_124 = new LoadInst(agg_init_gep, "", false, label_120);
    int64_124->setAlignment(8);
    int64_124_vec.push_back(int64_124);
    auto void_125 = new StoreInst(int64_124, ptr_result_vec[i], false, label_120);
    void_125->setAlignment(8);
  }

  CallInst* int32_126 = CallInst::Create(func_pos_start, "", label_120);
  int32_126->setCallingConv(CallingConv::C);
  int32_126->setTailCall(true);
  AttributeSet int32_126_PAL;
  int32_126->setAttributes(int32_126_PAL);

  CallInst* int32_127 = CallInst::Create(func_pos_step, "", label_120);
  int32_127->setCallingConv(CallingConv::C);
  int32_127->setTailCall(true);
  AttributeSet int32_127_PAL;
  int32_127->setAttributes(int32_127_PAL);

  CallInst* group_buff_idx = CallInst::Create(func_group_buff_idx, "", label_120);
  group_buff_idx->setCallingConv(CallingConv::C);
  group_buff_idx->setTailCall(true);
  AttributeSet group_buff_idx_PAL;
  group_buff_idx->setAttributes(group_buff_idx_PAL);

  CastInst* int64_128 = new SExtInst(int32_126, IntegerType::get(mod->getContext(), 64), "", label_120);
  ICmpInst* int1_129 = new ICmpInst(*label_120, ICmpInst::ICMP_SLT, int64_128, int64_123, "");
  BranchInst::Create(label__lr_ph, label_122, int1_129, label_120);

  // Block .lr.ph (label__lr_ph)
  CastInst* int64_131 = new SExtInst(int32_127, IntegerType::get(mod->getContext(), 64), "", label__lr_ph);
  BranchInst::Create(label_121, label__lr_ph);

  // Block  (label_121)
  Argument* fwdref_133 = new Argument(IntegerType::get(mod->getContext(), 64));
  PHINode* int64_pos_01 = PHINode::Create(IntegerType::get(mod->getContext(), 64), 2, "pos.01", label_121);
  int64_pos_01->addIncoming(int64_128, label__lr_ph);
  int64_pos_01->addIncoming(fwdref_133, label_121);

  std::vector<Value*> void_134_params;
  void_134_params.insert(void_134_params.end(), ptr_result_vec.begin(), ptr_result_vec.end());
  void_134_params.push_back(ptr_agg_init_val);
  void_134_params.push_back(int64_pos_01);
  void_134_params.push_back(frag_row_off);
  void_134_params.push_back(ptr_row_count_ptr);
  if (hoist_literals) {
    CHECK(literals);
    void_134_params.push_back(literals);
  }
  CallInst* void_134 = CallInst::Create(func_row_process, void_134_params, "", label_121);
  void_134->setCallingConv(CallingConv::C);
  void_134->setTailCall(false);
  AttributeSet void_134_PAL;
  void_134->setAttributes(void_134_PAL);

  BinaryOperator* int64_135 = BinaryOperator::CreateNSW(Instruction::Add, int64_pos_01, int64_131, "", label_121);
  ICmpInst* int1_136 = new ICmpInst(*label_121, ICmpInst::ICMP_SLT, int64_135, int64_123, "");
  BranchInst::Create(label_121, label___crit_edge, int1_136, label_121);

  // Block ._crit_edge (label___crit_edge)
  std::vector<Instruction*> int64__pre_vec;
  for (size_t i = 0; i < aggr_col_count; ++i) {
    auto int64__pre = new LoadInst(ptr_result_vec[i], ".pre", false, label___crit_edge);
    int64__pre->setAlignment(8);
    int64__pre_vec.push_back(int64__pre);
  }

  BranchInst::Create(label_122, label___crit_edge);

  // Block  (label_122)
  std::vector<PHINode*> int64_139_vec;
  for (int64_t i = aggr_col_count - 1; i >= 0; --i) {
    auto int64_139 = PHINode::Create(IntegerType::get(mod->getContext(), 64), 2, "", label_122);
    int64_139->addIncoming(int64__pre_vec[i], label___crit_edge);
    int64_139->addIncoming(int64_124_vec[i], label_120);
    int64_139_vec.insert(int64_139_vec.begin(), int64_139);
  }

  for (size_t i = 0; i < aggr_col_count; ++i) {
    auto idx_lv = ConstantInt::get(IntegerType::get(mod->getContext(), 32), i);
    auto out_gep = GetElementPtrInst::CreateInBounds(ptr_out, idx_lv, "", label_122);
    auto ptr_140 = new LoadInst(out_gep, "", false, label_122);
    ptr_140->setAlignment(8);
    auto slot_idx = BinaryOperator::CreateAdd(
        group_buff_idx, BinaryOperator::CreateMul(frag_idx, int32_127, "", label_122), "", label_122);
    auto ptr_141 = GetElementPtrInst::CreateInBounds(ptr_140, slot_idx, "", label_122);
    StoreInst* void_142 = new StoreInst(int64_139_vec[i], ptr_141, false, label_122);
    void_142->setAlignment(8);
  }

  ReturnInst::Create(mod->getContext(), label_122);

  // Resolve Forward References
  fwdref_133->replaceAllUsesWith(int64_135);
  delete fwdref_133;

  if (verifyFunction(*func_query_template)) {
    LOG(FATAL) << "Generated invalid code. ";
  }

  return func_query_template;
}

llvm::Function* query_group_by_template(llvm::Module* mod,
                                        const bool is_nested,
                                        const bool hoist_literals,
                                        const QueryMemoryDescriptor& query_mem_desc,
                                        const ExecutorDeviceType device_type,
                                        const bool check_scan_limit) {
  using namespace llvm;

  auto func_pos_start = pos_start(mod);
  CHECK(func_pos_start);
  auto func_pos_step = pos_step(mod);
  CHECK(func_pos_step);
  auto func_group_buff_idx = group_buff_idx(mod);
  CHECK(func_group_buff_idx);
  auto func_init_group_by_buffer = init_group_by_buffer(mod);
  auto func_row_process = row_process(mod, 0, is_nested, hoist_literals);
  CHECK(func_row_process);
  auto func_init_shared_mem = query_mem_desc.sharedMemBytes(device_type) ? mod->getFunction("init_shared_mem")
                                                                         : mod->getFunction("init_shared_mem_nop");
  CHECK(func_init_shared_mem);
  auto func_write_back =
      query_mem_desc.sharedMemBytes(device_type) ? mod->getFunction("write_back") : mod->getFunction("write_back_nop");
  CHECK(func_write_back);

  PointerType* PointerTy_1 = PointerType::get(IntegerType::get(mod->getContext(), 8), 0);
  PointerType* PointerTy_6 = PointerType::get(IntegerType::get(mod->getContext(), 64), 0);
  PointerType* PointerTy_9 = PointerType::get(PointerTy_1, 0);

  std::vector<Type*> FuncTy_12_args;
  FuncTy_12_args.push_back(PointerTy_9);
  if (hoist_literals) {
    FuncTy_12_args.push_back(PointerTy_1);
  }
  FuncTy_12_args.push_back(PointerTy_6);
  FuncTy_12_args.push_back(PointerTy_6);
  FuncTy_12_args.push_back(PointerTy_6);
  FuncTy_12_args.push_back(PointerTy_6);
  PointerType* PointerTy_13 = PointerType::get(PointerTy_6, 0);

  FuncTy_12_args.push_back(PointerTy_13);
  FuncTy_12_args.push_back(PointerTy_13);
  FuncTy_12_args.push_back(IntegerType::get(mod->getContext(), 32));
  FuncTy_12_args.push_back(IntegerType::get(mod->getContext(), 64));
  FuncTy_12_args.push_back(PointerType::get(IntegerType::get(mod->getContext(), 32), 0));

  FunctionType* FuncTy_12 = FunctionType::get(
      /*Result=*/Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_12_args,
      /*isVarArg=*/false);

  auto query_group_by_template_name = unique_name("query_group_by_template", is_nested);
  auto func_query_group_by_template = mod->getFunction(query_group_by_template_name);
  CHECK(!func_query_group_by_template);

  func_query_group_by_template = Function::Create(
      /*Type=*/FuncTy_12,
      /*Linkage=*/GlobalValue::ExternalLinkage,
      /*Name=*/"query_group_by_template",
      mod);

  func_query_group_by_template->setCallingConv(CallingConv::C);

  AttributeSet func_query_group_by_template_PAL;
  {
    SmallVector<AttributeSet, 4> Attrs;
    AttributeSet PAS;
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadNone);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 1U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadOnly);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 2U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadNone);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 3U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::ReadOnly);
      B.addAttribute(Attribute::NoCapture);
      PAS = AttributeSet::get(mod->getContext(), 4U, B);
    }

    Attrs.push_back(PAS);
    {
      AttrBuilder B;
      B.addAttribute(Attribute::UWTable);
      PAS = AttributeSet::get(mod->getContext(), ~0U, B);
    }

    Attrs.push_back(PAS);

    func_query_group_by_template_PAL = AttributeSet::get(mod->getContext(), Attrs);
  }
  func_query_group_by_template->setAttributes(func_query_group_by_template_PAL);

  Function::arg_iterator args = func_query_group_by_template->arg_begin();
  Value* ptr_byte_stream_143 = args++;
  ptr_byte_stream_143->setName("byte_stream");
  Value* literals{nullptr};
  if (hoist_literals) {
    literals = args++;
    literals->setName("literals");
  }
  Value* ptr_row_count_ptr_144 = args++;
  ptr_row_count_ptr_144->setName("row_count_ptr");
  Value* frag_row_off_ptr = args++;
  frag_row_off_ptr->setName("frag_row_off_ptr");
  Value* ptr_max_matched_ptr = args++;
  ptr_max_matched_ptr->setName("max_matched_ptr");
  Value* ptr_agg_init_val_145 = args++;
  ptr_agg_init_val_145->setName("agg_init_val");
  Value* ptr_group_by_buffers = args++;
  ptr_group_by_buffers->setName("group_by_buffers");
  Value* ptr_small_groups_buffer = args++;
  ptr_small_groups_buffer->setName("small_groups_buffer");
  Value* frag_idx = args++;
  frag_idx->setName("frag_idx");
  Value* ptr_join_hash_table = args++;
  ptr_join_hash_table->setName("join_hash_table");
  Value* ptr_error_code = args++;
  ptr_error_code->setName("error_code");

  BasicBlock* label_146 = BasicBlock::Create(mod->getContext(), "", func_query_group_by_template, 0);
  BasicBlock* label__lr_ph_147 = BasicBlock::Create(mod->getContext(), ".lr.ph", func_query_group_by_template, 0);
  BasicBlock* label_148 = BasicBlock::Create(mod->getContext(), "", func_query_group_by_template, 0);
  BasicBlock* label___crit_edge_loopexit =
      BasicBlock::Create(mod->getContext(), "._crit_edge.loopexit", func_query_group_by_template, 0);
  BasicBlock* label___crit_edge_149 =
      BasicBlock::Create(mod->getContext(), "._crit_edge", func_query_group_by_template, 0);

  // Block  (label_146)
  LoadInst* int64_150 = new LoadInst(ptr_row_count_ptr_144, "", false, label_146);
  int64_150->setAlignment(8);
  LoadInst* frag_row_off = new LoadInst(frag_row_off_ptr, "", false, label_146);
  frag_row_off->setAlignment(8);
  LoadInst* max_matched = new LoadInst(ptr_max_matched_ptr, "", false, label_146);
  int64_150->setAlignment(8);
  auto crt_matched_ptr = new AllocaInst(IntegerType::get(mod->getContext(), 64), "crt_matched", label_146);
  new StoreInst(ConstantInt::get(IntegerType::get(mod->getContext(), 64), 0), crt_matched_ptr, false, label_146);
  CallInst* int32_151 = CallInst::Create(func_pos_start, "", label_146);
  int32_151->setCallingConv(CallingConv::C);
  int32_151->setTailCall(true);
  AttributeSet int32_151_PAL;
  int32_151->setAttributes(int32_151_PAL);

  CallInst* int32_152 = CallInst::Create(func_pos_step, "", label_146);
  int32_152->setCallingConv(CallingConv::C);
  int32_152->setTailCall(true);
  AttributeSet int32_152_PAL;
  int32_152->setAttributes(int32_152_PAL);

  CallInst* group_buff_idx = CallInst::Create(func_group_buff_idx, "", label_146);
  group_buff_idx->setCallingConv(CallingConv::C);
  group_buff_idx->setTailCall(true);
  AttributeSet group_buff_idx_PAL;
  group_buff_idx->setAttributes(group_buff_idx_PAL);

  CastInst* int64_153 = new SExtInst(int32_151, IntegerType::get(mod->getContext(), 64), "", label_146);
  GetElementPtrInst* ptr_154 = GetElementPtrInst::Create(ptr_group_by_buffers, group_buff_idx, "", label_146);
  LoadInst* ptr_155 = new LoadInst(ptr_154, "", false, label_146);
  ptr_155->setAlignment(8);
  LoadInst* small_ptr_155{nullptr};
  if (query_mem_desc.getSmallBufferSizeBytes()) {
    auto small_ptr_154 = GetElementPtrInst::Create(ptr_small_groups_buffer, group_buff_idx, "", label_146);
    small_ptr_155 = new LoadInst(small_ptr_154, "", false, label_146);
    small_ptr_155->setAlignment(8);
  }
  CallInst::Create(
      func_init_group_by_buffer,
      std::vector<llvm::Value*>{
          ptr_155,
          ptr_agg_init_val_145,
          ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.entry_count),
          ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.group_col_widths.size()),
          ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.agg_col_widths.size()),
      },
      "",
      label_146);
  auto shared_mem_bytes_lv =
      ConstantInt::get(IntegerType::get(mod->getContext(), 32), query_mem_desc.sharedMemBytes(device_type));
  auto ptr_156 =
      CallInst::Create(func_init_shared_mem, std::vector<llvm::Value*>{ptr_155, shared_mem_bytes_lv}, "", label_146);
  ICmpInst* int1_156 = new ICmpInst(*label_146, ICmpInst::ICMP_SLT, int64_153, int64_150, "");
  BranchInst::Create(label__lr_ph_147, label___crit_edge_149, int1_156, label_146);

  // Block .lr.ph (label__lr_ph_147)
  CastInst* int64_158 = new SExtInst(int32_152, IntegerType::get(mod->getContext(), 64), "", label__lr_ph_147);
  BranchInst::Create(label_148, label__lr_ph_147);

  // Block  (label_148)
  Argument* fwdref_161 = new Argument(IntegerType::get(mod->getContext(), 64));
  PHINode* int64_pos_01_160 = PHINode::Create(IntegerType::get(mod->getContext(), 64), 2, "pos.01", label_148);
  int64_pos_01_160->addIncoming(int64_153, label__lr_ph_147);
  int64_pos_01_160->addIncoming(fwdref_161, label_148);

  std::vector<Value*> void_162_params;
  void_162_params.push_back(ptr_156);
  if (query_mem_desc.getSmallBufferSizeBytes()) {
    void_162_params.push_back(small_ptr_155);
  } else {
    void_162_params.push_back(Constant::getNullValue(PointerTy_6));
  }
  void_162_params.push_back(crt_matched_ptr);
  void_162_params.push_back(ptr_agg_init_val_145);
  void_162_params.push_back(int64_pos_01_160);
  void_162_params.push_back(frag_row_off);
  void_162_params.push_back(ptr_row_count_ptr_144);
  if (hoist_literals) {
    CHECK(literals);
    void_162_params.push_back(literals);
  }
  CallInst* void_162 = CallInst::Create(func_row_process, void_162_params, "", label_148);
  void_162->setCallingConv(CallingConv::C);
  void_162->setTailCall(true);
  AttributeSet void_162_PAL;
  void_162->setAttributes(void_162_PAL);

  BinaryOperator* int64_163 = BinaryOperator::Create(Instruction::Add, int64_pos_01_160, int64_158, "", label_148);
  ICmpInst* int1_164 = new ICmpInst(*label_148, ICmpInst::ICMP_SLT, int64_163, int64_150, "");
  if (check_scan_limit) {
    ICmpInst* limit_not_reached = new ICmpInst(
        *label_148, ICmpInst::ICMP_SLT, new LoadInst(crt_matched_ptr, "", false, label_148), max_matched, "");
    BranchInst::Create(label_148,
                       label___crit_edge_loopexit,
                       BinaryOperator::Create(BinaryOperator::And, int1_164, limit_not_reached, "", label_148),
                       label_148);
  } else {
    BranchInst::Create(label_148, label___crit_edge_loopexit, int1_164, label_148);
  }

  // Block ._crit_edge.loopexit (label___crit_edge_loopexit)
  BranchInst::Create(label___crit_edge_149, label___crit_edge_loopexit);

  // Block ._crit_edge (label___crit_edge_149)
  CallInst::Create(
      func_write_back, std::vector<Value*>{ptr_155, ptr_156, shared_mem_bytes_lv}, "", label___crit_edge_149);
  ReturnInst::Create(mod->getContext(), label___crit_edge_149);

  // Resolve Forward References
  fwdref_161->replaceAllUsesWith(int64_163);
  delete fwdref_161;

  if (verifyFunction(*func_query_group_by_template)) {
    LOG(FATAL) << "Generated invalid code. ";
  }

  return func_query_group_by_template;
}

std::string unique_name(const char* base_name, const bool is_nested) {
  char full_name[128] = {0};
  snprintf(full_name, sizeof(full_name), "%s_%u", base_name, static_cast<unsigned>(is_nested));
  return full_name;
}
