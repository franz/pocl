// LLVM module pass that ensures integere division by zero does not result in
// SIGFPE sent to the process
//
// Copyright (c) 2024 Michal Babej / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Pass.h>

#include "DivByZero.h"
#include "LLVMUtils.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <set>
#include <string>

#include "pocl_llvm_api.h"

//#define DEBUG_DIV_BY_ZERO

#define PASS_NAME "div-by-zero"
#define PASS_CLASS pocl::DivByZero
#define PASS_DESC "Handle integer division by zero in kernel code"

namespace pocl {

using namespace llvm;

static bool handleDivByZero(llvm::Function &F) {
  IRBuilder<> Builder(F.getContext());

  // TODO
  ValueToValueMapTy AlreadyFixedMap;

  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      BinaryOperator *BinI = dyn_cast<BinaryOperator>(BI++);
      if (BinI == nullptr)
        continue;
      if (!BinI->isIntDivRem())
        continue;
      if (dyn_cast<ConstantInt>(BinI->getOperand(1)))
        continue;
      Value *Divisor = BinI->getOperand(1);

      std::cerr << "Fixing DivByZero in second operand of Inst: \n";
      BinI->dump();

      BasicBlock *BB = BinI->getParent();
      BasicBlock *NewBBwithDiv = BB->splitBasicBlock(BinI, "div.by.zero");
      BB->getTerminator()->eraseFromParent();
      BasicBlock *SetDivisor = BasicBlock::Create(F.getContext(),
                                                  "set.divisor",
                                                  &F, NewBBwithDiv);

      Builder.SetInsertPoint(BB);
      Value *Compared = Builder.CreateCmp(
            CmpInst::Predicate::ICMP_EQ,
                Divisor, ConstantInt::get(Divisor->getType(), 0));
      Builder.CreateCondBr(Compared, SetDivisor, NewBBwithDiv);

      Builder.SetInsertPoint(SetDivisor);
      Builder.CreateBr(NewBBwithDiv);

      Builder.SetInsertPoint(BinI);
      PHINode *PhiVal = Builder.CreatePHI(Divisor->getType(), 2);
      PhiVal->setIncomingBlock(0, SetDivisor);
      PhiVal->setIncomingValue(0, ConstantInt::get(Divisor->getType(), 0));
      PhiVal->setIncomingBlock(1, BB);
      PhiVal->setIncomingValue(1, Divisor);
      BinI->setOperand(1, PhiVal);

      Changed = true;
    }
  }

  F.dump();
  return Changed;
}

llvm::PreservedAnalyses DivByZero::run(llvm::Function &F,
                                       llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return handleDivByZero(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
