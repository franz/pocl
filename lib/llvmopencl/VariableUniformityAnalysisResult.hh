// Implementation for VariableUniformityAnalysis function pass.
//
// Copyright (c) 2023 Michal Babej / Intel Finland Oy
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

namespace llvm {
class Function;
class LoopInfo;
class PostDominatorTree;
class BasicBlock;
class Value;
class Module;
class Loop;
class DominatorTree;

class PreservedAnalyses;
template <typename, typename...> class AnalysisManager;

// TODO this is ugly. Probably should just include CycleInfo
template <typename> class GenericCycleInfo;
template <typename> class GenericSSAContext;
using SSAContext = class GenericSSAContext<Function>;
using CycleInfo = GenericCycleInfo<SSAContext>;

} // namespace llvm

//#include <llvm/IR/CycleInfo.h>
#include <map>

namespace pocl {

class UA_Impl;

using UniformityIndex = std::map<const llvm::Value *, bool>;
using UniformityCache = std::map<llvm::Function *, UniformityIndex>;
using ValueDivergenceMap = std::map<const llvm::Value *, bool>;


class VariableUniformityAnalysisResult {

public:
  VariableUniformityAnalysisResult() : Impl(nullptr) {}
  ~VariableUniformityAnalysisResult();

  VariableUniformityAnalysisResult(const VariableUniformityAnalysisResult &) = delete;
  VariableUniformityAnalysisResult &operator=(const VariableUniformityAnalysisResult &) = delete;

  VariableUniformityAnalysisResult(VariableUniformityAnalysisResult &&R) {
    *this = std::move(R);
  }

  VariableUniformityAnalysisResult &operator=(VariableUniformityAnalysisResult &&R) {
    Impl = R.Impl;
    R.Impl = nullptr;
    uniformityCache_ = std::move(R.uniformityCache_);
    return *this;
  }

  bool runOnFunction(llvm::Function &F, llvm::LoopInfo &LI,
                     llvm::PostDominatorTree &PDT, llvm::CycleInfo &CI, llvm::DominatorTree &DT);
  bool isUniform(llvm::Function *F, llvm::Value *V);
  void setUniform(llvm::Function *F, llvm::Value *V, bool isUniform = true);
  void analyzeBBDivergence(llvm::Function *F, llvm::BasicBlock *BB,
                           llvm::BasicBlock *PreviousUniformBB,
                           llvm::PostDominatorTree &PDT);

  bool shouldBePrivatized(llvm::Function *F, llvm::Value *Val);
  bool doFinalization(llvm::Module &M);
  void markInductionVariables(llvm::Function &F, llvm::Loop &L);

  // TODO this could be wrong
#if LLVM_MAJOR >= MIN_LLVM_NEW_PASSMANAGER
  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses PA,
                  llvm::AnalysisManager<llvm::Function>::Invalidator &Inv);
#endif

private:
  bool isUniformityAnalyzed(llvm::Function *F, llvm::Value *V) const;
  bool isUniform2(llvm::Function *F, llvm::Value *V);

  mutable UniformityCache uniformityCache_;
  UA_Impl *Impl;
};

} // namespace pocl
