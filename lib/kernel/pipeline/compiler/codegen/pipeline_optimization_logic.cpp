#include "../pipeline_compiler.hpp"
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#ifndef NDEBUG
#include <llvm/IR/Verifier.h>
#endif
#include <llvm/Transforms/Scalar/MemCpyOptimizer.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/MemorySSA.h>

namespace kernel {


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief replacePhiCatchBlocksWith
 *
 * replace the phi catch with the actual exit blocks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::replacePhiCatchWithCurrentBlock(KernelBuilder & b, BasicBlock *& toReplace, BasicBlock * const phiContainer) {
    // NOTE: not all versions of LLVM seem to have BasicBlock::replacePhiUsesWith or PHINode::replaceIncomingBlockWith.
    // This code could be made to use those instead.

    assert (toReplace);

    BasicBlock * const to = b.GetInsertBlock();

    for (Instruction & inst : *phiContainer) {
        if (LLVM_LIKELY(isa<PHINode>(inst))) {
            PHINode & pn = cast<PHINode>(inst);
            for (unsigned i = 0; i != pn.getNumIncomingValues(); ++i) {
                if (pn.getIncomingBlock(i) == toReplace) {
                    pn.setIncomingBlock(i, to);
                }
            }
        } else {
            break;
        }
    }

    if (!toReplace->empty()) {
        Instruction * toMove = &toReplace->front();
        while (toMove) {
            Instruction * const next = toMove->getNextNode();
            toMove->removeFromParent();
            toMove->insertAfter(&to->back());
            toMove = next;
        }
    }


    toReplace->eraseFromParent();
    toReplace = to;

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::runOptimizationPasses(KernelBuilder & b) {

    // To make sure the optimizations aren't hiding an error, first run the verifier
    // detect any possible errors prior to optimizing it.

    Module * const m = b.getModule();

    #ifndef NDEBUG
    SmallVector<char, 256> tmp;
    raw_svector_ostream msg(tmp);
    bool BrokenDebugInfo = false;
    if (LLVM_UNLIKELY(verifyModule(*m, &msg, &BrokenDebugInfo))) {
        m->print(errs(), nullptr);
        report_fatal_error(StringRef(msg.str()));
    }
    #endif
    simplifyPhiNodes(m);

    FunctionAnalysisManager FAM;
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    FAM.registerPass([&] { return AssumptionAnalysis(); });
    FAM.registerPass([&] { return TargetIRAnalysis(); });
    FAM.registerPass([&] { return TargetLibraryAnalysis(); });
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return AAManager(); });
    FAM.registerPass([&] { return MemorySSAAnalysis(); });

    FunctionPassManager FPM;
    FPM.addPass(DCEPass()); // Eliminate any trivially dead code
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(EarlyCSEPass());
    FPM.addPass(MemCpyOptPass());

    Module * M = b.getModule();
    for (Function & F : *M) {
        if (F.empty()) continue;
        FPM.run(F, FAM);
    }

    simplifyPhiNodes(m);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief simplifyPhiNodes
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::simplifyPhiNodes(Module * const m) const {

    // LLVM is not aggressive enough with how it deals with phi nodes. To ensure that
    // we collapse every phi node in which all incoming values are identical into the
    // incoming value, we execute the following mini optimization pass.

    // TODO: check the newer versions of LLVM to see if any can do this now.

    SmallVector<BasicBlock *, 16> preds;
    SmallVector<Value *, 16> value;

    for (Function & f : m->getFunctionList()) {
        bool anyPhis = false;

        for (BasicBlock & bb : f) {

            preds.assign(pred_begin(&bb), pred_end(&bb));
            const auto n = preds.size();
            value.resize(n);

            Instruction * inst = &bb.front();
            while (isa<PHINode>(inst)) {
                PHINode * const phi = cast<PHINode>(inst);
                #ifndef NDEBUG
                if (LLVM_UNLIKELY(phi->getNumIncomingValues() != n || n == 0)) {
                    bb.print(errs(), nullptr, true, false);
                    errs() << "\n\nIllegal PHINode: ";
                    phi->print(errs(), true);
                }
                #endif
                inst = inst->getNextNode();
                if (LLVM_LIKELY(phi->hasNUsesOrMore(1))) {
                    Value * const value = phi->getIncomingValue(0);
                    assert (value);
                    const auto n = phi->getNumIncomingValues();
                    for (unsigned i = 1; i != n; ++i) {
                        Value * const op = phi->getIncomingValue(i);
                        assert (op);
                        if (LLVM_LIKELY(op != value)) {
                            goto keep_phi_node;
                        }
                    }
                    phi->replaceAllUsesWith(value);
                }

                RecursivelyDeleteDeadPHINode(phi);
                continue;
                // ----------------------------------------------------------------------------------
                //  canonicalize the phi node ordering for the eliminate duplicate phi node function
                // ----------------------------------------------------------------------------------
keep_phi_node:  bool canonicalize = false;
                for (unsigned i = 0; i != n; ++i) {
                    const auto f = std::find(preds.begin(), preds.end(), phi->getIncomingBlock(i));
                    assert ("phi-node has invalid incoming block?" && f != preds.end());
                    const auto j = std::distance(preds.begin(), f);
                    canonicalize |= (j != i);
                    value[j] = phi->getIncomingValue(i);
                }
                if (canonicalize) {
                    for (unsigned i = 0; i != n; ++i) {
                        phi->setIncomingBlock(i, preds[i]);
                        phi->setIncomingValue(i, value[i]);
                    }
                }
                anyPhis = true;
            }
            if (LLVM_LIKELY(anyPhis)) {
                EliminateDuplicatePHINodes(&bb);
            }
        }
    }
}

}
