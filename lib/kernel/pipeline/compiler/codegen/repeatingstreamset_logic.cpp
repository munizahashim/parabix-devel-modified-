#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateMetaDataForRepeatingStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateMetaDataForRepeatingStreamSets(BuilderRef b) {

    const PipelineKernel * const pk = cast<PipelineKernel>(mTarget);
    const auto & kernels = pk->getKernels();
    const auto m = kernels.size();

    flat_set<const RepeatingStreamSet *> touched;

    std::vector<Constant *> maxStrides;

    // the ordering of the kernels may differ between the input ordering of the
    // pipeline kernel and what was actually compiled by the program.

    for (unsigned i = 0; i < m; ++i) {
        const Kernel * const kernel = kernels[i].Object;
        const auto m = kernel->getNumOfStreamInputs();
        if (LLVM_UNLIKELY(kernel->generatesDynamicRepeatingStreamSets())) {
            maxStrides.push_back(b->getSize(MaximumNumOfStrides[i]));
        }
        for (unsigned i = 0; i != m; ++i) {
            const StreamSet * const input = kernel->getInputStreamSet(i);
            if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(input))) {
                if (cast<RepeatingStreamSet>(input)->isDynamic()) {
                    // Since the kernel/streamset graph relationships is part of a pipeline's
                    // signature, we do not need an entry for every shared streamset.

                    if (touched.emplace(cast<RepeatingStreamSet>(input)).second) {
                        Rational ub{0U};
                        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
                            const RelationshipNode & rn = mStreamGraph[streamSet];
                            assert (rn.Type == RelationshipNode::IsRelationship);
                            if (rn.Relationship == input) {
                                for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                                    const auto consumer = target(e, mBufferGraph);
                                    assert (consumer >= FirstKernel && consumer <= PipelineOutput);
                                    const BufferPort & bp = mBufferGraph[e];
                                    const auto m = MaximumNumOfStrides[consumer];
                                    ub = std::max(ub, bp.Maximum * m);
                                }
                                assert (ub.denominator() == 1);
//                                const BufferNode & bn = mBufferGraph[streamSet];
//                                if (bn.OverflowCapacity) {
//                                    ub += (bn.OverflowCapacity * b->getBitBlockWidth());
//                                }
                                break;
                            }
                        }
                        assert (ub.denominator() == 1);
                        maxStrides.push_back(b->getSize(ub.numerator()));
                    }
                }
            }
        }
    }

    Module * const module = mTarget->getModule();
    NamedMDNode * const md = module->getOrInsertNamedMetadata("rsl");
    assert (md->getNumOperands() == 0);
    Constant * ar = ConstantArray::get(ArrayType::get(b->getSizeTy(), maxStrides.size()), maxStrides);
    md->addOperand(MDNode::get(module->getContext(), {ConstantAsMetadata::get(ar)}));

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getMaximumNumOfStridesForRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
Constant * PipelineCompiler::getMaximumNumOfStridesForRepeatingStreamSet(BuilderRef b, const unsigned streamSet) const {
    Rational ub{0U};
    for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
        const auto consumer = target(e, mBufferGraph);
        assert (consumer >= FirstKernel && consumer <= PipelineOutput);
        const auto m = MaximumNumOfStrides[consumer] ;
        const BufferPort & bp = mBufferGraph[e];
        ub = std::max(ub, bp.Maximum * m);
    }
    assert (ub.denominator() == 1);
    return b->getSize(ub.numerator());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readsRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::readsRepeatingStreamSet() const {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isConstant()) return true;
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindRepeatingStreamSetInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::bindRepeatingStreamSetInitializationArguments(BuilderRef b, ArgIterator & arg, const ArgIterator & arg_end) const {

    // NOTE: the arguments here will be relative to the external program ordering and not the
    // pipeline's ordering.

    for (const auto streamSet : DynamicRepeatingStreamSetId) {

        assert (arg != arg_end);
        Value * const addr = &*arg++;
        assert (arg != arg_end);
        Value * const runLength = &*arg++;

        if (streamSet) {
            const auto handleName = REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet);
            Value * const handle = b->getScalarFieldPtr(handleName);
            const BufferNode & bn = mBufferGraph[streamSet];
            #ifndef NDEBUG
            const RelationshipNode & rn = mStreamGraph[streamSet];
            assert (rn.Type == RelationshipNode::IsRelationship);
            assert (isa<RepeatingStreamSet>(rn.Relationship));
            assert (cast<RepeatingStreamSet>(rn.Relationship)->isDynamic());
            #endif
            // external buffers already have a buffer handle
            RepeatingBuffer * const buffer = cast<RepeatingBuffer>(bn.Buffer);
            buffer->setHandle(handle);
            Value * const ba = b->CreatePointerCast(addr, buffer->getPointerType());
            buffer->setBaseAddress(b, ba);
            buffer->setModulus(runLength);            
            const auto lengthName = REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet);
            b->setScalarField(lengthName, runLength);
        }

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateGlobalDataForRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateGlobalDataForRepeatingStreamSet(BuilderRef b, const unsigned streamSet, Value * const expectedNumOfStrides) {
    const BufferNode & bn = mBufferGraph[streamSet];
    RepeatingBuffer * const buffer = cast<RepeatingBuffer>(bn.Buffer);

    const auto handleName = REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet);
    Value * const handle = b->getScalarFieldPtr(handleName);
    buffer->setHandle(handle);

    const RelationshipNode & rn = mStreamGraph[streamSet];
    assert (rn.Type == RelationshipNode::IsRelationship);
    const RepeatingStreamSet * const ss = cast<RepeatingStreamSet>(rn.Relationship);

    if (ss->isDynamic()) {
        assert (isFromCurrentFunction(b, buffer->getBaseAddress(b)));
    } else {
        Rational ub{1U};
        for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(e, mBufferGraph);
            assert (consumer >= FirstKernel && consumer <= PipelineOutput);
            const auto m = MaximumNumOfStrides[consumer] + 1;
            const BufferPort & bp = mBufferGraph[e];
            ub = std::max(ub, bp.Maximum * m);
        }
        assert (ub.denominator() == 1);
        const auto maxStrideLength = ub.numerator();
        auto info = cast<PipelineKernel>(mTarget)->createRepeatingStreamSet(b, ss, maxStrideLength);
        Value * const ba = b->CreatePointerCast(info.StreamSet, buffer->getPointerType());
        buffer->setBaseAddress(b, ba);
        buffer->setModulus(info.RunLength);
    }

#if 0

    BasicBlock * const copyAndExpandGlobal = b->CreateBasicBlock();
    BasicBlock * const copyAndExpandGlobalLoop = b->CreateBasicBlock();
    BasicBlock * const exit = b->CreateBasicBlock();

    // if we scale our expected
    ConstantInt * const sz_ZERO = b->getSize(0);
    ConstantInt * const sz_ONE = b->getSize(1);
    PointerType * const arrPtrTy = arrTy->getPointerTo();
    BasicBlock * const entryBlock = b->GetInsertBlock();
    Value * const noMemCpyNeeded = b->CreateICmpEQ(expectedNumOfStrides, sz_ONE);
    b->CreateCondBr(noMemCpyNeeded, exit, copyAndExpandGlobal);

    b->SetInsertPoint(copyAndExpandGlobal);
    ConstantInt * const baseAdditionalLength = b->getSize(additionalStrides);
    Value * const expandedAdditionalLength = b->CreateMul(expectedNumOfStrides, baseAdditionalLength);
    Value * const totalLength = b->CreateAdd(baseLength, expandedAdditionalLength);
    Value * const originalLength = b->CreateAdd(baseLength, baseAdditionalLength);
    Value * const arrTySize = ConstantExpr::getSizeOf(arrTy);
    Value * const bytesNeeded = b->CreateMul(arrTySize, totalLength);
    Value * const data = b->CreateAlignedMalloc(bytesNeeded, align);
    Value * const array = b->CreatePointerCast(data, arrPtrTy);
    b->CreateMemCpy(data, patternData, b->CreateMul(arrTySize, originalLength), align);
    b->setScalarField(REPEATING_STREAMSET_MALLOCED_DATA_PREFIX + std::to_string(streamSet), data);
    b->CreateBr(copyAndExpandGlobalLoop);

    b->SetInsertPoint(copyAndExpandGlobalLoop);
    PHINode * const copyIndexPhi = b->CreatePHI(b->getSizeTy(), 2);
    copyIndexPhi->addIncoming(sz_ZERO, copyAndExpandGlobal);
    FixedArray<Value *, 2> offset;
    offset[0] = sz_ZERO;
    offset[1] = b->CreateAdd(b->CreateMul(copyIndexPhi, baseLength), baseAdditionalLength);
    Value * const from = b->CreateGEP(array, offset);
    Value * const nextCopyIndex = b->CreateAdd(copyIndexPhi, sz_ONE);
    offset[1] = b->CreateAdd(b->CreateMul(nextCopyIndex, baseLength), baseAdditionalLength);
    Value * const to = b->CreateGEP(array, offset);
    b->CreateMemCpy(to, from, b->CreateMul(arrTySize, baseLength), align);
    copyIndexPhi->addIncoming(nextCopyIndex, copyAndExpandGlobalLoop);
    Value * const notDone = b->CreateICmpNE(nextCopyIndex, expectedNumOfStrides);
    b->CreateCondBr(notDone, copyAndExpandGlobalLoop, exit);

    b->SetInsertPoint(exit);
    PHINode * const addr = b->CreatePHI(arrPtrTy, 2);
    addr->addIncoming(patternData, entryBlock);
    addr->addIncoming(array, copyAndExpandGlobalLoop);
    Value * const ba = b->CreatePointerCast(addr, buffer->getPointerType());
    buffer->setBaseAddress(b, ba);
#endif
}

void PipelineCompiler::addRepeatingStreamSetBufferProperties(BuilderRef b) {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isConstant())) {
            auto & S = mStreamGraph[streamSet];
            assert (S.Type == RelationshipNode::IsRelationship);
            assert (isa<RepeatingStreamSet>(S.Relationship));

            Type * const handleTy = bn.Buffer->getHandleType(b);
            mTarget->addInternalScalar(handleTy,
                REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet),
                                       getCacheLineGroupId(PipelineOutput));
            if (cast<RepeatingStreamSet>(S.Relationship)->isDynamic()) {
                mTarget->addInternalScalar(b->getSizeTy(),
                    REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet),
                                           getCacheLineGroupId(PipelineOutput));
            }
//            mTarget->addInternalScalar(b->getVoidPtrTy(),
//                REPEATING_STREAMSET_MALLOCED_DATA_PREFIX + std::to_string(streamSet),
//                                       getCacheLineGroupId(PipelineOutput));
        }
    }
}

void PipelineCompiler::deallocateRepeatingBuffers(BuilderRef b) {
//    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
//        const BufferNode & bn = mBufferGraph[streamSet];
//        if (LLVM_UNLIKELY(bn.isConstant())) {
//            const auto bufferName = REPEATING_STREAMSET_MALLOCED_DATA_PREFIX + std::to_string(streamSet);
//            b->CreateFree(b->getScalarField(bufferName));
//        }
//    }
}

}
