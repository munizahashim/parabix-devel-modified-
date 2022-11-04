#include "../pipeline_compiler.hpp"

namespace kernel {

void PipelineCompiler::generateGlobalDataForRepeatingStreamSet(BuilderRef b, const unsigned streamSet, Value * const expectedNumOfStrides) {

    const RelationshipNode & rn = mStreamGraph[streamSet];
    assert (rn.Type == RelationshipNode::IsRelationship);
    const RepeatingStreamSet * const ss = cast<RepeatingStreamSet>(rn.Relationship);

    const auto fieldWidth = ss->getFieldWidth();
    const auto numElements = ss->getNumElements();
    const auto blockWidth = b->getBitBlockWidth();

    const auto maxVal = (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL;

    uint64_t patternLength = blockWidth;
    for (unsigned i = 0; i < numElements; ++i) {
        const auto & vec = ss->getPattern(i);
        const auto L = vec.size();
        if (LLVM_UNLIKELY(L == 0)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream msg(tmp);
            const auto input = out_edge(streamSet, mBufferGraph);
            const BufferPort & bp = mBufferGraph[input];
            const Kernel * const kernelObj = getKernel(target(input, mBufferGraph));
            const Binding & inputPort = kernelObj->getInputStreamSetBinding(bp.Port.Number);
            msg << "Zero-length repeating streamset elements are not permitted ("
                << kernelObj << "." << inputPort.getName() << " element " << i
                << ")";
            report_fatal_error(msg.str());
        }
        patternLength = boost::lcm<uint64_t>(patternLength, L);
        for (auto v : vec) {
            if (LLVM_UNLIKELY(v > maxVal)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                const auto input = out_edge(streamSet, mBufferGraph);
                const BufferPort & bp = mBufferGraph[input];
                const Kernel * const kernelObj = getKernel(target(input, mBufferGraph));
                const Binding & inputPort = kernelObj->getInputStreamSetBinding(bp.Port.Number);
                msg << "Repeating streamset value " << v
                    << " for " << kernelObj << "." << inputPort.getName()
                    << " exceeds a " << fieldWidth << "-bit value";
                report_fatal_error(msg.str());
            }
        }
    }

    assert ((patternLength % blockWidth) == 0 && "trivially true");

    const auto runLength = (patternLength / blockWidth);

    auto additionalStrides = 1U;
    for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {

        const auto consumer = target(e, mBufferGraph);
        assert (consumer >= FirstKernel && consumer <= PipelineOutput);
        const auto m = MaximumNumOfStrides[consumer];

        errs() << "m=" << m << "\n";

        const BufferPort & bp = mBufferGraph[e];

        errs() << "max=" << bp.Maximum.numerator() << "/" << bp.Maximum.denominator() << "\n";

        const auto s = bp.Maximum * Rational{m, blockWidth};
        assert (s.denominator() == 1);
        additionalStrides = std::max<unsigned>(additionalStrides, ceiling(s));
    }

    std::vector<Constant *> dataVectorArray(runLength + additionalStrides);

    FixedVectorType * const vecTy = b->getBitBlockType();
    IntegerType * const intTy = cast<IntegerType>(vecTy->getScalarType());
    const auto laneWidth = intTy->getIntegerBitWidth();
    const auto numLanes = blockWidth / laneWidth;
    ArrayType * const elementTy = ArrayType::get(vecTy, fieldWidth);
    ArrayType * const streamSetTy = ArrayType::get(elementTy, numElements);

    SmallVector<Constant *, 16> tmp(numLanes);
    SmallVector<Constant *, 16> tmp2(fieldWidth);
    SmallVector<Constant *, 16> tmp3(numElements);

    SmallVector<uint64_t, 16> elementPos(numElements, 0);

    for (unsigned r = 0; r < runLength; ++r) {
        for (unsigned p = 0; p < numElements; ++p) {
            const auto & vec = ss->getPattern(p);
            const auto L = vec.size();
            for (uint64_t i = 0; i < fieldWidth; ++i) {
                for (uint64_t j = 0; j < numLanes; ++j) {
                    uint64_t V = 0;
                    for (uint64_t k = 0; k != laneWidth; k += fieldWidth) {
                        auto & pos = elementPos[p];
                        const auto v = vec[pos % L];
                        V |= (v << k);
                        ++pos;
                    }
                    tmp[j] = ConstantInt::get(intTy, V, false);
                }
                tmp2[i] = ConstantVector::get(tmp);
            }
            tmp3[p] = ConstantArray::get(elementTy, tmp2);
        }
        dataVectorArray[r] = ConstantArray::get(streamSetTy, tmp3);
    }

    errs() << "RUN: " << runLength << ", AS=" << additionalStrides << "\n";


    for (unsigned r = 0; r < additionalStrides; ++r) {
        assert (dataVectorArray[r] == dataVectorArray[r % runLength]);
        assert (dataVectorArray[r]);
        dataVectorArray[r + runLength] = dataVectorArray[r];
    }

    ArrayType * const arrTy = ArrayType::get(streamSetTy, dataVectorArray.size());

    Constant * const patternVec = ConstantArray::get(arrTy, dataVectorArray);

    // TODO: we may have multiple uses of a repeating streamset. Will LLVM automatically
    // collapse all uses to a single global variable? If not, have the data analysis
    // identify equivalent patterns, merge the repeating streamsets, and tie the
    // streamset id to the global var.
    Module & mod = *b->getModule();
    GlobalVariable * const patternData =
        new GlobalVariable(mod, arrTy, true, GlobalValue::PrivateLinkage, patternVec);
    const auto align = blockWidth / 8;
    patternData->setAlignment(MaybeAlign{align});

    BasicBlock * const copyAndExpandGlobal = b->CreateBasicBlock();
    BasicBlock * const copyAndExpandGlobalLoop = b->CreateBasicBlock();
    BasicBlock * const exit = b->CreateBasicBlock();

    const BufferNode & bn = mBufferGraph[streamSet];
    assert (bn.isConstant());
    RepeatingBuffer * const buffer = cast<RepeatingBuffer>(bn.Buffer);

    ConstantInt * const baseLength = b->getSize(runLength);
    buffer->setModulus(baseLength);

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
    offset[1] = sz_ZERO;
    buffer->setBaseAddress(b, b->CreateGEP(addr, offset));
}

void PipelineCompiler::addRepeatingStreamSetBufferProperties(BuilderRef b) {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isConstant())) {
            Type * const handleTy = cast<RepeatingBuffer>(bn.Buffer)->getHandleType(b);
            mTarget->addInternalScalar(handleTy,
                REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet),
                                       getCacheLineGroupId(PipelineOutput));
            mTarget->addInternalScalar(b->getVoidPtrTy(),
                REPEATING_STREAMSET_MALLOCED_DATA_PREFIX + std::to_string(streamSet),
                                       getCacheLineGroupId(PipelineOutput));
        }
    }
}

void PipelineCompiler::deallocateRepeatingBuffers(BuilderRef b) {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isConstant())) {
            const auto bufferName = REPEATING_STREAMSET_MALLOCED_DATA_PREFIX + std::to_string(streamSet);
            b->CreateFree(b->getScalarField(bufferName));
        }
    }
}

}
