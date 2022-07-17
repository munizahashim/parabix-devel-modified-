/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#ifndef GREP_KERNEL_H
#define GREP_KERNEL_H

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <re/alphabet/alphabet.h>
#include <re/analysis/capture-ref.h>
#include <re/transforms/to_utf8.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <util/slab_allocator.h>


namespace IDISA { class IDISA_Builder; }
namespace re { class CC; class RE; }
namespace cc { class Alphabet; }
namespace grep { class GrepEngine; }
namespace kernel {


using ProgBuilderRef = const std::unique_ptr<ProgramBuilder> &;

class ExternalStreamObject {
public:
    using Allocator = SlabAllocator<ExternalStreamObject *>;
    enum class Kind : unsigned {
        PreDefined, PropertyExternal, CC_External, RE_External, Reference_External,
        WordBoundaryExternal, GraphemeClusterBreak, PropertyBasis
    };
    inline Kind getKind() const {
        return mKind;
    }
    std::string getName() {return mStreamSetName;}
    StreamSet * getStreamSet() {return mStreamSet;}
    std::vector<std::string> getInputNames() {return mInputStreamNames;}
    virtual void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) = 0;
    virtual std::pair<int, int> getLengthRange() {return std::make_pair(1,1);}
    virtual int getOffset() {return 0;}  //default offset unless overridden.
    bool isResolved() {return mStreamSet != nullptr;}
protected:
    static Allocator mAllocator;
    ExternalStreamObject(Kind k, std::string name, std::vector<std::string> inputNames) :
        mStreamSet(nullptr), mKind(k), mStreamSetName(name), mInputStreamNames(inputNames) {}
    StreamSet * mStreamSet;
public:
    void* operator new (std::size_t size) noexcept {
        return mAllocator.allocate<uint8_t>(size);
    }
private:
    Kind mKind;
    std::string mStreamSetName;
    std::vector<std::string> mInputStreamNames;
};

class PreDefined : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PreDefined;
    }
    static inline bool classof(const void *) {
        return false;
    }
    PreDefined(std::string ssName, StreamSet * predefined) :
        ExternalStreamObject(Kind::PreDefined, ssName, {}) {mStreamSet = predefined;}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override {}
};

class PropertyExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    PropertyExternal(re::Name * n) :
        ExternalStreamObject(Kind::PropertyExternal, n->getFullName(), {"u8_basis"}), mName(n) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    re::Name * mName;
};

class CC_External : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::CC_External;
    }
    static inline bool classof(const void *) {
        return false;
    }
    CC_External(std::string name, re::CC * cc) :
        ExternalStreamObject(Kind::CC_External, name, {"u8_basis"}), mCharClass(cc) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    re::CC * mCharClass;
};

class RE_External : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::RE_External;
    }
    static inline bool classof(const void *) {
        return false;
    }
    RE_External(std::string name, grep::GrepEngine * engine, re::RE * re) :
        ExternalStreamObject(Kind::RE_External, name, {"u8_basis"}), mGrepEngine(engine), mRE(re) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
    std::pair<int, int> getLengthRange() override;
    // RE_Externals are compiled using the ICgrep kernel, which returns offset 1.
    int getOffset() override {return 1;}
private:
    grep::GrepEngine *  mGrepEngine;
    re::RE * mRE;
};

class Reference_External : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::Reference_External;
    }
    static inline bool classof(const void *) {
        return false;
    }
    Reference_External(re::ReferenceInfo & refInfo, re::Reference * ref) :
        ExternalStreamObject(Kind::Reference_External, ref->getName(), {"u8_basis"}),
        mRefInfo(refInfo), mRef(ref) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    re::ReferenceInfo & mRefInfo;
    re::Reference * mRef;
};

class GraphemeClusterBreak : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::GraphemeClusterBreak;
    }
    static inline bool classof(const void *) {
        return false;
    }
    GraphemeClusterBreak(re::UTF8_Transformer * t) :
        ExternalStreamObject(Kind::GraphemeClusterBreak, "\\b{g}",
                             {"u8_basis", "u8index"}), mUTF8_transformer(t) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
    std::pair<int, int> getLengthRange() override {return std::make_pair(0, 0);}
    int getOffset() override {return 1;}
private:
    re::UTF8_Transformer * mUTF8_transformer;
};

class WordBoundaryExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::WordBoundaryExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    WordBoundaryExternal() :
        ExternalStreamObject(Kind::WordBoundaryExternal, "\\b", {"u8_basis", "u8index"}) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
    std::pair<int, int> getLengthRange() override {return std::make_pair(0, 0);}
    int getOffset() override {return 1;}
};

class PropertyBasisExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyBasis;
    }
    static inline bool classof(const void *) {
        return false;
    }
    PropertyBasisExternal(ProgBuilderRef b, StreamSet * input, UCD::property_t p) :
    ExternalStreamObject(Kind::PropertyBasis, basisName(p), {"u8_basis"}), mProperty(p) {}
    static std::string basisName(UCD::property_t p);
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    UCD::property_t mProperty;
};

class UTF8_index : public pablo::PabloKernel {
public:
    UTF8_index(BuilderRef kb, StreamSet * Source, StreamSet * u8index);
protected:
    void generatePabloMethod() override;
};

enum class GrepCombiningType {None, Exclude, Include};
class GrepKernelOptions {
    friend class ICGrepKernel;
public:
    using Alphabets = std::vector<std::pair<const cc::Alphabet *, StreamSet *>>;
    GrepKernelOptions(const cc::Alphabet * codeUnitAlphabet = &cc::UTF8, re::EncodingTransformer * encodingTransformer = nullptr);
    void setIndexingTransformer(re::EncodingTransformer *, StreamSet * indexStream);
    void setSource(StreamSet * s);
    void setCombiningStream(GrepCombiningType t, StreamSet * toCombine);
    void setResults(StreamSet * r);
    void addExternal(std::string name,
                     StreamSet * strm,
                     unsigned offset = 0,
                     std::pair<int, int> lengthRange = std::make_pair<int,int>(1, 1));
    void addAlphabet(const cc::Alphabet * a, StreamSet * basis);
    void setRE(re::RE * re);

protected:
    Bindings streamSetInputBindings();
    Bindings streamSetOutputBindings();
    Bindings scalarInputBindings();
    Bindings scalarOutputBindings();
    std::string makeSignature();

private:

    const cc::Alphabet *        mCodeUnitAlphabet;
    re::EncodingTransformer *   mEncodingTransformer;
    StreamSet *                 mSource = nullptr;
    StreamSet *                 mIndexStream = nullptr;
    GrepCombiningType           mCombiningType = GrepCombiningType::None;
    StreamSet *                 mCombiningStream = nullptr;
    StreamSet *                 mResults = nullptr;
    Bindings                    mExternalBindings;
    std::vector<unsigned>       mExternalOffsets;
    std::vector<std::pair<int, int>>       mExternalLengths;
    Alphabets                   mAlphabets;
    re::RE *                    mRE = nullptr;
};


class ICGrepKernel : public pablo::PabloKernel {
public:
    ICGrepKernel(BuilderRef iBuilder,
                 std::unique_ptr<GrepKernelOptions> && options);
    llvm::StringRef getSignature() const override;
    bool hasSignature() const override { return true; }
    bool hasFamilyName() const override { return true; }
protected:
    void generatePabloMethod() override;
private:
    std::unique_ptr<GrepKernelOptions>  mOptions;
    std::string                         mSignature;
};

class MatchedLinesKernel : public pablo::PabloKernel {
public:
    MatchedLinesKernel(BuilderRef builder, StreamSet * OriginalMatches, StreamSet * LineBreakStream, StreamSet * Matches);
protected:
    void generatePabloMethod() override;
};

class InvertMatchesKernel : public BlockOrientedKernel {
public:
    InvertMatchesKernel(BuilderRef b, StreamSet * OriginalMatches, StreamSet * LineBreakStream, StreamSet * Matches);
private:
    void generateDoBlockMethod(BuilderRef iBuilder) override;
};

class FixedMatchSpansKernel : public pablo::PabloKernel {
public:
    FixedMatchSpansKernel(BuilderRef builder, unsigned length, StreamSet * MatchFollows, StreamSet * MatchSpans);
    bool hasFamilyName() const override { return true; }
protected:
    void generatePabloMethod() override;
    unsigned mMatchLength;
};

//
//  Given an input stream consisting of spans of 1s, return a pair of
//  streams marking the starts of each span as well as the follows.
//
//  Ex:  spans   .....1111.......1111111.........1....
//       starts  .....1..........1...............1....
//       follows .........1.............1.........1...
//
class SpansToMarksKernel : public pablo::PabloKernel {
public:
    SpansToMarksKernel(BuilderRef builder, StreamSet * Spans, StreamSet * EndMarks);
protected:
    void generatePabloMethod() override;
};

class U8Spans : public pablo::PabloKernel {
public:
    U8Spans(BuilderRef builder, StreamSet * marks, StreamSet * u8index, StreamSet * spans);
protected:
    void generatePabloMethod() override;
};

class PopcountKernel : public pablo::PabloKernel {
public:
    PopcountKernel(BuilderRef builder, StreamSet * const toCount, Scalar * countResult);
protected:
    void generatePabloMethod() override;
};

class FixedDistanceMatchesKernel : public pablo::PabloKernel {
public:
    FixedDistanceMatchesKernel(BuilderRef b, unsigned distance, StreamSet * Basis, StreamSet * Matches, StreamSet * ToCheck  = nullptr);
protected:
    void generatePabloMethod() override;
private:
    unsigned mMatchDistance;
    bool mHasCheckStream;
};

class AbortOnNull final : public MultiBlockKernel {
public:
    AbortOnNull(BuilderRef, StreamSet * const InputStream, StreamSet * const OutputStream, Scalar * callbackObject);
private:
    void generateMultiBlockLogic(BuilderRef b, llvm::Value * const numOfStrides) final;

};

/* Given a marker position P, a before-context B and and after-context A, a
   context span is a set of consecutive 1 bits from positions P-B to P+A.

   This kernel computes a coalesced context span stream for all markers in
   a given marker stream.   Coalesced spans occur when markers are separated
   by A + B positions or fewer. */

class ContextSpan final : public pablo::PabloKernel {
public:
    ContextSpan(BuilderRef b, StreamSet * const markerStream, StreamSet * const contextStream, unsigned before, unsigned after);
protected:
    void generatePabloMethod() override;
private:
    const unsigned          mBeforeContext;
    const unsigned          mAfterContext;
};

void GraphemeClusterLogic(ProgBuilderRef P,
                          re::UTF8_Transformer * t,
                          StreamSet * Source, StreamSet * U8index, StreamSet * GCBstream);

void WordBoundaryLogic(ProgBuilderRef P,
                          StreamSet * Source, StreamSet * U8index, StreamSet * wordBoundary_stream);

}
#endif
