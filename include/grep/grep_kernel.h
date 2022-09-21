/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#ifndef GREP_KERNEL_H
#define GREP_KERNEL_H

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <re/alphabet/alphabet.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/analysis/capture-ref.h>
#include <re/transforms/to_utf8.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <util/slab_allocator.h>


namespace IDISA { class IDISA_Builder; }
namespace cc { class Alphabet; }
namespace re { class CC; class RE; }
namespace grep { class GrepEngine; }
namespace kernel {

using ProgBuilderRef = const std::unique_ptr<ProgramBuilder> &;

class ExternalStreamObject;
class ExternalStreamTable {
public:
    ExternalStreamTable() {}
    void emplace(std::string externalName, ExternalStreamObject * ext);
    ExternalStreamObject * lookup(std::string externalName);
    StreamSet * getStreamSet(ProgBuilderRef b, std::string externalName);
private:
    std::map<std::string, ExternalStreamObject *> mExternalMap;
};

using ExternalMapRef = ExternalStreamTable *;

class ExternalStreamObject {
public:
    using Allocator = SlabAllocator<ExternalStreamObject *>;
    enum class Kind : unsigned {
        PreDefined, PropertyExternal, CC_External, RE_External, StartAnchored,
        Reference_External,
        WordBoundaryExternal, GraphemeClusterBreak, PropertyBasis, Multiplexed
    };
    inline Kind getKind() const {
        return mKind;
    }
    std::string getName() {return mStreamSetName;}
    StreamSet * getStreamSet() {return mStreamSet;}
    virtual void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) = 0;
    virtual std::pair<int, int> getLengthRange() {return std::make_pair(1,1);}
    virtual int getOffset() {return 0;}  //default offset unless overridden.
    bool isResolved() {return mStreamSet != nullptr;}
    void setIndexing(ProgBuilderRef b, StreamSet * indexStrm);
protected:
    static Allocator mAllocator;
    ExternalStreamObject(Kind k, std::string name) :
        mKind(k), mStreamSetName(name), mIndexStream(nullptr), mStreamSet(nullptr)  {}
    void installStreamSet(ProgBuilderRef b, StreamSet * s);
public:
    void* operator new (std::size_t size) noexcept {
        return mAllocator.allocate<uint8_t>(size);
    }
private:
    Kind mKind;
    std::string mStreamSetName;
    StreamSet * mIndexStream;
protected:
    StreamSet * mStreamSet;
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
        ExternalStreamObject(Kind::PreDefined, ssName) {mStreamSet = predefined;}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override {}
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
        ExternalStreamObject(Kind::PropertyExternal, n->getFullName()), mName(n) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
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
        ExternalStreamObject(Kind::CC_External, name), mCharClass(cc) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
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
    RE_External(std::string name, grep::GrepEngine * engine, re::RE * re, const cc::Alphabet * a) :
        ExternalStreamObject(Kind::RE_External, name), mGrepEngine(engine), mRE(re), mIndexAlphabet(a) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
    std::pair<int, int> getLengthRange() override;
    int getOffset() override {return mOffset;}
private:
    grep::GrepEngine *  mGrepEngine;
    re::RE * mRE;
    const cc::Alphabet * mIndexAlphabet;
    unsigned mOffset;
};

class StartAnchoredExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::StartAnchored;
    }
    static inline bool classof(const void *) {
        return false;
    }
    StartAnchoredExternal(std::string name, grep::GrepEngine * engine, re::RE * re, const cc::Alphabet * a) :
        ExternalStreamObject(Kind::StartAnchored, name), mGrepEngine(engine), mRE(re), mIndexAlphabet(a) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
    std::pair<int, int> getLengthRange() override;
    int getOffset() override {return mOffset;}
private:
    grep::GrepEngine *  mGrepEngine;
    re::RE * mRE;
    const cc::Alphabet * mIndexAlphabet;
    unsigned mOffset;
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
        ExternalStreamObject(Kind::Reference_External, ref->getName()),
        mRefInfo(refInfo), mRef(ref) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
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
    GraphemeClusterBreak(grep::GrepEngine * engine) :
        ExternalStreamObject(Kind::GraphemeClusterBreak, "\\b{g}"), mGrepEngine(engine)  {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
    std::pair<int, int> getLengthRange() override {return std::make_pair(0, 0);}
    int getOffset() override {return 1;}
private:
    grep::GrepEngine *  mGrepEngine;
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
        ExternalStreamObject(Kind::WordBoundaryExternal, "\\b") {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
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
    PropertyBasisExternal(UCD::property_t p) :
    ExternalStreamObject(Kind::PropertyBasis,
                         "UCD:" + getPropertyFullName(p) + "_basis"), mProperty(p) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
private:
    UCD::property_t mProperty;
};

class MultiplexedExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::Multiplexed;
    }
    static inline bool classof(const void *) {
        return false;
    }
    MultiplexedExternal(cc::MultiplexedAlphabet * mpx) :
    ExternalStreamObject(Kind::Multiplexed, mpx->getName()), mAlphabet(mpx) {}
    void resolveStreamSet(ProgBuilderRef b, ExternalMapRef m) override;
private:
    cc::MultiplexedAlphabet * mAlphabet;
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
    GrepKernelOptions(const cc::Alphabet * codeUnitAlphabet = &cc::UTF8);
    void setIndexing(StreamSet * indexStream);
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
    unsigned getOffset() {return mOffset;}
protected:
    void generatePabloMethod() override;
private:
    std::unique_ptr<GrepKernelOptions>  mOptions;
    std::string                         mSignature;
    unsigned                            mOffset;
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
    FixedMatchSpansKernel(BuilderRef builder, unsigned length, unsigned offset, StreamSet * MatchMarks, StreamSet * MatchSpans);
    bool hasFamilyName() const override { return true; }
protected:
    void generatePabloMethod() override;
    unsigned mMatchLength;
    unsigned mOffset;
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
                          StreamSet * Source, StreamSet * U8index, StreamSet * GCBstream);

void WordBoundaryLogic(ProgBuilderRef P,
                          StreamSet * Source, StreamSet * U8index, StreamSet * wordBoundary_stream);

//  Find longest-match spans in start-end space.   Each 0 bit in start-end space
//  marks the occurrence of a necessary prefix of the RE, while each 1 bit marks
//  an actual match end for the full RE.  The results produced are (a) the start
//  position immediately preceding a full match, and (b) the longest full match
//  corresponding to that start position.
//  For example:
//  start-end stream:  00111110001100101000100
//  (a) starts:        .1.......1...1.1...1...
//  (b) longest end:   ......1....1..1.1...1..
//  The input in a singleton streamset for the start-end marks.
//  The output is a set of two streams for the start and longest end marks, respectively.

class LongestMatchMarks final : public pablo::PabloKernel {
public:
    LongestMatchMarks(BuilderRef b, StreamSet * start_ends, StreamSet * marks);
protected:
    void generatePabloMethod() override;
};

//  Compute match spans given a pair of streams marking a fixed prefix position
//  of the match, as well as the final position of the match.   The prefix
//  may be at an offset from the actual start position of the match.
//  For example, the pair of input streams:
//  prefix:  ...1......1.......1.....
//  final:   .....1.....1......1.....
//  the spans computed with a start offset of 2 are:
//  spans    .11111..1111....111.....
//
class InclusiveSpans final : public pablo::PabloKernel {
public:
    InclusiveSpans(BuilderRef b, StreamSet * marks, StreamSet * spans, unsigned start_offset);
protected:
    void generatePabloMethod() override;
private:
    unsigned mOffset;
};

void PrefixSuffixSpan(ProgBuilderRef P,
                      StreamSet * Prefix, StreamSet * Suffix, StreamSet * Spans, unsigned offset = 0);

}
#endif
