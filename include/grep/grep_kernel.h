/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#ifndef GREP_KERNEL_H
#define GREP_KERNEL_H

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <pablo/pablo_toolchain.h>
#include <re/alphabet/alphabet.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/analysis/capture-ref.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/re_name_gather.h>
#include <re/transforms/to_utf8.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/util/debug_display.h>
#include <util/slab_allocator.h>


namespace IDISA { class IDISA_Builder; }
namespace cc { class Alphabet; }
namespace re { class CC; class RE; }
namespace grep { class GrepEngine; }
namespace kernel {

using ProgBuilderRef = const std::unique_ptr<ProgramBuilder> &;

class ExternalStreamObject;

//  StreamIndexCode is used to identify the indexing base of
//  streamsets.   UTF-8 and Unicode indexed streams are two
//  common indexing bases.
using StreamIndexCode = unsigned;

struct StreamIndexInfo {
    std::string name;
    StreamIndexCode base;
    std::string indexStreamName;
};

class ExternalStreamTable {
public:
    ExternalStreamTable() : mIllustrator(nullptr) {}
    StreamIndexCode declareStreamIndex(std::string indexName, StreamIndexCode base = 0, std::string indexStreamName = "");
    StreamIndexCode getStreamIndex(std::string indexName);
    void declareExternal(StreamIndexCode c, std::string externalName, ExternalStreamObject * ext);
    ExternalStreamObject * lookup(StreamIndexCode c, std::string externalName);
    bool isDeclared(StreamIndexCode c, std::string externalName);
    bool hasReferenceTo(StreamIndexCode c, std::string externalName);
    StreamSet * getStreamSet(ProgBuilderRef b, StreamIndexCode c, std::string externalName);
    void resetExternals();  // Reset all externals to unresolved.
    void resolveExternals(ProgBuilderRef b);
    void setIllustrator(kernel::ParabixIllustrator * illustrator) {mIllustrator = illustrator;}
    ~ExternalStreamTable();
private:
    std::vector<StreamIndexInfo> mStreamIndices;
    std::vector<std::map<std::string, ExternalStreamObject *>> mExternalMap;
    kernel::ParabixIllustrator * mIllustrator;
};

using ExternalMapRef = ExternalStreamTable *;

class ExternalStreamObject {
    friend class ExternalStreamTable;
public:
    enum class Kind : unsigned {
        U21, PreDefined, LineStarts, CC_External, RE_External,
        PropertyExternal, PropertyBasis, PropertyDistance, PropertyBoundary,
        WordBoundaryExternal, GraphemeClusterBreak, Multiplexed,
        FilterByMask, FixedSpan, MarkedSpanExternal
    };
    inline Kind getKind() const {
        return mKind;
    }
    StreamSet * getStreamSet() {return mStreamSet;}
    // Most externals are computed from the basis bit stremas.
    virtual std::vector<std::string> getParameters() {return std::vector<std::string>{"basis"};}
    virtual void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) = 0;
    std::pair<int, int> getLengthRange() {return mLengthRange;}
    int getOffset() {return mOffset;}
    bool isResolved() {return mStreamSet != nullptr;}
    virtual ~ExternalStreamObject() {}
protected:
    ExternalStreamObject(Kind k, std::pair<int, int> lgthRange = std::make_pair(1,1), int offset = 0) :
        mKind(k), mLengthRange(lgthRange), mOffset(offset), mStreamSet(nullptr)  {}
    void installStreamSet(StreamSet * s);
protected:
    Kind mKind;
    std::pair<int, int> mLengthRange;
    int mOffset;
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
    std::vector<std::string> getParameters() override {return {};}
    PreDefined(StreamSet * predefined, std::pair<int, int> lgthRange = std::make_pair(1,1), int offset = 0) :
        ExternalStreamObject(Kind::PreDefined, lgthRange, offset) {mStreamSet = predefined;}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override {}
};

class LineStartsExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::LineStarts;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    LineStartsExternal(std::vector<std::string> parms = {"$"}) :
        ExternalStreamObject(Kind::LineStarts, std::make_pair(0, 0), 1), mParms(parms) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    std::vector<std::string> mParms;
};

class U21_External : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::U21;
    }
    static inline bool classof(const void *) {
        return false;
    }
    U21_External() : ExternalStreamObject(Kind::U21) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
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
        ExternalStreamObject(Kind::PropertyExternal), mName(n) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    re::Name * mName;
};

class PropertyBoundaryExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyBoundary;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    PropertyBoundaryExternal(UCD::property_t p) :
        ExternalStreamObject(Kind::PropertyBoundary, std::make_pair(0, 0), 1), mProperty(p) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    UCD::property_t mProperty;
};

class CC_External : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::CC_External;
    }
    static inline bool classof(const void *) {
        return false;
    }
    CC_External(re::CC * cc) :
        ExternalStreamObject(Kind::CC_External), mCharClass(cc) {}
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
    std::vector<std::string> getParameters() override {return mParams;}
    RE_External(grep::GrepEngine * engine, re::RE * re, const cc::Alphabet * a) :
        ExternalStreamObject(Kind::RE_External, re::getLengthRange(re, a), grepOffset(re)),
            mGrepEngine(engine), mRE(re), mIndexAlphabet(a), mParams(re::gatherExternals(re)) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    grep::GrepEngine *  mGrepEngine;
    re::RE * mRE;
    const cc::Alphabet * mIndexAlphabet;
    std::vector<std::string> mParams;
};

class PropertyDistanceExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::PropertyDistance;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    PropertyDistanceExternal(UCD::property_t p, unsigned dist) :
        ExternalStreamObject(Kind::PropertyDistance),
        mProperty(p), mDistance(dist) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    UCD::property_t mProperty;
    unsigned mDistance;
};

class GraphemeClusterBreak : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::GraphemeClusterBreak;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    GraphemeClusterBreak(grep::GrepEngine * engine, const cc::Alphabet * a) :
        ExternalStreamObject(Kind::GraphemeClusterBreak, std::make_pair(0, 0), 1), mGrepEngine(engine), mIndexAlphabet(a)  {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    grep::GrepEngine *  mGrepEngine;
    const cc::Alphabet * mIndexAlphabet;
};

class WordBoundaryExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::WordBoundaryExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    WordBoundaryExternal() :
        ExternalStreamObject(Kind::WordBoundaryExternal, std::make_pair(0, 0), 1) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
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
    ExternalStreamObject(Kind::PropertyBasis), mProperty(p) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
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
        ExternalStreamObject(Kind::Multiplexed), mAlphabet(mpx) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    cc::MultiplexedAlphabet * mAlphabet;
};

class FilterByMaskExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::FilterByMask;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    StreamIndexCode getBaseIndex() {return mBase;}
    FilterByMaskExternal(StreamIndexCode base, std::vector<std::string> paramNames, ExternalStreamObject * e) :
        ExternalStreamObject(Kind::FilterByMask, e->getLengthRange(), e->getOffset()),
            mBase(base), mParamNames(paramNames), mBaseExternal(e) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    StreamIndexCode mBase;
    std::vector<std::string> mParamNames;
    ExternalStreamObject * mBaseExternal;
};

class FixedSpanExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::FixedSpan;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    FixedSpanExternal(std::string matchMarks, unsigned lgth, int offset) :
        ExternalStreamObject(Kind::FixedSpan, std::make_pair(lgth, lgth), offset), mMatchMarks(matchMarks) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    std::string mMatchMarks;
};

class MarkedSpanExternal : public ExternalStreamObject {
public:
    static inline bool classof(const ExternalStreamObject * ext) {
        return ext->getKind() == Kind::MarkedSpanExternal;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::vector<std::string> getParameters() override;
    MarkedSpanExternal(std::string prefixMarks, unsigned prefixLgth, std::string matchEnds, unsigned offset) :
        ExternalStreamObject(Kind::MarkedSpanExternal, std::make_pair(prefixLgth, INT_MAX), offset),
        mPrefixMarks(prefixMarks), mPrefixLength(prefixLgth), mMatchMarks(matchEnds) {}
    void resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) override;
private:
    std::string mPrefixMarks;
    unsigned mPrefixLength;
    std::string mMatchMarks;
};

//
// Compute a stream marking UTF-8 character positions.   Each valid
// character is marked at the position of its final UTF-8 byte.
// If the optional linebreak parameter is included, also include
// any positions marked in this stream, including a possible extra
// bit just past EOF if the file is unterminated.
//
class UTF8_index : public pablo::PabloKernel {
public:
    UTF8_index(BuilderRef kb, StreamSet * Source, StreamSet * u8index, StreamSet * linebreak = nullptr);
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

//  Given selected UTF-8 characters identified in the marks stream, and
//  a u8index stream marking the last position of each UTF-8 character,
//  produce a stream of UTF-8 character spans, in which each position of
//  marked UTF-8 characters are identified with 1 bits.   The marks
//  stream may be marking the first byte of each UTF-8 sequence in the
//  BitMovementMode::Advance mode or the last byte of each UTF-8 sequence
//  in the BitMovementMode::LookAhead mode (default).
class U8Spans : public pablo::PabloKernel {
public:
    U8Spans(BuilderRef builder, StreamSet * marks, StreamSet * u8index, StreamSet * spans,
            pablo::BitMovementMode m = pablo::BitMovementMode::LookAhead);
protected:
    void generatePabloMethod() override;
private:
    pablo::BitMovementMode mBitMovement;
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

//  The LongestMatchMarks kernel computes longest-match spans in start-end space.
//  Logically, the input is a set of 2 streams marking, respectively, matches
//  of a necessary prefix of the RE, and matches to the full RE.   However,
//  a single combined stream may be provided as the start-end stream when these
//  two cases do not intersect.   In this case,each 0 bit in start-end space
//  marks the occurrence of a necessary prefix of the RE, while each 1 bit marks
//  an actual match end for the full RE.  The results produced are (a) the start
//  position immediately preceding a full match, and (b) the longest full match
//  corresponding to that start position.
//  For example:
//  start-end stream:  00111110001100101000100
//  (a) starts:        .1.......1...1.1...1...
//  (b) longest end:   ......1....1..1.1...1..
//  The output is a set of two streams for the start and longest end marks, respectively.

class LongestMatchMarks final : public pablo::PabloKernel {
public:
    LongestMatchMarks(BuilderRef b, StreamSet * start_ends, StreamSet * marks);
protected:
    void generatePabloMethod() override;
};

//  Compute match spans given a set of two streams marking a fixed prefix position
//  of the match, as well as the final position of the match.   The prefix may
//  be at an offset from the actual start position of the match, while the suffix
//  may be at an offset from the last matched position.
//  For example, the pair of input streams:
//  prefix:  ...1......1.......1.....
//  final:   .....1.....1......1.....
//  the spans computed with a prefix offset of 2 and suffix offset of 0 are:
//  spans    .11111..1111....111.....
//
class InclusiveSpans final : public pablo::PabloKernel {
public:
    InclusiveSpans(BuilderRef b, unsigned prefixOffset, unsigned suffixOffset,
                   StreamSet * marks, StreamSet * spans);
protected:
    void generatePabloMethod() override;
private:
    unsigned mPrefixOffset;
    unsigned mSuffixOffset;
};
}
#endif
