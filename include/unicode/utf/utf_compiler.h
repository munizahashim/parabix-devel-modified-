#pragma once

#include <unicode/core/UCD_Config.h>
#include <unicode/utf/utf_encoder.h>
#include <vector>
#include <boost/container/flat_map.hpp>

namespace cc {
    class CC_Compiler;
}

namespace re {
    class Name;
    class CC;
}

namespace pablo {
    class PabloBlock;
    class PabloBuilder;
    class PabloAST;
    class Var;
}

namespace UTF {

class UnicodeSet;

std::string kernelAnnotation();

class UTF_Compiler {
public:

    using CC = re::CC;
    using PabloBuilder = pablo::PabloBuilder;
    using PabloAST = pablo::PabloAST;
    using RangeList = std::vector<UCD::interval_t>;

    using TargetMap = boost::container::flat_map<const CC *, pablo::Var *>;
    using ValueMap = boost::container::flat_map<const CC *, PabloAST *>;
    using Values = std::vector<std::pair<ValueMap::key_type, ValueMap::mapped_type>>;

    static const RangeList defaultIfHierachy;
    static const RangeList noIfHierachy;

    enum class IfHierarchy {None, Default, Computed};
    using NameMap = boost::container::flat_map<re::Name *, PabloAST *>;

    UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb, unsigned lookAhead = 0, PabloAST * mask = nullptr);

    void addTarget(pablo::Var * theVar, const re::CC * theCC);

    void compile();

protected:

    void generateRange(const RangeList & ifRanges, PabloBuilder & entry);

    void generateRange(const RangeList & ifRanges, const codepoint_t lo, const codepoint_t hi, PabloBuilder & builder);

    void generateSubRanges(const codepoint_t lo, const codepoint_t hi, PabloBuilder & builder);

    PabloAST * sequenceGenerator(const RangeList && ranges, const unsigned byte_no, PabloBuilder & builder, PabloAST * target, PabloAST * prefix);

    PabloAST * sequenceGenerator(const codepoint_t lo, const codepoint_t hi, const unsigned byte_no, PabloBuilder & builder, PabloAST * target, PabloAST * prefix);

    PabloAST * ifTestCompiler(const codepoint_t lo, const codepoint_t hi, PabloBuilder & builder);

    PabloAST * ifTestCompiler(const codepoint_t lo, const codepoint_t hi, const unsigned byte_no, PabloBuilder & builder, PabloAST * target);

    PabloAST * makePrefix(const codepoint_t cp, const unsigned byte_no, PabloBuilder & builder, PabloAST * prefix);

    RangeList byteDefinitions(const RangeList & list, const unsigned byte_no);

    template <typename RangeListOrUnicodeSet>
    static RangeList rangeIntersect(const RangeListOrUnicodeSet & list, const codepoint_t lo, const codepoint_t hi);

    static RangeList rangeGaps(const RangeList & list, const codepoint_t lo, const codepoint_t hi);

    static RangeList outerRanges(const RangeList & list);

    static RangeList innerRanges(const RangeList & list);

private:
    UTF_Encoder             mEncoder;
    pablo::PabloBuilder &   mPb;
//    unsigned                mLookAhead;
    PabloAST *              mMask;
    std::unique_ptr<cc::CC_Compiler>       mCodeUnitCompiler;
    TargetMap               mTarget;
    ValueMap                mTargetValue;
};

using Target_List = std::vector<pablo::Var *>;
using CC_List = std::vector<re::CC *>;
using PabloAST = pablo::PabloAST;
using PabloBuilder = pablo::PabloBuilder;
using Basis_Set = std::vector<PabloAST *>;


struct Range {
    codepoint_t lo;
    codepoint_t hi;
    bool is_empty() {return lo > hi;}
};

class UTF_Lookahead_Compiler {
public:
    UTF_Lookahead_Compiler(pablo::Var * Var, PabloBuilder & pb);
    void compile(Target_List targets, CC_List ccs);
private:
    pablo::Var *            mBasisVar;
    PabloBuilder &          mPB;
    UTF_Encoder             mEncoder;
    Target_List             mTargets;
    //  Depending on the actual CC_List being compiled, up to
    //  4 scope positions will be defined, with corresponding basis
    //  sets and code unit compilers.
    unsigned                mScopeLength;
    Basis_Set               mScopeBasis[4];
    std::unique_ptr<cc::CC_Compiler> mCodeUnitCompilers[4];
    void createLengthHierarchy(CC_List & ccs);
    void extendLengthHierarchy(CC_List & ccs, Range r, PabloBuilder & pb);
    void subrangePartitioning(CC_List & ccs, Range & range, PabloAST * rangeTest, PabloBuilder & pb);
    void compileSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb);
    void compileUnguardedSubrange(CC_List & ccs, Range & subrange, PabloAST * subrangeTest, PabloBuilder & pb);
    unsigned costModel(CC_List ccs);
};




}

