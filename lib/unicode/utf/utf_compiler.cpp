#include <unicode/utf/utf_compiler.h>
#include <unicode/utf/utf_encoder.h>
#include <array>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_var.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/printer_pablos.h>
#include <re/alphabet/alphabet.h>
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_compiler.h>
#include <re/adt/re_name.h>
#include <re/adt/re_cc.h>
#include <unicode/core/unicode_set.h>
#include <toolchain/toolchain.h>
#include <llvm/Support/CommandLine.h>
#include <boost/intrusive/detail/math.hpp>

using namespace cc;
using namespace re;
using namespace pablo;
using namespace llvm;
using namespace boost::container;

namespace UTF {

static cl::opt<bool> UseComputedUTFHierarchy("UseComputedUTFHierarchy", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> BinaryLogicCostPerByte("BinaryLogicCostPerByte", cl::init(2), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> TernaryLogicCostPerByte("TernaryLogicCostPerByte", cl::init(1), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> ShiftCostFactor("ShiftCostFactor", cl::init(10), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> IfEmbeddingCostThreshhold("IfEmbeddingCostThreshhold", cl::init(15), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> PartitioningFactor("PartitioningFactor", cl::init(4), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> SuffixOptimization("SuffixOptimization", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> InitialBit7Test("InitialBit7Test", cl::init(false), cl::cat(codegen::CodeGenOptions));

std::string kernelAnnotation() {
    std::string a = "+b" + std::to_string(BinaryLogicCostPerByte);
    a += "t" + std::to_string(TernaryLogicCostPerByte);
    a += "s" + std::to_string(ShiftCostFactor);
    a += "i" + std::to_string(IfEmbeddingCostThreshhold);
    a += "p" + std::to_string(PartitioningFactor);
    if (SuffixOptimization) {
        a += "sfx";
    }
    if (InitialBit7Test) {
        a += "b7";
    }
    if (!UseComputedUTFHierarchy) {
        a += "+LegacyUTFH";
    }
    return a;
}



class UTF_Legacy_Compiler {
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

    UTF_Legacy_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb, PabloAST * mask = nullptr);

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

using PabloAST = pablo::PabloAST;
using PabloBuilder = pablo::PabloBuilder;
using Basis_Set = std::vector<PabloAST *>;


struct Range {
    codepoint_t lo;
    codepoint_t hi;
    bool is_empty() {return lo > hi;}
};

struct LengthInfo {
    unsigned lgth;
    Range range;
    CC_List ccs;
    Range actualRange;
    PabloAST * test;
    PabloAST * combined_test;
};

const UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::defaultIfHierachy = {
    // Non-ASCII
    {0x80, 0x10FFFF},
    // Two-byte sequences
    {0x80, 0x7FF},
    {0x100, 0x3FF},
    // 0100..017F; Latin Extended-A
    // 0180..024F; Latin Extended-B
    // 0250..02AF; IPA Extensions
    // 02B0..02FF; Spacing Modifier Letters
    {0x100, 0x2FF}, {0x100, 0x24F}, {0x100, 0x17F}, {0x180, 0x24F}, {0x250, 0x2AF}, {0x2B0, 0x2FF},
    // 0300..036F; Combining Diacritical Marks
    // 0370..03FF; Greek and Coptic
    {0x300, 0x36F}, {0x370, 0x3FF},
    // 0400..04FF; Cyrillic
    // 0500..052F; Cyrillic Supplement
    // 0530..058F; Armenian
    // 0590..05FF; Hebrew
    // 0600..06FF; Arabic
    {0x400, 0x5FF}, {0x400, 0x4FF}, {0x500, 0x058F}, {0x500, 0x52F}, {0x530, 0x58F}, {0x590, 0x5FF}, {0x600, 0x6FF},
    // 0700..074F; Syriac
    // 0750..077F; Arabic Supplement
    // 0780..07BF; Thaana
    // 07C0..07FF; NKo
    {0x700, 0x77F}, {0x700, 0x74F}, {0x750, 0x77F}, {0x780, 0x7FF}, {0x780, 0x7BF}, {0x7C0, 0x7FF},
    // Three-byte sequences
    {0x800, 0xFFFF}, {0x800, 0x4DFF}, {0x800, 0x1FFF}, {0x800, 0x0FFF},
    // 0800..083F; Samaritan
    // 0840..085F; Mandaic
    // 08A0..08FF; Arabic Extended-A
    // 0900..097F; Devanagari
    // 0980..09FF; Bengali
    // 0A00..0A7F; Gurmukhi
    // 0A80..0AFF; Gujarati
    // 0B00..0B7F; Oriya
    // 0B80..0BFF; Tamil
    // 0C00..0C7F; Telugu
    // 0C80..0CFF; Kannada
    // 0D00..0D7F; Malayalam
    // 0D80..0DFF; Sinhala
    // 0E00..0E7F; Thai
    // 0E80..0EFF; Lao
    // 0F00..0FFF; Tibetan
    {0x1000, 0x1FFF},
    // 1000..109F; Myanmar
    // 10A0..10FF; Georgian
    // 1100..11FF; Hangul Jamo
    // 1200..137F; Ethiopic
    // 1380..139F; Ethiopic Supplement
    // 13A0..13FF; Cherokee
    // 1400..167F; Unified Canadian Aboriginal Syllabics
    // 1680..169F; Ogham
    // 16A0..16FF; Runic
    // 1700..171F; Tagalog
    // 1720..173F; Hanunoo
    // 1740..175F; Buhid
    // 1760..177F; Tagbanwa
    // 1780..17FF; Khmer
    // 1800..18AF; Mongolian
    // 18B0..18FF; Unified Canadian Aboriginal Syllabics Extended
    // 1900..194F; Limbu
    // 1950..197F; Tai Le
    // 1980..19DF; New Tai Lue
    // 19E0..19FF; Khmer Symbols
    // 1A00..1A1F; Buginese
    // 1A20..1AAF; Tai Tham
    // 1AB0..1AFF; Combining Diacritical Marks Extended
    // 1B00..1B7F; Balinese
    // 1B80..1BBF; Sundanese
    // 1BC0..1BFF; Batak
    // 1C00..1C4F; Lepcha
    // 1C50..1C7F; Ol Chiki
    // 1CC0..1CCF; Sundanese Supplement
    // 1CD0..1CFF; Vedic Extensions
    // 1D00..1D7F; Phonetic Extensions
    // 1D80..1DBF; Phonetic Extensions Supplement
    // 1DC0..1DFF; Combining Diacritical Marks Supplement
    // 1E00..1EFF; Latin Extended Additional
    // 1F00..1FFF; Greek Extended
    {0x2000, 0x4DFF}, {0x2000, 0x2FFF},
    {0x3000, 0x4DFF},
    {0x4E00, 0x9FFF},
    // 4E00..9FFF; CJK Unified Ideographs
    {0xA000, 0xFFFF},

    {0x10000, 0x10FFFF}};

const UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::noIfHierachy = {{0x80, 0x10FFFF}};

class CostModel {

public:
    CostModel(UTF_Encoder & e, bool useTernaryLogic);
    unsigned incrementalCost(codepoint_t cp);
    void addExpensiveSubranges(UCD::UnicodeSet endpoints, codepoint_t lo_base, codepoint_t hi_ceil, UTF_Legacy_Compiler::RangeList & r);

private:
    UTF_Encoder & mEncoder;
    unsigned mCodeUnitLogicCost;
    std::vector<unsigned> mCurrentCodeUnits;
};

CostModel::CostModel(UTF_Encoder & e, bool useTernaryLogic) : mEncoder(e) {
    unsigned codeUnitSize = e.getCodeUnitBits();
    unsigned octets = (codeUnitSize + 7)/8;
    if (useTernaryLogic) {
        mCodeUnitLogicCost = octets * TernaryLogicCostPerByte;
    } else {
        mCodeUnitLogicCost = octets * BinaryLogicCostPerByte;
    }
}

unsigned CostModel::incrementalCost(codepoint_t cp) {
    unsigned units = mEncoder.encoded_length(cp);
    if (units != mCurrentCodeUnits.size()) {
        mCurrentCodeUnits.resize(units);
        for (unsigned i = 0; i < units; i++) {
            mCurrentCodeUnits[i] = mEncoder.nthCodeUnit(cp, i);
        }
        return (units - 1) * ShiftCostFactor + units * mCodeUnitLogicCost;
    } else {
        unsigned common_units = 0;
        for (unsigned i = 0; i < units - 1; i++) {
            if (mEncoder.nthCodeUnit(cp, i) == mCurrentCodeUnits[i]) {
                common_units++;
            } else {
                break;
            }
        }
        unsigned new_units = units - common_units;
        return (new_units - 1) * ShiftCostFactor + new_units * mCodeUnitLogicCost;
    }
}

void CostModel::addExpensiveSubranges(UCD::UnicodeSet endpoints, codepoint_t lo_base, codepoint_t hi_ceil, UTF_Legacy_Compiler::RangeList & r) {
    UTF_Legacy_Compiler::RangeList expensiveRanges;
    bool base_cp_found = false;
    bool limit_cp_found = false;
    codepoint_t base_cp = 0;
    codepoint_t limit_cp = 0;
    unsigned range_cost = 0;
    for (const auto & range : endpoints) {
        codepoint_t range_lo = lo_codepoint(range);
        codepoint_t range_hi = hi_codepoint(range);
        if (range_hi < lo_base) continue;  // Ignore everthing below lo_base.
        if (range_lo < lo_base) range_lo = lo_base;
        if (range_hi > hi_ceil) range_hi = hi_ceil;
        for (codepoint_t endpt = range_lo; endpt <= range_hi; endpt++) {
            if (limit_cp_found && (endpt < limit_cp)) continue;
            if (!base_cp_found) {
                base_cp = endpt & ~ 0x3F;  // mask off low 6 dynamic_bits
                base_cp_found = true;
                limit_cp_found = false;
                range_cost = 0;
            }
            if (!limit_cp_found) {
                range_cost += incrementalCost(endpt);
                if (range_cost > IfEmbeddingCostThreshhold) {
                    limit_cp = endpt | 0x3F;
                    limit_cp_found = true;
                    r.push_back(interval_t{base_cp, limit_cp});
                    base_cp_found = false;
                }
            }
        }
    }
}

UCD::UnicodeSet computeEndpoints(const std::vector<re::CC *> & CCs) {
    UCD::UnicodeSet endpoints;
    for (unsigned i = 0; i < CCs.size(); i++) {
        for (const auto range : *CCs[i]) {
            const auto lo = re::lo_codepoint(range);
            const auto hi = re::hi_codepoint(range);
            endpoints.insert(lo);
            endpoints.insert(hi);
        }
    }
    return endpoints;
}

UCD::UnicodeSet computeEndpoints(const std::vector<const re::CC *> & CCs) {
    UCD::UnicodeSet endpoints;
    for (unsigned i = 0; i < CCs.size(); i++) {
        for (const auto range : *CCs[i]) {
            const auto lo = re::lo_codepoint(range);
            const auto hi = re::hi_codepoint(range);
            endpoints.insert(lo);
            endpoints.insert(hi);
        }
    }
    return endpoints;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateRange
 ** ------------------------------------------------------------------------------------------------------------- */
void UTF_Legacy_Compiler::generateRange(const RangeList & ifRanges, PabloBuilder & entry) {
    // Pregenerate the suffix var outside of the if ranges. The DCE pass will either eliminate it if it's not used or the
    // code sinking pass will move appropriately into an inner if block.
    if (mMask) {
        auto nested = entry.createScope();
        entry.createIf(mMask, nested);
        generateRange(ifRanges, 0, UCD::UNICODE_MAX, nested);
    } else {
        generateRange(ifRanges, 0, UCD::UNICODE_MAX, entry);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateRange
 * @param ifRangeList
 ** ------------------------------------------------------------------------------------------------------------- */
void UTF_Legacy_Compiler::generateRange(const RangeList & ifRanges, const codepoint_t lo, const codepoint_t hi, PabloBuilder & builder) {

    // Codepoints in unenclosed ranges will be computed unconditionally.
    // Generate them first so that computed subexpressions may be shared
    // with calculations within the if hierarchy.
    const auto enclosed = rangeIntersect(ifRanges, lo, hi);
    for (const auto & rg : rangeGaps(enclosed, lo, hi)) {
        generateSubRanges(lo_codepoint(rg), hi_codepoint(rg), builder);
    }

    for (auto & f : mTargetValue) {
        const auto v = mTarget.find(f.first);
        assert (v != mTarget.end());
        PabloAST * value = f.second;
        if (!isa<Zeroes>(value)) {
            if (LLVM_LIKELY(builder.getParent() != nullptr)) {
                value = builder.createOr(v->second, value);
            }
            builder.createAssign(v->second, value);
            f.second = builder.createZeroes();
        }
    }

    const auto outer = outerRanges(enclosed);
    const auto inner = innerRanges(enclosed);

    Values nonIntersectingTargets;

    for (const auto & range : outer) {

        // Split our current target list into two sets: the intersecting and non-intersecting ones. Any non-
        // intersecting set will be removed from the current map to eliminate the possibility of it being
        // considered until after we leave the current range. The intersecting sets are also stored to ensure
        // that we know what the original target value was going into this range block so tha we can OR the
        // inner value with the outer value.

        for (auto f = mTargetValue.begin(); f != mTargetValue.end(); ) {
            if (f->first->intersects(range.first, range.second)) {
                ++f;
            } else {
                nonIntersectingTargets.push_back(*f);
                f = mTargetValue.erase(f);
            }
        }
        if (mTargetValue.size() > 0) {
            auto inner_block = builder.createScope();
            builder.createIf(ifTestCompiler(range.first, range.second, builder), inner_block);
            generateRange(inner, range.first, range.second, inner_block);
        }
        for (auto t : nonIntersectingTargets) {
            mTargetValue.insert(t);
        }
        nonIntersectingTargets.clear();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSubRanges
 ** ------------------------------------------------------------------------------------------------------------- */
void UTF_Legacy_Compiler::generateSubRanges(const codepoint_t lo, const codepoint_t hi, PabloBuilder & builder) {
    for (auto & t : mTargetValue) {
        const auto range = rangeIntersect(*t.first, lo, hi);
        PabloAST * target = t.second;
        // Divide by UTF-8 length, separating out E0, ED, F0 and F4 ranges
        const std::array<interval_t, 9> ranges =
            {{{0, 0x7F}, {0x80, 0x7FF}, {0x800, 0xFFF}, {0x1000, 0xD7FF}, {0xD800, 0xDFFF},
             {0xE000, 0xFFFF}, {0x10000, 0x3FFFF}, {0x40000, 0xFFFFF}, {0x100000, 0x10FFFF}}};
        for (auto r : ranges) {
            const auto subrange = rangeIntersect(range, lo_codepoint(r), hi_codepoint(r));
            target = sequenceGenerator(std::move(subrange), 1, builder, target, nullptr);
        }
        t.second = target;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief sequenceGenerator
 * @param ifRangeList
 *
 *
 * Generate remaining code to match UTF-8 code sequences within the codepoint set cpset, assuming that the code
 * matching the sequences up to byte number code_unit have been generated.
 ** ------------------------------------------------------------------------------------------------------------- */
PabloAST * UTF_Legacy_Compiler::sequenceGenerator(const RangeList && ranges, const unsigned code_unit, PabloBuilder & builder, PabloAST * target, PabloAST * prefix) {

    if (LLVM_LIKELY(ranges.size() > 0)) {

        codepoint_t lo, hi;
        std::tie(lo, hi) = ranges[0];

        const auto lgth_min = mEncoder.encoded_length(lo_codepoint(ranges.front()));
        const auto lgth_max = mEncoder.encoded_length(hi_codepoint(ranges.back()));

        if (lgth_min != lgth_max) {
            const auto mid = mEncoder.max_codepoint_of_length(lgth_min);
            target = sequenceGenerator(std::move(rangeIntersect(ranges, lo, mid)), code_unit, builder, target, prefix);
            target = sequenceGenerator(std::move(rangeIntersect(ranges, mid + 1, hi)), code_unit, builder, target, prefix);
        } else if (code_unit == lgth_min) {
            // We have a single code unit remaining to match for all code points in this CC.
            // Use the character class compiler to generate matches for these codepoints.
            PabloAST * var = mCodeUnitCompiler->compileCC(makeCC(byteDefinitions(ranges, code_unit), &Byte), builder);
            PabloAST * prior = makePrefix(lo, code_unit, builder, prefix);
            if (code_unit <= 1) {
                target = builder.createOr(target, var);
            } else {
                target = builder.createOrAnd(target, var, builder.createAdvance(prior, 1));
            }
        } else {
            for (auto rg : ranges) {
                codepoint_t lo, hi;
                std::tie(lo, hi) = rg;
                auto lo_unit = mEncoder.nthCodeUnit(lo, code_unit);
                auto hi_unit = mEncoder.nthCodeUnit(hi, code_unit);
                if (lo_unit != hi_unit) {
                    if (!mEncoder.isLowCodePointAfterNthCodeUnit(lo, code_unit)) {
                        codepoint_t mid = mEncoder.maxCodePointWithCommonCodeUnits(lo, code_unit);
                        target = sequenceGenerator(lo, mid, code_unit, builder, target, prefix);
                        target = sequenceGenerator(mid + 1, hi, code_unit, builder, target, prefix);
                    } else if (!mEncoder.isHighCodePointAfterNthCodeUnit(hi, code_unit)) {
                        codepoint_t mid = mEncoder.minCodePointWithCommonCodeUnits(hi, code_unit);
                        target = sequenceGenerator(lo, mid - 1, code_unit, builder, target, prefix);
                        target = sequenceGenerator(mid, hi, code_unit, builder, target, prefix);
                    } else { // we have a prefix group covering a full range
                        PabloAST * var = mCodeUnitCompiler->compileCC(makeByte(lo_unit, hi_unit), builder);
                        unsigned len = mEncoder.encoded_length(lo);
                        if (code_unit > 1) {
                            var = builder.createAnd(builder.createAdvance(prefix, 1), var);
                        }
                        for (unsigned i = code_unit+1; i < len; ++i) {
                            lo_unit = mEncoder.nthCodeUnit(lo, i);
                            hi_unit = mEncoder.nthCodeUnit(hi, i);
                            PabloAST * sfx = mCodeUnitCompiler->compileCC(makeByte(lo_unit, hi_unit), builder);
                            var = builder.createAnd(sfx, builder.createAdvance(var, 1));
                        }
                        lo_unit = mEncoder.nthCodeUnit(lo, lgth_min);
                        hi_unit = mEncoder.nthCodeUnit(hi, lgth_min);
                        PabloAST * sfx = mCodeUnitCompiler->compileCC(makeByte(lo_unit, hi_unit), builder);
                        target = builder.createOrAnd(target, sfx, builder.createAdvance(var, 1));
                    }
                } else { // lo_unit == hi_unit
                    PabloAST * var = mCodeUnitCompiler->compileCC(makeByte(lo_unit, hi_unit), builder);
                    if (code_unit > 1) {
                        var = builder.createAnd(builder.createAdvance(prefix ? prefix : var, 1), var);
                    }
                    if (code_unit < mEncoder.encoded_length(lo)) {
                        target = sequenceGenerator(lo, hi, code_unit + 1, builder, target, var);
                    }
                }
            }
        }
    }
    return target;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief sequenceGenerator
 ** ------------------------------------------------------------------------------------------------------------- */
inline PabloAST * UTF_Legacy_Compiler::sequenceGenerator(const codepoint_t lo, const codepoint_t hi, const unsigned code_unit, PabloBuilder & builder, PabloAST * target, PabloAST * prefix) {
    return sequenceGenerator({{ lo, hi }}, code_unit, builder, target, prefix);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ifTestCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
inline PabloAST * UTF_Legacy_Compiler::ifTestCompiler(const codepoint_t lo, const codepoint_t hi, PabloBuilder & builder) {
    return ifTestCompiler(lo, hi, 1, builder, builder.createOnes());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ifTestCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
PabloAST * UTF_Legacy_Compiler::ifTestCompiler(const codepoint_t lo, const codepoint_t hi, const unsigned code_unit, PabloBuilder & builder, PabloAST * target) {

    codepoint_t lo_unit = mEncoder.nthCodeUnit(lo, code_unit);
    codepoint_t hi_unit = mEncoder.nthCodeUnit(hi, code_unit);
    const bool at_lo_boundary = (lo == 0 || mEncoder.nthCodeUnit(lo - 1, code_unit) != lo_unit);
    const bool at_hi_boundary = (hi == 0x10FFFF || mEncoder.nthCodeUnit(hi + 1, code_unit) != hi_unit);

    if (at_lo_boundary && at_hi_boundary) {
        PabloAST * cc = mCodeUnitCompiler->compileCC(makeByte(lo_unit, hi_unit), builder);
        target = builder.createAnd(cc, target);
    } else if (lo_unit == hi_unit) {
        PabloAST * cc = mCodeUnitCompiler->compileCC(makeByte(lo_unit, hi_unit), builder);
        target = builder.createAnd(cc, target);
        target = builder.createAdvance(target, 1);
        target = ifTestCompiler(lo, hi, code_unit + 1, builder, target);
    } else if (!at_hi_boundary) {
        const auto mid = mEncoder.minCodePointWithCommonCodeUnits(hi, code_unit);
        PabloAST * e1 = ifTestCompiler(lo, mid - 1, code_unit, builder, target);
        PabloAST * e2 = ifTestCompiler(mid, hi, code_unit, builder, target);
        target = builder.createOr(e1, e2);
    } else {
        const auto mid = mEncoder.maxCodePointWithCommonCodeUnits(lo, code_unit);
        PabloAST * e1 = ifTestCompiler(lo, mid, code_unit, builder, target);
        PabloAST * e2 = ifTestCompiler(mid + 1, hi, code_unit, builder, target);
        target = builder.createOr(e1, e2);
    }
    return target;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief definePrecedingPrefix
 * @param ifRangeList
 *
 *
 * Ensure the sequence of preceding bytes is defined, up to, but not including the given code_unit
 ** ------------------------------------------------------------------------------------------------------------- */
PabloAST * UTF_Legacy_Compiler::makePrefix(const codepoint_t cp, const unsigned code_unit, PabloBuilder & builder, PabloAST * prefix) {
    assert (code_unit >= 1 && code_unit <= 4);
    assert (code_unit == 1 || prefix != nullptr);
    for (unsigned i = 1; i != code_unit; ++i) {
        const CC * const cc = makeByte(mEncoder.nthCodeUnit(cp, i));
        PabloAST * var = mCodeUnitCompiler->compileCC(cc, builder);
        if (i > 1) {
            var = builder.createAnd(var, builder.createAdvance(prefix, 1));
        }
        prefix = var;
    }
    return prefix;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief definePrecedingPrefix
 * @param ifRangeList
 *
 *
 * Ensure the sequence of preceding bytes is defined, up to, but not including the given code_unit
 ** ------------------------------------------------------------------------------------------------------------- */
UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::byteDefinitions(const RangeList & list, const unsigned code_unit) {
    RangeList result;
    result.reserve(list.size());
    for (const auto & i : list) {
        result.emplace_back(mEncoder.nthCodeUnit(lo_codepoint(i), code_unit), mEncoder.nthCodeUnit(hi_codepoint(i), code_unit));
    }
    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief rangeIntersect
 * @param list
 * @param lo
 * @param hi
 ** ------------------------------------------------------------------------------------------------------------- */
template <typename RangeListOrUnicodeSet>
UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::rangeIntersect(const RangeListOrUnicodeSet & list, const codepoint_t lo, const codepoint_t hi) {
    RangeList result;
    for (const auto & i : list) {
        if ((lo_codepoint(i) <= hi) && (hi_codepoint(i) >= lo)) {
            result.emplace_back(std::max(lo, lo_codepoint(i)), std::min(hi, hi_codepoint(i)));
        }
    }
    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief rangeGaps
 * @param cc
 * @param lo
 * @param hi
 ** ------------------------------------------------------------------------------------------------------------- */
UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::rangeGaps(const RangeList & list, const codepoint_t lo, const codepoint_t hi) {
    RangeList gaps;
    if (LLVM_LIKELY(lo < hi)) {
        if (LLVM_UNLIKELY(list.empty())) {
            gaps.emplace_back(lo, hi);
        } else {
            codepoint_t cp = lo;
            for (const auto & i : list) {
                if (hi_codepoint(i) < cp) {
                    continue;
                } else if (lo_codepoint(i) > cp) {
                    gaps.emplace_back(cp, lo_codepoint(i) - 1);
                } else if (hi_codepoint(i) >= hi) {
                    continue;
                }
                cp = hi_codepoint(i) + 1;
            }
        }
    }
    return gaps;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief outerRanges
 * @param list
 ** ------------------------------------------------------------------------------------------------------------- */
UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::outerRanges(const RangeList & list) {
    RangeList ranges;
    if (LLVM_LIKELY(list.size() > 0)) {
        auto i = list.cbegin();
        for (auto j = i + 1; j != list.cend(); ++j) {
            if (hi_codepoint(*j) > hi_codepoint(*i)) {
                ranges.emplace_back(lo_codepoint(*i), hi_codepoint(*i));
                i = j;
            }
        }
        if (LLVM_LIKELY(i != list.end())) {
            ranges.emplace_back(lo_codepoint(*i), hi_codepoint(*i));
        }
    }
    return ranges;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief innerRanges
 ** ------------------------------------------------------------------------------------------------------------- */
UTF_Legacy_Compiler::RangeList UTF_Legacy_Compiler::innerRanges(const RangeList & list) {
    RangeList ranges;
    if (LLVM_LIKELY(list.size() > 0)) {
        for (auto i = list.cbegin(), j = i + 1; j != list.cend(); ++j) {
            if (hi_codepoint(*j) <= hi_codepoint(*i)) {
                ranges.emplace_back(lo_codepoint(*j), hi_codepoint(*j));
            } else {
                i = j;
            }
        }
    }
    return ranges;
}

void UTF_Legacy_Compiler::addTarget(Var * theVar, const CC * theCC) {
    mTarget.emplace(theCC, theVar);
    mTargetValue.emplace(theCC, mPb.createZeroes());
}

void UTF_Legacy_Compiler::compile() {
    if (!UseComputedUTFHierarchy) {
        generateRange(defaultIfHierachy, mPb);
    } else {
        std::vector<const CC *> CCs;
        for (auto & f: mTargetValue) {
            CCs.push_back(f.first);
        }

        UCD::UnicodeSet endpoints = computeEndpoints(CCs);
        CostModel m(mEncoder, false);
        
        RangeList rgs = {{0x80, 0x10FFFF}, {0x80, 0x7FF}};
        m.addExpensiveSubranges(endpoints, 0x80, 0x7FF, rgs);
        rgs.push_back({0x800, 0xFFFF});
        m.addExpensiveSubranges(endpoints, 0x800, 0xFFFF, rgs);
        rgs.push_back({0x10000, 0x10FFFF});
        m.addExpensiveSubranges(endpoints, 0x10000, 0x10FFFF, rgs);
        generateRange(rgs, mPb);

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
UTF_Legacy_Compiler::UTF_Legacy_Compiler(Var * basis_var, pablo::PabloBuilder & pb, PabloAST * mask)
: mPb(pb), mMask(mask) {
    llvm::ArrayType * ty = cast<ArrayType>(basis_var->getType());
    unsigned streamCount = ty->getArrayNumElements();
    if (streamCount == 1) {
        VectorType * const vt = cast<VectorType>(ty->getArrayElementType());
        const auto streamWidth = vt->getElementType()->getIntegerBitWidth();
        mEncoder.setCodeUnitBits(streamWidth);
        mCodeUnitCompiler =
        std::make_unique<cc::Direct_CC_Compiler>(pb.createExtract(basis_var, pb.getInteger(0)));
    } else {
        mEncoder.setCodeUnitBits(streamCount);
        std::vector<PabloAST *> basis_set(streamCount);
        for (unsigned i = 0; i < streamCount; i++) {
            basis_set[i] = pb.createExtract(basis_var, pb.getInteger(i));
        }
        mCodeUnitCompiler =
        std::make_unique<cc::Parabix_CC_Compiler_Builder>(basis_set);
    }
}


using boost::intrusive::detail::ceil_log2;

//
// Determine the actual range of codepoints encountered in a CC_List.
// If all the CCs are empty, return the impossible range {0x10FFF, 0}.
Range CC_Set_Range(CC_List ccs) {
    codepoint_t lo = 0x10FFFF;
    codepoint_t hi = 0;
    for (unsigned i = 0; i < ccs.size(); i++) {
        if (!ccs[i]->empty()) {
            lo = std::min(lo_codepoint(ccs[i]->front()), lo);
            hi = std::max(hi_codepoint(ccs[i]->back()), hi);
        }
    }
    return Range{lo, hi};
}

void extract_CCs_by_range(Range r, CC_List & ccs, CC_List & in_range) {
    re::CC * rangeCC = re::makeCC(r.lo, r.hi, &Unicode);
    for (unsigned i = 0; i < ccs.size(); i++) {
        in_range[i] = re::intersectCC(ccs[i], rangeCC);
    }
}

class New_UTF_Compiler {
public:
    New_UTF_Compiler(pablo::Var * Var, PabloBuilder & pb);
    void compile(Target_List targets, CC_List ccs);
protected:
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
    void extendLengthHierarchy(std::vector<LengthInfo> & lengthInfo, unsigned i, PabloBuilder & pb);
    void subrangePartitioning(CC_List & ccs, Range & range, PabloAST * rangeTest, PabloBuilder & pb);
    void compileSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb);
    void compileUnguardedSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb);
    re::CC * prefixCC(CC_List & ccs);
    re::CC * codeUnitCC(re::CC * cc, unsigned codeunit);
    unsigned costModel(CC_List & ccs);
    virtual void prepareScope(unsigned scope, PabloBuilder & pb) = 0;
    virtual PabloAST * compile_code_unit(unsigned lgth, unsigned position, re::CC * unitCC, PabloBuilder & pb) = 0;
};

New_UTF_Compiler::New_UTF_Compiler(Var * basis_var, pablo::PabloBuilder & pb) :
        mBasisVar(basis_var), mPB(pb)  {
    mEncoder.setCodeUnitBits(8);
}


void New_UTF_Compiler::compile(Target_List targets, CC_List ccs) {
    //  Initialize all the target vars to 0.
    mTargets = targets;
    for (unsigned i = 0; i < targets.size(); i++) {
        mPB.createAssign(mTargets[i], mPB.createZeroes());
    }
    llvm::ArrayType * ty = cast<ArrayType>(mBasisVar->getType());
    unsigned streamCount = ty->getArrayNumElements();
    assert(streamCount == 8); // For now.
    mScopeBasis[0].resize(streamCount);
    for (unsigned i = 0; i < streamCount; i++) {
        mScopeBasis[0][i] = mPB.createExtract(mBasisVar, mPB.getInteger(i));
    }
    mCodeUnitCompilers[0] =
        std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[0]);
    mScopeLength = 1;
    createLengthHierarchy(ccs);
}


void New_UTF_Compiler::createLengthHierarchy(CC_List & ccs) {
    // Maximum number of code units (UTF-8) is 4.
    std::vector<LengthInfo> lengthInfo;
    //
    // For each possible multibyte sequence length, determine
    // (1) the subsets of the ccs that are of that length
    // (2) a narrow codepoint range of the ccs of this length
    // (3) a narrow test for the prefixes of multibyte sequences of that lgth.
    codepoint_t lgth_base = 0;
    for (unsigned lgth = 1; lgth <= mEncoder.encoded_length(0x10FFFF); lgth++) {
        Range lgth_range{lgth_base, mEncoder.max_codepoint_of_length(lgth)};
        CC_List length_ccs(ccs.size());
        extract_CCs_by_range(lgth_range, ccs, length_ccs);
        Range actual_range = CC_Set_Range(length_ccs);
        if (!actual_range.is_empty()) {
            PabloAST * lgth_test = nullptr;
            if (lgth > 1) {
                re::CC * lgth_test_CC = prefixCC(length_ccs);
                lgth_test = compile_code_unit(lgth, 1, lgth_test_CC, mPB);
            } else {
                lgth_test = mPB.createOnes(); // mPB.createNot(mScopeBasis[0][7]);
            }
            lengthInfo.push_back(LengthInfo{lgth, lgth_range, length_ccs, actual_range, lgth_test, lgth_test});
            llvm::errs() << "lengthInfo.size() = " << lengthInfo.size() << "\n";
        }
        lgth_base = lgth_range.hi + 1;
    }
    if (lengthInfo.size() >= 2) {
        PabloAST * combined = lengthInfo[lengthInfo.size() - 1].test;
        for (int i = lengthInfo.size() - 2; i >= 0; i--) {
            llvm::errs() << "i  = " << i << "\n";
            if (lengthInfo[i].lgth > 1) {
                lengthInfo[i].combined_test = mPB.createOr(lengthInfo[i].test, combined);
                combined = lengthInfo[i].combined_test;
            }
        }
    }
    unsigned first_pfx = 0;
    if (lengthInfo[0].lgth == 1) {
        compileSubrange(lengthInfo[0].ccs, lengthInfo[0].range, lengthInfo[0].test, lengthInfo[0].actualRange, mPB);
        first_pfx = 1;
    }
    if (InitialBit7Test) {
        auto nested = mPB.createScope();
        mPB.createIf(mScopeBasis[0][7], nested);
        extendLengthHierarchy(lengthInfo, first_pfx, nested);
    } else {
        extendLengthHierarchy(lengthInfo, first_pfx, mPB);
    }
}

void New_UTF_Compiler::extendLengthHierarchy(std::vector<LengthInfo> & lengthInfo, unsigned i, PabloBuilder & pb) {
    llvm::errs() << "extendLengthHierarchy(" << i << ")\n";
    if (lengthInfo.size() <= i) return;
    PabloAST * outer_test = lengthInfo[i].combined_test;
    auto nested = pb.createScope();
    pb.createIf(outer_test, nested);
    //PabloAST * inner_test = outer_test;
    while (mScopeLength < lengthInfo[i].lgth) {
        prepareScope(mScopeLength, nested);
        mScopeLength++;
    }
    codepoint_t test_lo = mEncoder.minCodePointWithCommonCodeUnits(lengthInfo[i].actualRange.lo, 1);
    codepoint_t test_hi = mEncoder.maxCodePointWithCommonCodeUnits(lengthInfo[i].actualRange.hi, 1);
    Range testRange{test_lo, test_hi};
    subrangePartitioning(lengthInfo[i].ccs, testRange, lengthInfo[i].test, nested);
    extendLengthHierarchy(lengthInfo, i+1, nested);
}

void New_UTF_Compiler::subrangePartitioning(CC_List & ccs, Range & range, PabloAST * rangeTest, PabloBuilder & pb) {
    auto differing_bits = range.lo ^ range.hi;
    // Calculate the index of the high differing bit, noting that
    // the special case if the differing bits value is exactly 2^k.
    unsigned high_differing_bit = ceil_log2(differing_bits + 1) - 1;
    codepoint_t partition_size = (1U << high_differing_bit)/PartitioningFactor;
    llvm::errs() << "subrangePartitioning(";
    llvm::errs().write_hex(range.lo);
    llvm::errs() << ", ";
    llvm::errs().write_hex(range.hi);
    llvm::errs() << ")\n";
    llvm::errs() << "  partition_size = " << partition_size << "\n";
    if (partition_size < 32) {
        partition_size = 32;
    }
    if (range.hi-range.lo <= partition_size) {
        compileUnguardedSubrange(ccs, range, rangeTest, range, pb);
        return;
    }
    codepoint_t mask = partition_size - 1;
    codepoint_t base = range.lo & ~mask;
    for (unsigned partition_lo = base; partition_lo < range.hi; partition_lo += partition_size) {
        unsigned partition_hi = std::min(partition_lo + partition_size - 1, range.hi);
        Range partition{partition_lo, partition_hi};
        compileSubrange(ccs, range, rangeTest, partition, pb);
    }
}

void New_UTF_Compiler::compileSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb) {
    CC_List subrangeCCs(ccs.size());
    extract_CCs_by_range(subrange, ccs, subrangeCCs);
    //  If there are no CCs that intersect the subrange, no code
    //  generation is required.
    Range actual_subrange = CC_Set_Range(subrangeCCs);
    if (actual_subrange.is_empty()) return;
    //
    // Determine whether compilation of the CCs is below our cost model threshhold.
    unsigned costFactor = costModel(subrangeCCs);
    llvm::errs() << "compileSubrange(";
    llvm::errs().write_hex(enclosingRange.lo);
    llvm::errs() << ", ";
    llvm::errs().write_hex(enclosingRange.hi);
    llvm::errs() << ") subrange(";
    llvm::errs().write_hex(subrange.lo);
    llvm::errs() << ", ";
    llvm::errs().write_hex(subrange.hi);
    llvm::errs() << ")\n";
    llvm::errs() << "  costFactor = " << costFactor << "\n";
    if (costFactor < IfEmbeddingCostThreshhold) {
        compileUnguardedSubrange(subrangeCCs, enclosingRange, enclosingTest, enclosingRange, pb);
        return;
    }
    // The subrange logic cost exceeds our cost model threshhold.
    // Construct a guarded if-block and partition into further subranges.
    unsigned encoded_length = mEncoder.encoded_length(enclosingRange.lo);
    unsigned code_unit_to_test = mEncoder.common_code_units(enclosingRange.lo, enclosingRange.hi) + 1;
    unsigned lo_unit = mEncoder.nthCodeUnit(actual_subrange.lo, code_unit_to_test);
    unsigned hi_unit = mEncoder.nthCodeUnit(actual_subrange.hi, code_unit_to_test);
    llvm::errs() << "  code_unit_to_test = " << code_unit_to_test << "\n";
    llvm::errs() << "  lo_unit = " << lo_unit << "  hi_unit = " << hi_unit << "\n";
    // Adjust for suffixes by masking off suffix bit 7;
    if (SuffixOptimization && (code_unit_to_test > 1)) {
        lo_unit &= 0x3F;
        hi_unit &= 0x3F;
    }
    CC * unitCC = re::makeByte(lo_unit, hi_unit);
    //

    // TODO: Consider optimizing this test to take advantage of any bits
    // the current unit that have been determined by enclosingTest.

    PabloAST * unit_test = compile_code_unit(encoded_length, code_unit_to_test, unitCC, pb);
    PabloAST * subrange_test = pb.createAnd(enclosingTest, unit_test);

    // Construct an if-block.
    auto nested = pb.createScope();
    pb.createIf(subrange_test, nested);
    codepoint_t test_lo = mEncoder.minCodePointWithCommonCodeUnits(actual_subrange.lo, code_unit_to_test);
    codepoint_t test_hi = mEncoder.maxCodePointWithCommonCodeUnits(actual_subrange.hi, code_unit_to_test);
    Range testRange{test_lo, test_hi};
    subrangePartitioning(subrangeCCs, testRange, subrange_test, nested);
}

void New_UTF_Compiler::compileUnguardedSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb) {
    CC_List subrangeCCs(ccs.size());
    extract_CCs_by_range(subrange, ccs, subrangeCCs);
    //  If there are no CCs that intersect the subrange, no code
    //  generation is required.
    Range actual_subrange = CC_Set_Range(subrangeCCs);
    if (actual_subrange.is_empty()) return;
    unsigned encoded_length = mEncoder.encoded_length(actual_subrange.lo);
    unsigned code_unit_to_test = mEncoder.common_code_units(enclosingRange.lo, enclosingRange.hi) + 1;
    if (code_unit_to_test == encoded_length) {
        // We are on the final code unit; compile each nonempty CC and
        // combine into the top-level Var for this CC.
        for (unsigned i = 0; i < ccs.size(); i++) {
            if (!subrangeCCs[i]->empty()) {
                re::CC * unitCC = codeUnitCC(subrangeCCs[i], code_unit_to_test);
                PabloAST * final_unit = compile_code_unit(code_unit_to_test, code_unit_to_test, unitCC, pb);
                PabloAST * test = pb.createAnd(enclosingTest, final_unit);
                pb.createAssign(mTargets[i], pb.createOr(mTargets[i], test));
            }
        }
    } else {
        // We are not at the final unit.  Narrow down to subsubranges based
        // on each possible code unit at this level.
        unsigned lo_unit = mEncoder.nthCodeUnit(subrange.lo, code_unit_to_test);
        unsigned hi_unit = mEncoder.nthCodeUnit(subrange.hi, code_unit_to_test);
        codepoint_t min_cp = mEncoder.minCodePointWithCommonCodeUnits(subrange.lo, code_unit_to_test);
        codepoint_t max_cp = mEncoder.maxCodePointWithCommonCodeUnits(subrange.hi, code_unit_to_test);
        if (lo_unit != hi_unit) {
            codepoint_t mid_cp = mEncoder.maxCodePointWithCommonCodeUnits(subrange.lo, code_unit_to_test);
            Range lo_subrange{min_cp, mid_cp};
            Range hi_subrange{mid_cp+1, max_cp};
            compileUnguardedSubrange(subrangeCCs, enclosingRange, enclosingTest, lo_subrange, pb);
            compileUnguardedSubrange(subrangeCCs, enclosingRange, enclosingTest, hi_subrange, pb);
        } else {
            re::CC * unitCC = re::makeByte(lo_unit, lo_unit);
            PabloAST * unit_test = compile_code_unit(encoded_length, code_unit_to_test, unitCC, pb);
            PabloAST * subrangeTest = pb.createAnd(enclosingTest, unit_test);
            Range testedSubrange{min_cp, max_cp};
            compileUnguardedSubrange(subrangeCCs, testedSubrange, subrangeTest, actual_subrange, pb);
        }
    }
}

re::CC * New_UTF_Compiler::codeUnitCC(re::CC * cc, unsigned codeunit) {
    re::CC * unitCC = re::makeCC(&Byte);
    for (auto i : *cc) {
        unsigned lo_unit = mEncoder.nthCodeUnit(lo_codepoint(i), codeunit);
        unsigned hi_unit = mEncoder.nthCodeUnit(hi_codepoint(i), codeunit);
        //
        // Mask off bits 6 and 7 for suffixes.
        if (SuffixOptimization && (codeunit > 1)) {
            lo_unit &= 0x3F;
            hi_unit &= 0x3F;
        }
        unitCC = makeCC(unitCC, makeByte(lo_unit, hi_unit));
    }
    return unitCC;
}

re::CC * New_UTF_Compiler::prefixCC(CC_List & ccs) {
    re::CC * unitCC = re::makeCC(&Byte);
    for (auto cc : ccs) {
        for (auto i : *cc) {
            unsigned lo_unit = mEncoder.nthCodeUnit(lo_codepoint(i), 1);
            unsigned hi_unit = mEncoder.nthCodeUnit(hi_codepoint(i), 1);
            unitCC = makeCC(unitCC, makeByte(lo_unit, hi_unit));
        }
    }
    return unitCC;
}


unsigned New_UTF_Compiler::costModel(CC_List & ccs) {
    UCD::UnicodeSet endpoints = computeEndpoints(ccs);
    Range cc_span = CC_Set_Range(ccs);
    if (cc_span.is_empty()) return 0;
    unsigned bits_to_test = ceil_log2(cc_span.hi - cc_span.lo + 1);
    unsigned total_codepoints = endpoints.count();
    return total_codepoints * bits_to_test * BinaryLogicCostPerByte / 8;
}

class UTF_Lookahead_Compiler : public New_UTF_Compiler {
public:
    UTF_Lookahead_Compiler(pablo::Var * Var, PabloBuilder & pb);
protected:
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    PabloAST * compile_code_unit(unsigned lgth, unsigned position, re::CC * unitCC, PabloBuilder & pb) override;
};

class UTF_Advance_Compiler : public New_UTF_Compiler {
public:
    UTF_Advance_Compiler(pablo::Var * Var, PabloBuilder & pb);
protected:
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    PabloAST * compile_code_unit(unsigned lgth, unsigned position, re::CC * unitCC, PabloBuilder & pb) override;
};

UTF_Lookahead_Compiler::UTF_Lookahead_Compiler(pablo::Var * v, PabloBuilder & pb) :
New_UTF_Compiler(v, pb) {}

void UTF_Lookahead_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    mScopeBasis[scope].resize(mScopeBasis[0].size());
    for (unsigned i = 0; i < mScopeBasis[0].size(); i++) {
        mScopeBasis[scope][i] = pb.createLookahead(mScopeBasis[0][i], scope);
    }
#if 0
    // Combine the suffix bits into the current test and discard them.
    if (SuffixOptimization) {
        // Combine the suffix bits into the current test and discard them.
        inner_test = nested.createAnd(mScopeBasis[scope][7], inner_test);
        lengthInfo[i].test = nested.createAnd(nested.createNot(mScopeBasis[scope][6]), inner_test);
        mScopeBasis[scope].resize(6);
    }
#endif
    mCodeUnitCompilers[scope] =
    std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[scope]);
}

PabloAST * UTF_Lookahead_Compiler::compile_code_unit(unsigned lgth, unsigned pos, re::CC * unitCC, PabloBuilder & pb) {
    return mCodeUnitCompilers[pos - 1]->compileCC(unitCC, pb);
}

UTF_Advance_Compiler::UTF_Advance_Compiler(pablo::Var * v, PabloBuilder & pb) :
    New_UTF_Compiler(v, pb) {}

void UTF_Advance_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    mScopeBasis[scope].resize(mScopeBasis[0].size());
    for (unsigned i = 0; i < mScopeBasis[0].size(); i++) {
        mScopeBasis[scope][i] = pb.createAdvance(mScopeBasis[0][i], scope);
    }
#if 0
    // Combine the suffix bits into the current test and discard them.
    if (SuffixOptimization) {
        // Combine the suffix bits into the current test and discard them.
        inner_test = nested.createAnd(mScopeBasis[scope][7], inner_test);
        lengthInfo[i].test = nested.createAnd(nested.createNot(mScopeBasis[scope][6]), inner_test);
        mScopeBasis[scope].resize(6);
    }
#endif
    mCodeUnitCompilers[scope] =
    std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[scope]);
}

PabloAST * UTF_Advance_Compiler::compile_code_unit(unsigned lgth, unsigned pos, re::CC * unitCC, PabloBuilder & pb) {
    return mCodeUnitCompilers[lgth - pos]->compileCC(unitCC, pb);
}

UTF_Compiler::UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
             pablo::PabloAST * mask,
             pablo::BitMovementMode mode) :
mVar(basisVar), mPB(pb), mMask(mask), mBitMovement(mode) {}

UTF_Compiler::UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
             pablo::BitMovementMode mode) :
mVar(basisVar), mPB(pb), mMask(nullptr), mBitMovement(mode) {}

void UTF_Compiler::compile(Target_List targets, CC_List ccs) {
    if (mBitMovement == pablo::BitMovementMode::LookAhead) {
        if (mMask != nullptr) {
            llvm::report_fatal_error("mask parameter for the UTF lookahead compiler is a future option.");
        }
        UTF_Lookahead_Compiler utf_compiler(mVar, mPB);
        utf_compiler.compile(targets, ccs);
    } else {
        UTF_Legacy_Compiler utf_compiler(mVar, mPB, mMask);
        for (unsigned i = 0; i < targets.size(); i++) {
            utf_compiler.addTarget(targets[i], ccs[i]);
        }
        utf_compiler.compile();
    }
}

}
