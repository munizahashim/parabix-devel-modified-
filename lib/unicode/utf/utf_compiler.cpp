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
#include <pablo/bixnum/bixnum.h>
#include <re/alphabet/alphabet.h>
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_compiler.h>
#include <re/alphabet/alphabet.h>
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

static cl::opt<bool> UseLegacyUTFHierarchy("UseLegacyUTFHierarchy", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> BinaryLogicCostPerByte("BinaryLogicCostPerByte", cl::init(2), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> TernaryLogicCostPerByte("TernaryLogicCostPerByte", cl::init(1), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> ShiftCostFactor("ShiftCostFactor", cl::init(10), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> IfEmbeddingCostThreshhold("IfEmbeddingCostThreshhold", cl::init(15), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> PartitioningCostThreshhold("PartitioningCostThreshhold", cl::init(12), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> PartitioningFactor("PartitioningFactor", cl::init(4), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> SuffixOptimization("SuffixOptimization", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> InitialNonASCIITest("InitialNonASCIITest", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> PrefixCCTest("PrefixCCTest", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> UTF_CompilationTracing("UTF_CompilationTracing", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> BixNumCCs("BixNumCCs", cl::init(false), cl::cat(codegen::CodeGenOptions));

std::string kernelAnnotation() {
    std::string a = "+b" + std::to_string(BinaryLogicCostPerByte);
    a += "t" + std::to_string(TernaryLogicCostPerByte);
    a += "s" + std::to_string(ShiftCostFactor);
    a += "i" + std::to_string(IfEmbeddingCostThreshhold);
    a += "p" + std::to_string(PartitioningCostThreshhold);
    a += "f" + std::to_string(PartitioningFactor);
    if (SuffixOptimization) {
        a += "+sfx";
    }
    if (InitialNonASCIITest) {
        a += "+nA";
    }
    if (PrefixCCTest) {
        a += "+pCC";
    }
    if (BixNumCCs) {
        a += "+bx";
    }
    if (UseLegacyUTFHierarchy) {
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
using boost::intrusive::detail::ceil_log2;

struct Range {
    codepoint_t lo;
    codepoint_t hi;
    bool is_empty() {return lo > hi;}
    unsigned significant_bits() {
        auto differing_bits = lo ^ hi;
        return ceil_log2(differing_bits + 1);
    }
    unsigned total_bits() {
        return ceil_log2(hi+1);
    }
    std::string hex_string() {
        std::stringstream s;
        s << std::hex << lo << "_" << hi;
        return s.str();
    }
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

re::CC * reduceCC(re::CC * cc, codepoint_t mask, cc::Alphabet & a) {
    UCD::UnicodeSet reduced;
    for (const auto range : *cc) {
        const auto lo = re::lo_codepoint(range) & mask;
        const auto hi = re::hi_codepoint(range) & mask;
        reduced.insert_range(lo, hi);
    }
    return re::makeCC(reduced, &a);
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
    if (UseLegacyUTFHierarchy) {
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

class Unicode_Range_Compiler {
public:
    Unicode_Range_Compiler(Basis_Set & basis, Target_List & targets, PabloBuilder & pb) :
        mBasis (basis), mTargets(targets), mPB(pb) {}
    void compile(CC_List & ccs, Range & rg, PabloAST * test);
protected:
    Basis_Set   &           mBasis;
    Target_List  &          mTargets;
    PabloBuilder &          mPB;
    unsigned costModel(CC_List & ccs);
    void subrangePartitioning(CC_List & ccs, Range & range, PabloAST * rangeTest, PabloBuilder & pb);
    void compileSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb);
    void compileUnguardedSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb);
    PabloAST * compileCodeRange(Range & codepointRange, Range & enclosingRange, PabloAST * enclosing, PabloBuilder & pb);
};

void Unicode_Range_Compiler::compile(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest) {
    subrangePartitioning(ccs, enclosingRange, enclosingTest, mPB);
}

unsigned Unicode_Range_Compiler::costModel(CC_List & ccs) {
    UCD::UnicodeSet endpoints = computeEndpoints(ccs);
    Range cc_span = CC_Set_Range(ccs);
    if (cc_span.is_empty()) return 0;
    unsigned bits_to_test = cc_span.significant_bits();
    unsigned total_codepoints = endpoints.count();
    return total_codepoints * bits_to_test * BinaryLogicCostPerByte / 8;
}


void Unicode_Range_Compiler::subrangePartitioning(CC_List & ccs, Range & range, PabloAST * rangeTest, PabloBuilder & pb) {
    unsigned range_bits = range.significant_bits();
    codepoint_t partition_size = (1U << (range_bits - 1))/PartitioningFactor;
    if (UTF_CompilationTracing) {
        llvm::errs() << "URC::subrangePartitioning(" << range.hex_string() << ")\n";
        llvm::errs() << "  partition_size = " << partition_size << "\n";
    }
    if (partition_size <= 32) {
        compileUnguardedSubrange(ccs, range, rangeTest, range, pb);
        return;
    }
    std::string range_alphabet = "Low" + std::to_string(range_bits);
    cc::CodeUnitAlphabet CodeAlpha(range_alphabet, range_alphabet, range_bits);
    pablo::BixNumCompiler bnc(pb);
    BixNum rangeBasis = bnc.Truncate(mBasis, range_bits);
    std::unique_ptr<cc::CC_Compiler> rangeCompiler;
    rangeCompiler = std::make_unique<cc::Parabix_CC_Compiler_Builder>(rangeBasis);
    codepoint_t range_mask = (1U << range_bits) - 1;
    codepoint_t partition_mask = partition_size - 1;
    codepoint_t base = range.lo & ~partition_mask;
    for (unsigned partition_lo = base; partition_lo < range.hi; partition_lo += partition_size) {
        unsigned partition_hi = std::min(partition_lo + partition_size - 1, range.hi);
        Range partition{partition_lo, partition_hi};
        CC_List partitionCCs(ccs.size());
        extract_CCs_by_range(partition, ccs, partitionCCs);
        Range actual_subrange = CC_Set_Range(partitionCCs);
        if (!actual_subrange.is_empty()) {
            unsigned subpartition_bits = actual_subrange.significant_bits();
            codepoint_t mask = (1u << subpartition_bits) - 1;
            Range subpartition{actual_subrange.lo & ~mask, actual_subrange.hi | mask};
            if (UTF_CompilationTracing) {
                llvm::errs() << "partition.significant_bits() = " << partition.significant_bits() << "\n";
                llvm::errs() << "actual_subrange: " << actual_subrange.hex_string() << "\n";
                llvm::errs() << "subpartition: " << subpartition.hex_string() << "\n";
                llvm::errs() << "actual_subrange.significant_bits() = " << subpartition_bits << "\n";
            }
            re::CC * subpartitionCC = re::makeCC(subpartition.lo & range_mask, subpartition.hi & range_mask, &CodeAlpha);
            PabloAST * subpartitionTest = rangeCompiler->compileCC(subpartitionCC, pb);
/*
            PabloAST * subpartitionTest = rangeTest;
            for (unsigned bit = partition_bits; bit <= high_differing_bit; bit++) {
                PabloAST * bitTest;
                if (((subpartition.lo >> bit) & 1) == 1) {
                    bitTest = mBasis[bit];
                } else {
                    bitTest = pb.createNot(mBasis[bit]);
                }
                if (bit == high_differing_bit) {
                    subpartitionTest = pb.createAnd(subpartitionTest, bitTest, "Range_" + subpartition.hex_string());
                } else {
                    subpartitionTest = pb.createAnd(subpartitionTest, bitTest);
                }
            }
 */
            subpartitionTest = pb.createAnd(rangeTest, subpartitionTest, "Range_" + subpartition.hex_string());
            compileSubrange(partitionCCs, subpartition, subpartitionTest, actual_subrange, pb);
        }
    }
}

void Unicode_Range_Compiler::compileSubrange(CC_List & subrangeCCs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb) {
    //
    // Determine whether compilation of the CCs is below our cost model threshhold.
    unsigned costFactor = costModel(subrangeCCs);
    if (UTF_CompilationTracing) {
        llvm::errs() << "URC::compileSubrange(" << enclosingRange.hex_string() << ") subrange(" << subrange.hex_string() << ")\n";
        llvm::errs() << "  costFactor = " << costFactor << "\n";
    }
    if (costFactor < IfEmbeddingCostThreshhold) {
        if (costFactor < PartitioningCostThreshhold) {
            compileUnguardedSubrange(subrangeCCs, enclosingRange, enclosingTest, subrange, pb);
        } else {
            subrangePartitioning(subrangeCCs, enclosingRange, enclosingTest, pb);
        }
        return;
    }
    // The subrange logic cost exceeds our cost model threshhold.
    // Construct a guarded if-block and partition into further subranges.
    PabloAST * unit_test = compileCodeRange(subrange, enclosingRange, enclosingTest, pb);
    PabloAST * subrange_test = pb.createAnd(enclosingTest, unit_test);

    // Construct an if-block.
    auto nested = pb.createScope();
    pb.createIf(subrange_test, nested);
    subrangePartitioning(subrangeCCs, subrange, subrange_test, nested);
}

void Unicode_Range_Compiler::compileUnguardedSubrange(CC_List & ccs, Range & enclosingRange, PabloAST * enclosingTest, Range & subrange, PabloBuilder & pb) {
    CC_List subrangeCCs(ccs.size());
    extract_CCs_by_range(subrange, ccs, subrangeCCs);
    //  If there are no CCs that intersect the subrange, no code
    //  generation is required.
    Range actual_subrange = CC_Set_Range(subrangeCCs);
    if (actual_subrange.is_empty()) return;
    if (UTF_CompilationTracing) {
        llvm::errs() << "URC::compileUnguardedSubrange(" << enclosingRange.hex_string() << ") subrange(" << subrange.hex_string() << ")\n";
    }
    if (BixNumCCs) {
        for (unsigned i = 0; i < subrangeCCs.size(); i++) {
            for (const auto range : *subrangeCCs[i]) {
                Range r{re::lo_codepoint(range), re::hi_codepoint(range)};
                PabloAST * compiled = compileCodeRange(r, enclosingRange, enclosingTest, pb);
                pb.createAssign(mTargets[i], pb.createOr(mTargets[i], compiled));
            }
        }
    } else {
        unsigned significant_bits = enclosingRange.significant_bits();
        std::string alphabet = "Low" + std::to_string(significant_bits);
        cc::CodeUnitAlphabet CodeAlpha(alphabet, alphabet, significant_bits);
        codepoint_t mask = (1U << significant_bits) - 1;
        pablo::BixNumCompiler bnc(pb);
        BixNum truncated = bnc.Truncate(mBasis, significant_bits);
        std::unique_ptr<cc::CC_Compiler> truncatedCompiler;
        truncatedCompiler = std::make_unique<cc::Parabix_CC_Compiler_Builder>(truncated);
        for (unsigned i = 0; i < subrangeCCs.size(); i++) {
            re::CC * reducedCC = reduceCC(subrangeCCs[i], mask, CodeAlpha);
            PabloAST * compiled = truncatedCompiler->compileCC(reducedCC, pb);
            pb.createAssign(mTargets[i], pb.createOr(mTargets[i], pb.createAnd(compiled, enclosingTest)));
        }
    }
}

PabloAST * Unicode_Range_Compiler::compileCodeRange(Range & codepointRange, Range & enclosingRange, PabloAST * enclosing, PabloBuilder & pb) {
    pablo::BixNumCompiler bnc(pb);
    unsigned significant_bits = enclosingRange.significant_bits();
    codepoint_t mask = (1 << significant_bits) - 1;
    if (UTF_CompilationTracing) {
        llvm::errs() << "compileCodeRange(";
        llvm::errs().write_hex(codepointRange.lo);
        llvm::errs() << ", ";
        llvm::errs().write_hex(codepointRange.hi);
        llvm::errs() << ")\n";
        llvm::errs() << "  significant_bits = " << significant_bits << "\n";
    }
    BixNum truncated = bnc.Truncate(mBasis, significant_bits);
    codepoint_t lo = codepointRange.lo & mask;
    codepoint_t hi = codepointRange.hi & mask;
    if (lo == hi) {
        return pb.createAnd(enclosing, bnc.EQ(truncated, lo, "EQ_" + std::to_string(lo)));
    } else {
        PabloAST * lo_test = bnc.UGE(truncated, lo, "UGE_" + std::to_string(lo));
        PabloAST * hi_test = bnc.ULE(truncated, hi, "ULE_" + std::to_string(hi));
        return pb.createAnd3(enclosing, lo_test, hi_test);
    }
}

class New_UTF_Compiler {
public:
    New_UTF_Compiler(pablo::Var * Var, PabloBuilder & pb, pablo::PabloAST * mask);
    void compile(Target_List targets, CC_List ccs);
protected:
    pablo::Var *            mBasisVar;
    PabloBuilder &          mPB;
    pablo::PabloAST *       mMask;
    UTF_Encoder             mEncoder;
    Target_List             mTargets;
    //  Depending on the actual CC_List being compiled, up to
    //  4 scope positions will be defined, with corresponding basis
    //  sets and code unit compilers.
    Basis_Set               mScopeBasis[4];
    std::unique_ptr<cc::CC_Compiler> mCodeUnitCompilers[4];
    virtual void createInitialHierarchy(CC_List & ccs) = 0;
    virtual void prepareScope(unsigned scope, PabloBuilder & pb) = 0;
    virtual Basis_Set prepareUnifiedBasis(Range basis_range) = 0;
};

New_UTF_Compiler::New_UTF_Compiler(Var * basis_var, pablo::PabloBuilder & pb, pablo::PabloAST * mask) :
        mBasisVar(basis_var), mPB(pb), mMask(mask)  {
}


void New_UTF_Compiler::compile(Target_List targets, CC_List ccs) {
    //  Initialize all the target vars to 0.
    mTargets = targets;
    for (unsigned i = 0; i < targets.size(); i++) {
        mPB.createAssign(mTargets[i], mPB.createZeroes());
    }
    llvm::ArrayType * ty = cast<ArrayType>(mBasisVar->getType());
    unsigned streamCount = ty->getArrayNumElements();
    mScopeBasis[0].resize(streamCount);
    for (unsigned i = 0; i < streamCount; i++) {
        mScopeBasis[0][i] = mPB.createExtract(mBasisVar, mPB.getInteger(i));
    }
    mCodeUnitCompilers[0] =
        std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[0]);
    createInitialHierarchy(ccs);
}

PabloAST * combineOr(PabloAST * e1, PabloAST * e2, PabloBuilder & pb) {
    if (e1 == nullptr) return e2;
    if (e2 == nullptr) return e1;
    return pb.createOr(e1, e2);
}

PabloAST * combineAnd(PabloAST * e1, PabloAST * e2, PabloBuilder & pb) {
    if (e1 == nullptr) return e2;
    if (e2 == nullptr) return e1;
    return pb.createAnd(e1, e2);
}

class U21_Compiler : public New_UTF_Compiler {
public:
    U21_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mask) : New_UTF_Compiler(v, pb, mask) {
        mEncoder.setCodeUnitBits(21);
    }
protected:
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    Basis_Set prepareUnifiedBasis(Range basis_range) override;
    void createInitialHierarchy(CC_List & ccs) override;
};

void U21_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    //  The concept of scope is not relevant when a full set
    //  of 21 Unicode bit streams is provided.
}

Basis_Set U21_Compiler::prepareUnifiedBasis(Range basis_range) {
    Basis_Set basis(ceil_log2(basis_range.hi));
    for (unsigned i = 0; i < basis.size(); i++) {
        basis[i] = mScopeBasis[0][i];
    }
    return basis;
}

void U21_Compiler::createInitialHierarchy(CC_List & ccs) {
    if (UTF_CompilationTracing) {
        llvm::errs() << "U21_Compiler\n";
    }
    if (InitialNonASCIITest) {
        Range ASCII_Range{0, 0x7F};
        CC_List ASCII_ccs(ccs.size());
        extract_CCs_by_range(ASCII_Range, ccs, ASCII_ccs);
        Basis_Set ASCIIBasis = prepareUnifiedBasis(ASCII_Range);
        Unicode_Range_Compiler ASCII_compiler(ASCIIBasis, mTargets, mPB);
        PabloAST * e1 = mPB.createOr3(mScopeBasis[0][7], mScopeBasis[0][8], mScopeBasis[0][9]);
        PabloAST * e2 = mPB.createOr3(mScopeBasis[0][10], mScopeBasis[0][11], mScopeBasis[0][12]);
        PabloAST * e3 = mPB.createOr3(mScopeBasis[0][13], mScopeBasis[0][14], mScopeBasis[0][15]);
        PabloAST * e4 = mPB.createOr3(mScopeBasis[0][16], mScopeBasis[0][17], mScopeBasis[0][18]);
        PabloAST * e5 = mPB.createOr3(mScopeBasis[0][19], mScopeBasis[0][20], e1);
        PabloAST * e6 = mPB.createOr3(e2, e3, e4);
        PabloAST * nonASCII = mPB.createOr(e5, e6);
        ASCII_compiler.compile(ccs, ASCII_Range, mPB.createNot(nonASCII));
        auto nested = mPB.createScope();
        mPB.createIf(combineAnd(mMask, nonASCII, mPB), nested);
        Range nonASCII_Range{0x80, 0x10FFFF};
        CC_List nonASCII_ccs(ccs.size());
        extract_CCs_by_range(nonASCII_Range, ccs, nonASCII_ccs);
        Basis_Set UnifiedBasis = prepareUnifiedBasis(nonASCII_Range);
        Unicode_Range_Compiler range_compiler(UnifiedBasis, mTargets, nested);
        range_compiler.compile(nonASCII_ccs, nonASCII_Range, nonASCII);
    } else {
        Range UnicodeRange{0, 0x10FFFF};
        Basis_Set UnifiedBasis = prepareUnifiedBasis(UnicodeRange);
        if (mMask) {
            auto nested = mPB.createScope();
            mPB.createIf(mMask, nested);
            Unicode_Range_Compiler range_compiler(UnifiedBasis, mTargets, nested);
            range_compiler.compile(ccs, UnicodeRange, mMask);
        } else {
            Unicode_Range_Compiler range_compiler(UnifiedBasis, mTargets, mPB);
            range_compiler.compile(ccs, UnicodeRange, mPB.createOnes());
        }
    }
}

enum U8_Seq_Kind : unsigned {ASCII, TwoByte, ThreeByte, FourByte};
std::vector<Range> UTF8_Range =
    {Range{0, 0x7F}, Range{0x80, 0x7FF}, Range{0x800, 0xFFFF}, Range{0x10000, 0x10FFFF}};

struct SeqData {
    CC_List                 seqCCs;
    Range                   actualRange;
    re::CC *                byte1CC;
    PabloAST *              test;
    PabloAST *              combinedTest;
    PabloAST *              suffixTest;
};

class U8_Compiler : public New_UTF_Compiler {
public:
    U8_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mask) : New_UTF_Compiler(v, pb, mask) {
        mEncoder.setCodeUnitBits(8);
    }
protected:
    SeqData                 mSeqData[4];
    re::CC * prefixCC(CC_List & ccs);
    void lengthAnalysis(CC_List & ccs);
    void preparePrefixTests(PabloAST * enclosing_test, PabloBuilder & pb);
    void createInitialHierarchy(CC_List & ccs) override;
    void extendLengthHierarchy(PabloAST * enclosingTest, PabloBuilder & pb);
    void prepareFixedLengthHierarchy(U8_Seq_Kind k, PabloAST * enclosingTest, PabloBuilder & pb);
    PabloAST * compilePrefix(re::CC * prefixCC, PabloBuilder & pb);
};

re::CC * U8_Compiler::prefixCC(CC_List & ccs) {
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

void U8_Compiler::lengthAnalysis(CC_List & ccs) {
    for (unsigned k = ASCII; k <= FourByte; k++) {
        mSeqData[k].seqCCs.resize(ccs.size());
        extract_CCs_by_range(UTF8_Range[k], ccs, mSeqData[k].seqCCs);
        mSeqData[k].actualRange = CC_Set_Range(mSeqData[k].seqCCs);
        mSeqData[k].byte1CC = prefixCC(mSeqData[k].seqCCs);
        mSeqData[k].test = nullptr;
        mSeqData[k].suffixTest = nullptr;
    }
}

PabloAST * U8_Compiler::compilePrefix(re::CC * prefixCC, PabloBuilder & pb) {
    return mCodeUnitCompilers[0]->compileCC(prefixCC, pb);
}

void U8_Compiler::preparePrefixTests(PabloAST * enclosing_test, PabloBuilder & pb) {
    for (unsigned k = TwoByte; k <= FourByte; k++) {
        if (mSeqData[k].actualRange.is_empty()) {
            mSeqData[k].test = nullptr;
        } else {
            mSeqData[k].test = compilePrefix(mSeqData[k].byte1CC, pb);
        }
    }
    mSeqData[FourByte].combinedTest = mSeqData[FourByte].test;
    mSeqData[ThreeByte].combinedTest = combineOr(mSeqData[ThreeByte].test, mSeqData[FourByte].combinedTest, pb);
    mSeqData[TwoByte].combinedTest = combineOr(mSeqData[TwoByte].test, mSeqData[ThreeByte].combinedTest, pb);
}


void U8_Compiler::createInitialHierarchy(CC_List & ccs) {
    lengthAnalysis(ccs);
    if (!mSeqData[ASCII].actualRange.is_empty()) {
        PabloAST * ASCII_test = mPB.createNot(mScopeBasis[0][7]);
        Basis_Set ASCIIBasis = prepareUnifiedBasis(UTF8_Range[ASCII]);
        Unicode_Range_Compiler ASCII_compiler(ASCIIBasis, mTargets, mPB);
        ASCII_compiler.compile(ccs, UTF8_Range[ASCII], ASCII_test);
    }
    CC_List nonASCII_ccs(ccs.size());
    Range nonASCII_Range{0x80, 0x10FFFF};
    extract_CCs_by_range(nonASCII_Range, ccs, nonASCII_ccs);
    Range actual_subrange = CC_Set_Range(nonASCII_ccs);
    if (!actual_subrange.is_empty()) {
        if (InitialNonASCIITest) {
            PabloAST * nonASCII_test = combineAnd(mMask, mScopeBasis[0][7], mPB);
            auto nested = mPB.createScope();
            mPB.createIf(nonASCII_test, nested);
            preparePrefixTests(nonASCII_test, nested);
            extendLengthHierarchy(nonASCII_test, nested);
        } else if (!PrefixCCTest) {
            unsigned lo_prefix = mEncoder.nthCodeUnit(actual_subrange.lo, 1);
            unsigned hi_prefix = mEncoder.nthCodeUnit(actual_subrange.hi, 1);
            re::CC * prefix_range_CC = re::makeCC(lo_prefix, hi_prefix, &Byte);
            PabloAST * prefix_range_test = combineAnd(mMask, compilePrefix(prefix_range_CC, mPB), mPB);
            auto nested = mPB.createScope();
            mPB.createIf(prefix_range_test, nested);
            preparePrefixTests(prefix_range_test, nested);
            extendLengthHierarchy(prefix_range_test, nested);
        } else {
            preparePrefixTests(mMask, mPB);
            auto nested = mPB.createScope();
            mPB.createIf(mSeqData[TwoByte].combinedTest, nested);
            extendLengthHierarchy(mSeqData[TwoByte].combinedTest, nested);
        }
    }
}

void U8_Compiler::extendLengthHierarchy(PabloAST * enclosingTest, PabloBuilder & pb) {
    if (!mSeqData[TwoByte].byte1CC->empty()) {
        prepareScope(TwoByte, pb);
        prepareFixedLengthHierarchy(TwoByte, enclosingTest, pb);
    }
    PabloAST * outer_test = mSeqData[ThreeByte].combinedTest;
    if (outer_test == nullptr) return;
    auto nested = pb.createScope();
    pb.createIf(outer_test, nested);
    if (!mSeqData[ThreeByte].byte1CC->empty()) {
        prepareScope(ThreeByte, nested);
        prepareFixedLengthHierarchy(ThreeByte, outer_test, nested);
    }
    if (!mSeqData[FourByte].byte1CC->empty()) {
        prepareScope(FourByte, nested);
        prepareFixedLengthHierarchy(FourByte, outer_test, nested);
    }
}

void U8_Compiler::prepareFixedLengthHierarchy(U8_Seq_Kind k, PabloAST * enclosingTest, PabloBuilder & pb) {
    // No code generation if there are no CCs within this range.
    if (mSeqData[k].actualRange.is_empty()) return;
    codepoint_t test_lo = mEncoder.minCodePointWithCommonCodeUnits(mSeqData[k].actualRange.lo, 1);
    codepoint_t test_hi = mEncoder.maxCodePointWithCommonCodeUnits(mSeqData[k].actualRange.hi, 1);
    Range testRange{test_lo, test_hi};
    if (mSeqData[k].test == enclosingTest) {
        // No further test needed.
        Basis_Set UnifiedBasis = prepareUnifiedBasis(testRange);
        Unicode_Range_Compiler range_compiler(UnifiedBasis, mTargets, pb);
        range_compiler.compile(mSeqData[k].seqCCs, testRange, mSeqData[k].test);
    } else {
        auto nested = pb.createScope();
        pb.createIf(mSeqData[k].test, nested);
        Basis_Set UnifiedBasis = prepareUnifiedBasis(testRange);
        Unicode_Range_Compiler range_compiler(UnifiedBasis, mTargets, nested);
        range_compiler.compile(mSeqData[k].seqCCs, testRange, mSeqData[k].test);
    }
}

class U8_Lookahead_Compiler : public U8_Compiler {
public:
    U8_Lookahead_Compiler(pablo::Var * Var, PabloBuilder & pb, pablo::PabloAST * mask);
protected:
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    Basis_Set prepareUnifiedBasis(Range basis_range) override;
};

class U8_Advance_Compiler : public U8_Compiler {
public:
    U8_Advance_Compiler(pablo::Var * Var, PabloBuilder & pb, pablo::PabloAST * mask);
protected:
    PabloAST * mSuffix;
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    Basis_Set prepareUnifiedBasis(Range basis_range) override;
};

U8_Lookahead_Compiler::U8_Lookahead_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mask) :
U8_Compiler(v, pb, mask) {
}

void U8_Lookahead_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    for (unsigned sfx = 1; sfx <= scope; sfx++) {
        bool basis_needed = mScopeBasis[sfx].size() == 0;
        if (basis_needed) {
            mScopeBasis[sfx].resize(mScopeBasis[0].size());
            for (unsigned i = 0; i < mScopeBasis[0].size(); i++) {
                mScopeBasis[sfx][i] = pb.createLookahead(mScopeBasis[0][i], sfx);
            }
            PabloAST * sfxmark = pb.createAnd(pb.createNot(mScopeBasis[sfx][6]), mScopeBasis[sfx][7]);
            if (sfx > 1) {
                mSeqData[sfx].suffixTest = pb.createAnd(mSeqData[sfx-1].suffixTest, sfxmark);
            } else {
                mSeqData[sfx].suffixTest = sfxmark;
            }
            mSeqData[scope].test = pb.createAnd(mSeqData[scope].test, mSeqData[sfx].suffixTest);
            mCodeUnitCompilers[sfx] =
            std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[sfx]);
        }
    }
}

Basis_Set U8_Lookahead_Compiler::prepareUnifiedBasis(Range cc_range) {
    unsigned lgth = mEncoder.encoded_length(cc_range.hi);
    unsigned total_bits = ceil_log2(mEncoder.max_codepoint_of_length(lgth));
    unsigned variable_bits = ceil_log2(cc_range.lo ^ cc_range.hi);
    if (UTF_CompilationTracing) {
        llvm::errs() << "prepareUnifiedBasis(" << cc_range.hex_string() << ")\n";
        llvm::errs() << "  total_bits = " << total_bits << "\n";
        llvm::errs() << "  variable_bits = " << variable_bits << "\n";
    }
    Basis_Set UnifiedBasis(total_bits);
    unsigned bits_per_unit = (lgth == 1) ? 7 : 6;
    unsigned max_suffix = lgth - 1;
    for (unsigned i = 0; i < variable_bits; i++) {
        unsigned suffix_pos = max_suffix - i/bits_per_unit;
        unsigned scope_bit = i % bits_per_unit;
        UnifiedBasis[i] = mScopeBasis[suffix_pos][scope_bit];
    }
    for (unsigned i = variable_bits; i < total_bits; i++) {
        unsigned fixed_bit_val = (cc_range.lo >> i) & 1;
        if (fixed_bit_val == 1) {
            UnifiedBasis[i] = mPB.createOnes();
        } else {
            UnifiedBasis[i] = mPB.createZeroes();
        }
    }
    return UnifiedBasis;
}

U8_Advance_Compiler::U8_Advance_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mMask) :
U8_Compiler(v, pb, mMask), mSuffix(nullptr) {
}

void U8_Advance_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    if (mSeqData[scope].byte1CC->empty()) return;
    if (mSuffix == nullptr) {
        mSuffix = pb.createAnd(mScopeBasis[0][7], pb.createNot(mScopeBasis[0][6]), "mSuffix");
        mSeqData[1].suffixTest = mSuffix;
    }
    for (unsigned sfx = 2; sfx <= scope; sfx++) {
        if (mSeqData[sfx].suffixTest == nullptr) {
            PabloAST * adv_suffix = pb.createAdvance(mSeqData[sfx-1].suffixTest, 1);
            mSeqData[sfx].suffixTest = pb.createAnd(adv_suffix, mSuffix, "scope" + std::to_string(sfx) + "_sfx_test");
        }
    }
    // For each prior suffix, we need 6 data bits.
    for (unsigned sfx = 1; sfx < scope; sfx++) {
        unsigned sfx_bits = mScopeBasis[sfx].size();
        if (sfx_bits < 6) {
            mScopeBasis[sfx].resize(6);
            for (unsigned i = sfx_bits; i < 6; i++) {
                mScopeBasis[sfx][i] = pb.createAdvance(mScopeBasis[0][i], sfx);
            }
        }
    }
    codepoint_t test_lo = mEncoder.minCodePointWithCommonCodeUnits(mSeqData[scope].actualRange.lo, 1);
    codepoint_t test_hi = mEncoder.maxCodePointWithCommonCodeUnits(mSeqData[scope].actualRange.hi, 1);
    Range testRange{test_lo, test_hi};
    unsigned variable_bits = testRange.significant_bits();
    unsigned prefix_bits = 0;
    if (variable_bits > scope * 6) {
        prefix_bits = variable_bits - scope * 6;
        mScopeBasis[scope].resize(prefix_bits);
        for (unsigned i = 0; i < prefix_bits; i++) {
            mScopeBasis[scope][i] = pb.createAdvance(mScopeBasis[0][i], scope);
        }
    }
    if (UTF_CompilationTracing) {
        llvm::errs() << "scope = " << scope << ", variable_bits = " << variable_bits << ", prefix_bits = " << prefix_bits << "\n";
    }
    //
    mSeqData[scope].test = pb.createAnd(mSeqData[scope].suffixTest, pb.createAdvance(mSeqData[scope].test, scope));
}


Basis_Set U8_Advance_Compiler::prepareUnifiedBasis(Range cc_range) {
    unsigned lgth = mEncoder.encoded_length(cc_range.hi);
    unsigned total_bits = cc_range.total_bits(); //ceil_log2(mEncoder.max_codepoint_of_length(lgth));
    unsigned variable_bits = ceil_log2(cc_range.lo ^ cc_range.hi);
    if (UTF_CompilationTracing) {
        llvm::errs() << "prepareUnifiedBasis(" << cc_range.hex_string() << ")\n";
        llvm::errs() << "  total_bits = " << total_bits << "\n";
        llvm::errs() << "  variable_bits = " << variable_bits << "\n";
    }
    Basis_Set UnifiedBasis(total_bits);
    unsigned bits_per_unit = (lgth == 1) ? 7 : 6;
    for (unsigned i = 0; i < variable_bits; i++) {
        unsigned u8_pos = i/bits_per_unit;
        unsigned scope_bit = i % bits_per_unit;
        UnifiedBasis[i] = mScopeBasis[u8_pos][scope_bit];
    }
    for (unsigned i = variable_bits; i < total_bits; i++) {
        unsigned fixed_bit_val = (cc_range.lo >> i) & 1;
        if (fixed_bit_val == 1) {
            UnifiedBasis[i] = mPB.createOnes();
        } else {
            UnifiedBasis[i] = mPB.createZeroes();
        }
    }
    return UnifiedBasis;
}

UTF_Compiler::UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
             pablo::PabloAST * mask,
             pablo::BitMovementMode mode) :
mVar(basisVar), mPB(pb), mMask(mask), mBitMovement(mode) {
}

UTF_Compiler::UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
             pablo::BitMovementMode mode) :
mVar(basisVar), mPB(pb), mMask(nullptr), mBitMovement(mode) {}

void UTF_Compiler::compile(Target_List targets, CC_List ccs) {
    llvm::ArrayType * ty = cast<ArrayType>(mVar->getType());
    unsigned streamCount = ty->getArrayNumElements();
    unsigned codeUnitBits;
    if (streamCount == 1) {
        VectorType * const vt = cast<VectorType>(ty->getArrayElementType());
        const auto streamWidth = vt->getElementType()->getIntegerBitWidth();
        codeUnitBits = streamWidth;
        UTF_Legacy_Compiler utf_compiler(mVar, mPB, mMask);
        for (unsigned i = 0; i < targets.size(); i++) {
            utf_compiler.addTarget(targets[i], ccs[i]);
        }
        utf_compiler.compile();
        return;
    }
    codeUnitBits = streamCount;
    if (codeUnitBits == 8) {
        if (mBitMovement == pablo::BitMovementMode::LookAhead) {
            U8_Lookahead_Compiler utf_compiler(mVar, mPB, mMask);
            utf_compiler.compile(targets, ccs);
        } else if (!UseLegacyUTFHierarchy) {
            U8_Advance_Compiler utf_compiler(mVar, mPB, mMask);
            utf_compiler.compile(targets, ccs);
        } else {
            UTF_Legacy_Compiler utf_compiler(mVar, mPB, mMask);
            for (unsigned i = 0; i < targets.size(); i++) {
                utf_compiler.addTarget(targets[i], ccs[i]);
            }
            utf_compiler.compile();
        }
    } else {
        if (UseLegacyUTFHierarchy) {
            UTF_Legacy_Compiler utf_compiler(mVar, mPB, mMask);
            for (unsigned i = 0; i < targets.size(); i++) {
                utf_compiler.addTarget(targets[i], ccs[i]);
            }
            utf_compiler.compile();
        } else {
            U21_Compiler u21_compiler(mVar, mPB, mMask);
            u21_compiler.compile(targets, ccs);
        }
    }
}

}
