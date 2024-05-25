/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <grep/grep_engine.h>

#include <atomic>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <sched.h>
#include <boost/filesystem.hpp>
#include <toolchain/toolchain.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Casting.h>
#include <grep/regex_passes.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/core/idisa_target.h>
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/core/callback.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <kernel/unicode/boundary_kernels.h>
#include <re/unicode/resolve_properties.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/util/linebreak_kernel.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/streamutils/until_n.h>
#include <kernel/streamutils/sentinel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <pablo/pablo_kernel.h>
#include <re/adt/adt.h>
#include <re/adt/re_utility.h>
#include <re/printer/re_printer.h>
#include <re/alphabet/alphabet.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/re_name_gather.h>
#include <re/analysis/capture-ref.h>
#include <re/analysis/collect_ccs.h>
#include <re/cc/cc_kernel.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/re_contextual_simplification.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/to_utf8.h>
#include <re/transforms/remove_nullable.h>
#include <re/transforms/replaceCC.h>
#include <re/transforms/re_multiplex.h>
#include <re/transforms/expand_permutes.h>
#include <re/transforms/name_intro.h>
#include <re/transforms/reference_transform.h>
#include <re/transforms/variable_alt_promotion.h>
#include <re/unicode/casing.h>
#include <re/unicode/boundaries.h>
#include <re/unicode/re_name_resolve.h>
#include <unicode/data/PropertyObjectTable.h>
#include <sys/stat.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <grep/grep_toolchain.h>
#include <toolchain/toolchain.h>
#include <sys/mman.h>
#include <util/aligned_allocator.h>

using namespace llvm;
using namespace cc;
using namespace kernel;

namespace grep {

using UntilNMode = UntilNkernel::Mode;

static cl::opt<UntilNMode>
MaxLimitTerminationMode("maxlimit-termination-mode",
                  cl::init(UntilNMode::TerminateAtN),
                  cl::desc("method of pipeline termination when -m=maxlimit is reached."),
                  cl::values(clEnumValN(UntilNMode::ReportAcceptedLengthAtAndBeforeN, "report", "halt pipeline after maxlimit using truncated streamset"),
                             clEnumValN(UntilNMode::TerminateAtN, "terminate", "halt pipeline after maxlimit using streamset copy"),
                             clEnumValN(UntilNMode::ZeroAfterN, "zero", "fully process the file")));


const auto ENCODING_BITS = 8;

void GrepCallBackObject::handle_signal(unsigned s) {
    if (static_cast<GrepSignal>(s) == GrepSignal::BinaryFile) {
        mBinaryFile = true;
    } else {
        llvm::report_fatal_error("Unknown GrepSignal");
    }
}

extern "C" void accumulate_match_wrapper(intptr_t accum_addr, const size_t lineNum, char * line_start, char * line_end) {
    assert ("passed a null accumulator" && accum_addr);
    reinterpret_cast<MatchAccumulator *>(accum_addr)->accumulate_match(lineNum, line_start, line_end);
}

extern "C" void finalize_match_wrapper(intptr_t accum_addr, char * buffer_end) {
    assert ("passed a null accumulator" && accum_addr);
    reinterpret_cast<MatchAccumulator *>(accum_addr)->finalize_match(buffer_end);
}

extern "C" unsigned get_file_count_wrapper(intptr_t accum_addr) {
    assert ("passed a null accumulator" && accum_addr);
    return reinterpret_cast<MatchAccumulator *>(accum_addr)->getFileCount();
}

extern "C" size_t get_file_start_pos_wrapper(intptr_t accum_addr, unsigned fileNo) {
    assert ("passed a null accumulator" && accum_addr);
    return reinterpret_cast<MatchAccumulator *>(accum_addr)->getFileStartPos(fileNo);
}

extern "C" void set_batch_line_number_wrapper(intptr_t accum_addr, unsigned fileNo, size_t batchLine) {
    assert ("passed a null accumulator" && accum_addr);
    reinterpret_cast<MatchAccumulator *>(accum_addr)->setBatchLineNumber(fileNo, batchLine);
}

// Grep Engine construction and initialization.

GrepEngine::GrepEngine(BaseDriver &driver) :
    mSuppressFileMessages(false),
    mBinaryFilesMode(argv::Text),
    mPreferMMap(false),
    mColoring(false),
    mShowFileNames(false),
    mStdinLabel("(stdin)"),
    mShowLineNumbers(false),
    mBeforeContext(0),
    mAfterContext(0),
    mInitialTab(false),
    mCaseInsensitive(false),
    mInvertMatches(false),
    mMaxCount(0),
    mGrepStdIn(false),
    mNullMode(NullCharMode::Data),
    mGrepDriver(driver),
    mMainMethod(nullptr),
    mNextFileToGrep(0),
    mNextFileToPrint(0),
    grepMatchFound(false),
    mGrepRecordBreak(GrepRecordBreakKind::LF),
    mExternalComponents(static_cast<Component>(0)),
    mInternalComponents(static_cast<Component>(0)),
    mIndexAlphabet(&cc::UTF8),
    mLineBreakStream(nullptr),
    mU8index(nullptr),
    mEngineThread(pthread_self()) {

    }

GrepEngine::~GrepEngine() { }

QuietModeEngine::QuietModeEngine(BaseDriver &driver) : GrepEngine(driver) {
    mEngineKind = EngineKind::QuietMode;
    mMaxCount = 1;
}

MatchOnlyEngine::MatchOnlyEngine(BaseDriver & driver, bool showFilesWithMatch, bool useNullSeparators) :
    GrepEngine(driver), mRequiredCount(showFilesWithMatch) {
    mEngineKind = EngineKind::MatchOnly;
    mFileSuffix = useNullSeparators ? std::string("\0", 1) : "\n";
    mMaxCount = 1;
    mShowFileNames = true;
}

CountOnlyEngine::CountOnlyEngine(BaseDriver &driver) : GrepEngine(driver) {
    mEngineKind = EngineKind::CountOnly;
    mFileSuffix = ":";
}

EmitMatchesEngine::EmitMatchesEngine(BaseDriver &driver)
: GrepEngine(driver) {
    mEngineKind = EngineKind::EmitMatches;
    mFileSuffix = mInitialTab ? "\t:" : ":";
}

bool GrepEngine::hasComponent(Component compon_set, Component c) {
    return (static_cast<component_t>(compon_set) & static_cast<component_t>(c)) != 0;
}

void GrepEngine::GrepEngine::setComponent(Component & compon_set, Component c) {
    compon_set = static_cast<Component>(static_cast<component_t>(compon_set) | static_cast<component_t>(c));
}

void GrepEngine::setRecordBreak(GrepRecordBreakKind b) {
    mGrepRecordBreak = b;
}

namespace fs = boost::filesystem;

std::vector<std::vector<std::string>> formFileGroups(std::vector<fs::path> paths) {
    const unsigned maxFilesPerGroup = 32;
    const uintmax_t FileBatchThreshold = 4 * codegen::SegmentSize;
    std::vector<std::vector<std::string>> groups;
    // The total size of files in the current group, or 0 if the
    // the next file should start a new group.
    uintmax_t groupTotalSize = 0;
    for (auto p : paths) {
        boost::system::error_code errc;
        auto s = fs::file_size(p, errc);
        if ((s > 0) && (s < FileBatchThreshold)) {
            if (groupTotalSize == 0) {
                groups.push_back({p.string()});
                groupTotalSize = s;
            } else {
                groups.back().push_back(p.string());
                groupTotalSize += s;
                if ((groupTotalSize > FileBatchThreshold) || (groups.back().size() == maxFilesPerGroup)) {
                    // Signal to start a new group
                    groupTotalSize = 0;
                }
            }
        } else {
            // For large files, or in the case of non-regular file or other error,
            // the path is saved in its own group.
            groups.push_back({p.string()});
            // This group is done, signal to start a new group.
            groupTotalSize = 0;
        }
    }
    return groups;
}

void GrepEngine::initFileResult(const std::vector<boost::filesystem::path> & paths) {
    const unsigned n = paths.size();
    mResultStrs.resize(n);
    mFileStatus.resize(n, FileStatus::Pending);
    mInputPaths = paths;
    mFileGroups = formFileGroups(paths);
    const unsigned numOfThreads = std::min(static_cast<unsigned>(codegen::TaskThreads),
                                           std::max(static_cast<unsigned>(mFileGroups.size()), 1u));
    codegen::setTaskThreads(numOfThreads);
}

//
// Moving matches to EOL.   Mathches need to be aligned at EOL if for
// scanning or counting processes (with a max count != 1).   If the REs
// are not all anchored, then we need to move the matches to EOL.
bool GrepEngine::matchesToEOLrequired () {
    // Moving matches is required for UnicodeLines mode, because matches
    // may be on the CR of a CRLF.
    if (mGrepRecordBreak == GrepRecordBreakKind::Unicode) return true;
    // If all REs are anchored to EOL already, then we can avoid moving them.
    if (hasEndAnchor(mRE)) return false;
    //
    // Not all REs are anchored.   We can avoid moving matches, if we are
    // in MatchOnly mode (or CountOnly with MaxCount = 1) and no invert match inversion.
    return (mEngineKind == EngineKind::EmitMatches) || (mMaxCount != 1) || mInvertMatches;
}

void GrepEngine::initRE(re::RE * re) {
    if ((mEngineKind != EngineKind::EmitMatches) || mInvertMatches) {
        mColoring = false;
    }
    mRE = expandPermutes(re);
    mRE = resolveModesAndExternalSymbols(mRE, mCaseInsensitive);
    // Determine the unit of length for the RE.  If the RE involves
    // fixed length UTF-8 sequences only, then UTF-8 can be used
    // for most efficient processing.   Otherwise we must use full
    // Unicode length calculations.
    mLengthAlphabet = &cc::UTF8;
    if (!validateFixedUTF8(mRE) || (mGrepRecordBreak == GrepRecordBreakKind::Unicode)) {
        setComponent(mExternalComponents, Component::UTF8index);
        mLengthAlphabet = &cc::Unicode;
    }
    StreamIndexCode u8 = mExternalTable.declareStreamIndex("u8");
    StreamIndexCode Unicode = mExternalTable.declareStreamIndex("U", u8, "u8index");

    mRefInfo = re::buildReferenceInfo(mRE);
    if (!mRefInfo.twixtREs.empty()) {
        UnicodeIndexing = true;
        auto indexCode = mExternalTable.getStreamIndex(cc::Unicode.getCode());
        setComponent(mExternalComponents, Component::S2P);
        re::FixedReferenceTransformer FRT(mRefInfo);
        mRE = FRT.transformRE(mRE);
        for (auto m : FRT.mNameMap) {
            re::Reference * ref = cast<re::Reference>(m.second);
            UCD::property_t p = ref->getReferencedProperty();
            std::string instanceName = ref->getInstanceName();
            auto captureLen = getLengthRange(ref->getCapture(), &cc::Unicode).first;
            if (captureLen != 1) {
                llvm::report_fatal_error("Capture length > 1 is a future extension");
            }
            auto mapping = mRefInfo.twixtREs.find(instanceName);
            auto twixtLen = getLengthRange(mapping->second, &cc::Unicode).first;
            auto dist = captureLen + twixtLen;
            mExternalTable.declareExternal(indexCode, m.first, new PropertyDistanceExternal(p, dist));
            UCD::PropertyObject * propObj = UCD::getPropertyObject(p);
            if (isa<UCD::EnumeratedPropertyObject>(propObj)) {
                std::string extName = UCD::getPropertyFullName(p) + "_basis";
                mExternalTable.declareExternal(indexCode, extName, new PropertyBasisExternal(p));
            } else if (isa<UCD::CodePointPropertyObject>(propObj)) {
                // Identity or other codepoint properties
                auto u8_u21 = new U21_External();
                mExternalTable.declareExternal(u8, "u21", u8_u21);
                mExternalTable.declareExternal(Unicode, "basis", new FilterByMaskExternal(u8, {"u8index", "u21"}, u8_u21));
            }
        }
    }
    if (mColoring) {
        mRE = zeroBoundElimination(mRE);
        mRE = variableAltPromotion(mRE, mLengthAlphabet);
    } else {
        mRE = remove_nullable_ends(mRE);
    }
    mRE = regular_expression_passes(mRE);
    UCD::PropertyExternalizer PE;
    mRE = PE.transformRE(mRE);
    for (auto m : PE.mNameMap) {
        if (re::PropertyExpression * pe = dyn_cast<re::PropertyExpression>(m.second)) {
            if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
                re::RE * propRE = pe->getResolvedRE();
                if (getLengthRange(propRE, &cc::UTF8).second > 1) {
                    mLengthAlphabet = &cc::Unicode;
                    break;
                }
            }
        }
    }
    auto lgth_range = getLengthRange(mRE, mLengthAlphabet);
    // For length 0 regular expressions (e.g. a zero-width assertion like $)
    // there will be no match spans to color.
    if (lgth_range.second == 0) mColoring = false;
    if ((mLengthAlphabet == &cc::Unicode) && mColoring) {
        mExternalTable.declareExternal(u8, "LineStarts", new LineStartsExternal());
        UnicodeIndexing = true;
    }
    if (UnicodeIndexing) {
        mIndexAlphabet = &cc::Unicode;
        setComponent(mExternalComponents, Component::S2P);
        setComponent(mExternalComponents, Component::UTF8index);
        if (!mExternalTable.isDeclared(Unicode, "basis")) {
            const auto UnicodeSets = re::collectCCs(mRE, *mIndexAlphabet);
            if (!UnicodeSets.empty()) {
                auto mpx = makeMultiplexedAlphabet("mpx", UnicodeSets);
                mRE = transformCCs(mpx, mRE, re::NameTransformationMode::None);
                mExternalTable.declareExternal(Unicode, mpx->getName() + "_basis", new MultiplexedExternal(mpx));
            }
        }
    }
    auto indexCode = mExternalTable.getStreamIndex(mIndexAlphabet->getCode());
    if (hasGraphemeClusterBoundary(mRE)) {
        auto GCB_basis = new PropertyBasisExternal(UCD::GCB);
        mExternalTable.declareExternal(u8, "UCD:" + getPropertyFullName(UCD::GCB) + "_basis", GCB_basis);
        re::RE * epict_pe = UCD::linkAndResolve(re::makePropertyExpression("Extended_Pictographic"));
        re::Name * epict = cast<re::Name>(UCD::externalizeProperties(epict_pe));
        mExternalTable.declareExternal(u8, epict->getFullName(), new PropertyExternal(epict));
        auto u8_GCB = new GraphemeClusterBreak(this, &cc::UTF8);
        mExternalTable.declareExternal(u8, "\\b{g}", u8_GCB);
        if (indexCode == Unicode) {
            mExternalTable.declareExternal(Unicode, "\\b{g}", new FilterByMaskExternal(u8, {"u8index","\\b{g}"}, u8_GCB));
        }
    }
    for (auto m : PE.mNameMap) {
        if (re::PropertyExpression * pe = dyn_cast<re::PropertyExpression>(m.second)) {
            if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
                mExternalTable.declareExternal(indexCode, m.first, new PropertyExternal(re::makeName(m.first, m.second)));
            } else { //PropertyExpression::Kind::Boundary
                UCD::property_t prop = static_cast<UCD::property_t>(pe->getPropertyCode());
                if (prop != UCD::g) {  // Boundary expressions, except GCB.
                    auto prop_basis = new PropertyBasisExternal(prop);
                    mExternalTable.declareExternal(indexCode, getPropertyFullName(prop) + "_basis", prop_basis);
                    auto boundary = new PropertyBoundaryExternal(prop);
                    mExternalTable.declareExternal(indexCode, m.first, boundary);
                }
           }
        } else {
            llvm::report_fatal_error("Expected property expression");
        }
    }
    if (mIndexAlphabet == &cc::UTF8) {
        bool useInternalNaming = mLengthAlphabet == &cc::Unicode;
        mRE = toUTF8(mRE, useInternalNaming);
    }
    re::VariableLengthCCNamer CCnamer;
    mRE = CCnamer.transformRE(mRE);
    for (auto m : CCnamer.mNameMap) {
        mExternalTable.declareExternal(indexCode, m.first, new CC_External(cast<re::CC>(m.second)));
    }
    if (hasWordBoundary(mRE)) {
        mExternalTable.declareExternal(indexCode, "\\b", new WordBoundaryExternal());
    }
    if ((mEngineKind == EngineKind::EmitMatches) && mColoring && !mInvertMatches) {
        setComponent(mExternalComponents, Component::MatchSpans);
    }
    if (matchesToEOLrequired()) {
        // Move matches to EOL.   This may be achieved internally by modifying
        // the regular expression or externally.   The internal approach is more
        // generally more efficient, but cannot be used if colorization is needed
        // or in UnicodeLines mode.
        if ((mGrepRecordBreak == GrepRecordBreakKind::Unicode) || (mEngineKind == EngineKind::EmitMatches) || mInvertMatches || UnicodeIndexing) {
            setComponent(mExternalComponents, Component::MoveMatchesToEOL);
        } else {
            setComponent(mInternalComponents, Component::MoveMatchesToEOL);
        }
    }
    if (hasComponent(mInternalComponents, Component::MoveMatchesToEOL)) {
        if (!hasEndAnchor(mRE)) {
            mRE = re::makeSeq({mRE, re::makeRep(re::makeAny(mLengthAlphabet), 0, re::Rep::UNBOUNDED_REP), re::makeEnd()});
        }
    }
    if (mColoring) {
        auto indexing = mExternalTable.getStreamIndex(mIndexAlphabet->getCode());
        re::FixedSpanNamer FLnamer(mIndexAlphabet);
        mRE = FLnamer.transformRE(mRE);
        for (auto m : FLnamer.mNameMap) {
            auto r = new RE_External(this, m.second, mIndexAlphabet);
            auto lgth = r->getLengthRange().first;
            auto offset = r->getOffset();
            mExternalTable.declareExternal(indexing, m.first, r);
            if (lgth > 0) {
                auto spanName = m.first + "Span";
                mExternalTable.declareExternal(indexing, spanName, new FixedSpanExternal(m.first, lgth, offset));
                mSpanNames.push_back(spanName);
            }
        }
        re::UniquePrefixNamer UPnamer;
        mRE = UPnamer.transformRE(mRE);
        for (auto m : UPnamer.mNameMap) {
            std::string nameStr = m.first;
            re::RE * namedRE = m.second;
            re::Name * prefixName = cast<re::Name>(cast<re::Seq>(namedRE)->front());
            std::string prefixStr = prefixName->getFullName();
            auto pfxExternal = new RE_External(this, prefixName->getDefinition(), mIndexAlphabet);
            mExternalTable.declareExternal(indexing, prefixStr, pfxExternal);
            auto fullExt = new RE_External(this, namedRE, mIndexAlphabet);
            mExternalTable.declareExternal(indexing, nameStr, fullExt);
            auto prefixLgth = pfxExternal->getLengthRange().first + pfxExternal->getOffset();
            auto offset = fullExt->getOffset();
            auto spanName = nameStr + "Span";
            mExternalTable.declareExternal(indexing, spanName, new MarkedSpanExternal(prefixStr, prefixLgth, nameStr, offset));
            mSpanNames.push_back(spanName);
        }
        re::Repeated_CC_Seq_Namer RCCSnamer;
        mRE = RCCSnamer.transformRE(mRE);
        for (auto m : RCCSnamer.mNameMap) {
            std::string nameStr = m.first;
            re::RE * namedRE = m.second;
            auto r = new RE_External(this, namedRE, mIndexAlphabet);
            mExternalTable.declareExternal(indexing, nameStr, r);
            auto f = RCCSnamer.mInfoMap.find(nameStr);
            if (f != RCCSnamer.mInfoMap.end()) {
                const re::CC * varCC = f->second.first;
                unsigned fixed= f->second.second;
                auto maskName = nameStr + "mask";
                auto e1 = new CCmask(mIndexAlphabet, varCC);
                mExternalTable.declareExternal(indexing, maskName, e1);
                auto spanName = nameStr + "Span";
                auto e2 = new MaskedFixedSpanExternal(maskName, nameStr, fixed, grepOffset(namedRE));
                mExternalTable.declareExternal(indexing, spanName, e2);
                mSpanNames.push_back(spanName);
            }
        }
    }
    if (mLengthAlphabet == &cc::Unicode) {
        setComponent(mExternalComponents, Component::S2P);
        setComponent(mExternalComponents, Component::UTF8index);
    } else if (!byteTestsWithinLimit(mRE, ByteCClimit)) {
        setComponent(mExternalComponents, Component::S2P);
    }
}

StreamSet * GrepEngine::getBasis(ProgBuilderRef P, StreamSet * ByteStream) {
    StreamSet * Source = ByteStream;
    if (codegen::EnableIllustrator) {
        P->captureByteData("Source", ByteStream);
    }
    auto u8 = mExternalTable.getStreamIndex(cc::UTF8.getCode());
    if (hasComponent(mExternalComponents, Component::S2P)) {
        StreamSet * BasisBits = P->CreateStreamSet(ENCODING_BITS, 1);
        Selected_S2P(P, ByteStream, BasisBits);
        Source = BasisBits;
        mExternalTable.declareExternal(u8, "basis", new PreDefined(BasisBits));
    } else {
        mExternalTable.declareExternal(u8, "basis", new PreDefined(ByteStream));
    }
    return Source;
}

void GrepEngine::grepPrologue(ProgBuilderRef P, StreamSet * SourceStream) {
    mLineBreakStream = nullptr;
    mU8index = nullptr;

    Scalar * const callbackObject = P->getInputScalar("callbackObject");
    if (mBinaryFilesMode == argv::Text) {
        mNullMode = NullCharMode::Data;
    } else if (mBinaryFilesMode == argv::WithoutMatch) {
        mNullMode = NullCharMode::Abort;
    } else {
        mNullMode = NullCharMode::Break;
    }
    mLineBreakStream = P->CreateStreamSet(1, 1);
    if (codegen::EnableIllustrator && hasComponent(mExternalComponents, Component::S2P)) {
        P->captureBixNum("basis", SourceStream);
    }
    if (mGrepRecordBreak == GrepRecordBreakKind::Unicode) {
        mU8index = P->CreateStreamSet(1, 1);
        UnicodeLinesLogic(P, SourceStream, mLineBreakStream, mU8index, UnterminatedLineAtEOF::Add1, mNullMode, callbackObject);
    }
    else {
        if (mGrepRecordBreak == GrepRecordBreakKind::LF) {
            Kernel * k = P->CreateKernelCall<UnixLinesKernelBuilder>(SourceStream, mLineBreakStream, UnterminatedLineAtEOF::Add1, mNullMode, callbackObject);
            if (mNullMode == NullCharMode::Abort) {
                k->link("signal_dispatcher", kernel::signal_dispatcher);
            }
        } else { // if (mGrepRecordBreak == GrepRecordBreakKind::Null) {
            P->CreateKernelCall<NullDelimiterKernel>(SourceStream, mLineBreakStream, UnterminatedLineAtEOF::Add1);
        }
        if (hasComponent(mExternalComponents, Component::UTF8index)) {
            mU8index = P->CreateStreamSet(1, 1);
            P->CreateKernelCall<UTF8_index>(SourceStream, mU8index, mLineBreakStream);
        }
    }
    auto u8 = mExternalTable.getStreamIndex(cc::UTF8.getCode());
    if (mU8index) {
        if (codegen::EnableIllustrator) {
            P->captureBitstream("mU8index", mU8index);
        }
        auto u8 = mExternalTable.getStreamIndex(cc::UTF8.getCode());
        mExternalTable.declareExternal(u8, "u8index", new PreDefined(mU8index));
    }
    if (codegen::EnableIllustrator) {
        P->captureBitstream("mLineBreakStream", mLineBreakStream);
    }
    auto u8_LB = new PreDefined(mLineBreakStream);//, std::make_pair(0, 0), 1);
    mExternalTable.declareExternal(u8, "$", u8_LB);
    if (UnicodeIndexing) {
        auto Unicode = mExternalTable.getStreamIndex(cc::Unicode.getCode());
        mExternalTable.declareExternal(Unicode, "$", new FilterByMaskExternal(u8, {"u8index", "$"}, u8_LB));
    }
}

void GrepEngine::prepareExternalStreams(ProgBuilderRef P, StreamSet * SourceStream) {
    mExternalTable.resolveExternals(P);
}

void GrepEngine::addExternalStreams(ProgBuilderRef P, const cc::Alphabet * indexAlphabet, std::unique_ptr<GrepKernelOptions> & options, re::RE * regexp, StreamSet * indexMask) {
    auto indexing = mExternalTable.getStreamIndex(indexAlphabet->getCode());
    re::Alphabet_Set alphas;
    re::collectAlphabets(regexp, alphas);
    std::set<re::Name *> externals;
    re::gatherNames(regexp, externals);
    // We may end up with multiple instances of a Name, but we should
    // only add the external once.
    std::set<std::string> extNames;
    for (const auto & e : externals) {
        auto name = e->getFullName();
        if ((extNames.count(name) == 0) && mExternalTable.isDeclared(indexing, name)) {
            extNames.insert(name);
            const auto & ext = mExternalTable.lookup(indexing, name);
            StreamSet * extStream = mExternalTable.getStreamSet(P, indexing, name);
            const auto offset = ext->getOffset();
            std::pair<int, int> lengthRange = ext->getLengthRange();
            options->addExternal(name, extStream, offset, lengthRange);
        } else {
            // We have a name that has not been set up as an external.
            // Its definition will need to be processed.
            re::RE * defn = e->getDefinition();
            if (defn) re::collectAlphabets(defn, alphas);
        }
    }
    for (auto & a : alphas) {
        if (const MultiplexedAlphabet * mpx = dyn_cast<MultiplexedAlphabet>(a)) {
            std::string basisName = a->getName() + "_basis";
            StreamSet * alphabetBasis = mExternalTable.getStreamSet(P, indexing, basisName);
            if (codegen::EnableIllustrator) {
                P->captureBixNum(basisName, alphabetBasis);
            }
            options->addAlphabet(mpx, alphabetBasis);
        } else {
            StreamSet * alphabetBasis = mExternalTable.getStreamSet(P, indexing, "basis");
            options->addAlphabet(a, alphabetBasis);
        }
    }
}

StreamSet * GrepEngine::getMatchSpan(ProgBuilderRef P, re::RE * r, StreamSet * MatchResults) {
    auto indexing = mExternalTable.getStreamIndex(mIndexAlphabet->getCode());
    if (mSpanNames.empty() == false) {
        std::vector<StreamSet *> allSpans;
        for (unsigned i = 0; i < mSpanNames.size(); i++) {
            allSpans.push_back(mExternalTable.getStreamSet(P, indexing, mSpanNames[i]));
        }
        if (allSpans.size() == 1) return allSpans[0];
        StreamSet * mergedSpans = P->CreateStreamSet(1, 1);
        P->CreateKernelCall<StreamsMerge>(allSpans, mergedSpans);
        return mergedSpans;
    }
    if (re::Alt * alt = dyn_cast<re::Alt>(r)) {
        std::vector<StreamSet *> allSpans;
        int i = 0;
        if (alt->empty()) return MatchResults;
        for (auto & e : *alt) {
            auto a = getMatchSpan(P, e, MatchResults);
            std::string ct = std::to_string(i);
            if (codegen::EnableIllustrator) {
                P->captureBitstream(ct, a);
            }
            allSpans.push_back(a);
            i++;
        }
        StreamSet * mergedSpans = P->CreateStreamSet(1, 1);
        P->CreateKernelCall<StreamsMerge>(allSpans, mergedSpans);
        return mergedSpans;
    } else {
        int spanLgth = re::getLengthRange(r, mIndexAlphabet).first;
        StreamSet * spans = P->CreateStreamSet(1, 1);
        P->CreateKernelFamilyCall<FixedMatchSpansKernel>(spanLgth, grepOffset(r), MatchResults, spans);
        return spans;
    }
}

unsigned GrepEngine::RunGrep(ProgBuilderRef P, const cc::Alphabet * indexAlphabet, re::RE * re, StreamSet * Results) {
    auto options = std::make_unique<GrepKernelOptions>(indexAlphabet);
    StreamSet * indexStream = nullptr;
    if (indexAlphabet == &cc::UTF8) {
        if (mLengthAlphabet == &cc::Unicode) {
            indexStream = mU8index;
            options->setIndexing(indexStream);
        }
    }
    options->setRE(re);
    auto indexing = mExternalTable.getStreamIndex(indexAlphabet->getCode());
    options->setBarrier(mExternalTable.getStreamSet(P, indexing, "$"));
    addExternalStreams(P, indexAlphabet, options, re, indexStream);
    options->setResults(Results);
    Kernel * k = P->CreateKernelFamilyCall<ICGrepKernel>(std::move(options));
    if (codegen::EnableIllustrator) {
        P->captureBitstream("RunGrep", Results);
    }
    return reinterpret_cast<ICGrepKernel *>(k)->getOffset();
}

StreamSet * GrepEngine::initialMatches(ProgBuilderRef P, StreamSet * InputStream) {
    mExternalTable.resetExternals();
    StreamSet * SourceStream = getBasis(P, InputStream);
    grepPrologue(P, SourceStream);
    prepareExternalStreams(P, SourceStream);
    StreamSet * Matches = P->CreateStreamSet();
    RunGrep(P, mIndexAlphabet, mRE, Matches);
    if (mIndexAlphabet == &cc::Unicode) {
        StreamSet * u8index1 = P->CreateStreamSet(1, 1);
        P->CreateKernelCall<AddSentinel>(mU8index, u8index1);
        StreamSet * Results = P->CreateStreamSet(1, 1);
        SpreadByMask(P, u8index1, Matches, Results);
        Matches = Results;
    }
    if (codegen::EnableIllustrator) {
        P->captureBitstream("ICgrep kernel matches", Matches);
    }
    return Matches;
}

StreamSet * GrepEngine::matchedLines(ProgBuilderRef P, StreamSet * initialMatches) {
    StreamSet * MatchedLineEnds = nullptr;
    if (hasComponent(mExternalComponents, Component::MoveMatchesToEOL)) {
        StreamSet * const MovedMatches = P->CreateStreamSet();
        P->CreateKernelCall<MatchedLinesKernel>(initialMatches, mLineBreakStream, MovedMatches);
        if (codegen::EnableIllustrator) {
            P->captureBitstream("MovedMatches", MovedMatches);
        }
        MatchedLineEnds = MovedMatches;
    } else {
        MatchedLineEnds = initialMatches;
    }
    if (mInvertMatches) {
        StreamSet * const InvertedMatches = P->CreateStreamSet();
        P->CreateKernelCall<InvertMatchesKernel>(MatchedLineEnds, mLineBreakStream, InvertedMatches);
        if (codegen::EnableIllustrator) {
            P->captureBitstream("InvertedMatches", InvertedMatches);
        }
        MatchedLineEnds = InvertedMatches;
    }
    if (mMaxCount > 0) {
        StreamSet * MaxCountLines = nullptr;
        Scalar * const maxCount = P->getInputScalar("maxCount");
        const UntilNMode m = MaxLimitTerminationMode;
        if (m == UntilNMode::ReportAcceptedLengthAtAndBeforeN) {
            MaxCountLines = P->CreateTruncatedStreamSet(MatchedLineEnds);
        } else {
            MaxCountLines = P->CreateStreamSet();
        }
        P->CreateKernelCall<UntilNkernel>(maxCount, MatchedLineEnds, MaxCountLines, m);
        if (codegen::EnableIllustrator) {
            P->captureBitstream("MaxCountLines", MaxCountLines);
        }
        MatchedLineEnds = MaxCountLines;
        StreamSet * TruncatedLines =
            streamutils::Merge(P, {{MaxCountLines, {0}}, {mLineBreakStream, {0}}});
        if (codegen::EnableIllustrator) {
            P->captureBitstream("TruncatedLines", TruncatedLines);
        }
        mLineBreakStream = TruncatedLines;
    }
    if (codegen::EnableIllustrator) {
        P->captureBitstream("MatchedLineEnds", MatchedLineEnds);
    }
    return MatchedLineEnds;
}

StreamSet * GrepEngine::grepPipeline(ProgBuilderRef P, StreamSet * InputStream) {
    StreamSet * Matches = initialMatches(P, InputStream);
    return matchedLines(P, Matches);
}



// The QuietMode, MatchOnly and CountOnly engines share a common code generation main function,
// which returns a count of the matches found (possibly subject to a MaxCount).
//

void GrepEngine::grepCodeGen() {
    auto & b = mGrepDriver.getBuilder();

    auto P = mGrepDriver.makePipeline(
                // inputs
                {Binding{b.getSizeTy(), "useMMap"},
                Binding{b.getInt32Ty(), "fileDescriptor"},
                Binding{b.getIntAddrTy(), "callbackObject"},
                Binding{b.getSizeTy(), "maxCount"}}
                ,// output
                {Binding{b.getInt64Ty(), "countResult"}});

    Scalar * const useMMap = P->getInputScalar("useMMap");
    Scalar * const fileDescriptor = P->getInputScalar("fileDescriptor");
    StreamSet * const ByteStream = P->CreateStreamSet(1, ENCODING_BITS);
    P->CreateKernelCall<FDSourceKernel>(useMMap, fileDescriptor, ByteStream);
    StreamSet * const Matches = grepPipeline(P, ByteStream);
    P->CreateKernelCall<PopcountKernel>(Matches, P->getOutputScalar("countResult"));

    mMainMethod = P->compile();
}

//
//  Default Report Match:  lines are emitted with whatever line terminators are found in the
//  input.  However, if the final line is not terminated, a new line is appended.
//
constexpr size_t batch_alignment = 64;

void EmitMatch::setFileLabel(std::string fileLabel) {
    if (mShowFileNames) {
        mLinePrefix = fileLabel + (mInitialTab ? "\t:" : ":");
    } else mLinePrefix = "";
}

void EmitMatch::setStringStream(std::ostringstream * s) {
    mResultStr = s;
}

unsigned EmitMatch::getFileCount() {
    mCurrentFile = 0;
    if (mFileNames.size() == 0) return 1;
    return mFileNames.size();
}

size_t EmitMatch::getFileStartPos(unsigned fileNo) {
    if (mFileStartPositions.size() == 0) return 0;
    assert(fileNo < mFileStartPositions.size());
    //llvm::errs() << "getFileStartPos(" << fileNo << ") = ";
    //llvm::errs().write_hex(mFileStartPositions[fileNo]);
    //llvm::errs() << "  file = " << mFileNames[fileNo] << "\n";
    return mFileStartPositions[fileNo];
}

void EmitMatch::setBatchLineNumber(unsigned fileNo, size_t batchLine) {
    //llvm::errs() << "setBatchLineNumber(" << fileNo << ", " << batchLine << ")  file = " << mFileNames[fileNo] << "\n";
    mFileStartLineNumbers[fileNo+1] = batchLine;
    if (!mTerminated) *mResultStr << "\n";
    mTerminated = true;
}

void EmitMatch::accumulate_match (const size_t lineNum, char * line_start, char * line_end) {
    //llvm::errs() << "lineNum = " << lineNum << "\n";
    while ((mCurrentFile + 1 < mFileStartPositions.size()) && (mFileStartLineNumbers[mCurrentFile + 1] <= lineNum)) {
        mCurrentFile++;
        //llvm::errs() << "mCurrentFile = " << mCurrentFile << "\n";
        setFileLabel(mFileNames[mCurrentFile]);
    }
    size_t relLineNum = mCurrentFile > 0 ? lineNum - mFileStartLineNumbers[mCurrentFile] : lineNum;
    if (mContextGroups && (lineNum > mLineNum + 1) && (relLineNum > 0)) {
        *mResultStr << "--\n";
    }
    *mResultStr << mLinePrefix;
    if (mShowLineNumbers) {
        // Internally line numbers are counted from 0.  For display, adjust
        // the line number so that lines are numbered from 1.
        if (mInitialTab) {
            *mResultStr << relLineNum+1 << "\t:";
        }
        else {
            *mResultStr << relLineNum+1 << ":";
        }
    }

    const auto bytes = line_end - line_start + 1;
    mResultStr->write(line_start, bytes);
    mLineCount++;
    mLineNum = lineNum;
    unsigned last_byte = *line_end;
    mTerminated = (last_byte >= 0x0A) && (last_byte <= 0x0D);
    if (LLVM_UNLIKELY(!mTerminated)) {
        if (last_byte == 0x85) {  //  Possible NEL terminator.
            mTerminated = (bytes >= 2) && (static_cast<unsigned>(line_end[-1]) == 0xC2);
        }
        else {
            // Possible LS or PS terminators.
            mTerminated = (bytes >= 3) && (static_cast<unsigned>(line_end[-2]) == 0xE2)
                                       && (static_cast<unsigned>(line_end[-1]) == 0x80)
                                       && ((last_byte == 0xA8) || (last_byte == 0xA9));
        }
    }
}

void EmitMatch::finalize_match(char * buffer_end) {
    if (!mTerminated) *mResultStr << "\n";
}

class GrepColourizationPipeline : public PipelineKernel {
public:
    GrepColourizationPipeline(KernelBuilder & b,
                              StreamSet * SourceCoords,
                              StreamSet * MatchSpans,
                              StreamSet * Basis,
                              Scalar * const callbackObject)
        : PipelineKernel(b
                         // signature
                         , [&]() -> std::string {
                             return pablo::annotateKernelNameWithPabloDebugFlags("GrepColourization");
                         }()
                         // contains kernel family calls
                         , false
                         // kernel list
                         , {}
                         // called functions
                         , {}
                         // stream inputs
                         , {Bind("SourceCoords", SourceCoords, GreedyRate(1), Deferred()),
                            Bind("MatchSpans", MatchSpans, GreedyRate(), Deferred()),
                            Bind("Basis", Basis, GreedyRate(), Deferred())}
                         // stream outputs
                         , {}
                         // input scalars
                         , {Binding{b.getIntAddrTy(), "callbackObject", callbackObject}}
                         // output scalars
                         , {}
                         // internally generated streamsets
                         , {}
                         // length assertions
                         , {}) {
        addAttribute(InternallySynchronized());
        addAttribute(MustExplicitlyTerminate());
        addAttribute(SideEffecting());
        // TODO: study the I/O settings to see what the best balance is for memory vs. throughput.

        // TODO: I'm not sure how safe the greedyrate is here. When compiling the nested kernel,
        // the pipeline compiler doesn't really understand how to treat the greedy input rate
        // as a "production" rate. The simulator inside needs more information to understand it
        // as a dataflow rate but current modelling system isn't very good for that.

    }

protected:

    void instantiateInternalKernels(const std::unique_ptr<PipelineBuilder> & E) final {
        const std::string ESC = "\x1B";
        const std::vector<std::string> colorEscapes = {ESC + "[01;31m" + ESC + "[K", ESC + "[m"};
        std::vector<uint64_t> colorEscapeBytes;
        const  unsigned insertLengthBits = 4;
        std::vector<unsigned> insertAmts;
        for (auto & s : colorEscapes) {
            insertAmts.push_back(s.size());
            for (auto & ch : s) {
                colorEscapeBytes.push_back(static_cast<uint64_t>(ch));
            }
        }

        StreamSet * const MatchSpans = getInputStreamSet(1);
        StreamSet * const InsertMarks = E->CreateStreamSet(2, 1);
        E->CreateKernelCall<SpansToMarksKernel>(MatchSpans, InsertMarks);

        StreamSet * const InsertBixNum = E->CreateStreamSet(insertLengthBits, 1);
        E->CreateKernelCall<ZeroInsertBixNum>(insertAmts, InsertMarks, InsertBixNum);
        StreamSet * const SpreadMask = InsertionSpreadMask(E, InsertBixNum, InsertPosition::Before);

        StreamSet * const Basis = getInputStreamSet(2);
        StreamSet * ColorEscapeBasis = E->CreateRepeatingBixNum(8, colorEscapeBytes);
        StreamSet * ColorizedBasis = E->CreateStreamSet(8);
        MergeByMask(E, SpreadMask, Basis, ColorEscapeBasis, ColorizedBasis);

        StreamSet * const ColorizedBytes  = E->CreateStreamSet(1, 8);
        E->CreateKernelCall<P2SKernel>(ColorizedBasis, ColorizedBytes);

        StreamSet * ColorizedBreaks = E->CreateStreamSet(1);
        E->CreateKernelCall<UnixLinesKernelBuilder>(ColorizedBasis, ColorizedBreaks, UnterminatedLineAtEOF::Add1);

        StreamSet * const ColorizedCoords = E->CreateStreamSet(3, sizeof(size_t) * 8);
        E->CreateKernelCall<MatchCoordinatesKernel>(ColorizedBreaks, ColorizedBreaks, ColorizedCoords, 1);

        // TODO: source coords >= colorized coords until the final stride?
        // E->AssertEqualLength(SourceCoords, ColorizedCoords);

        StreamSet * const SourceCoords = getInputStreamSet(0);
        Scalar * const callbackObject = getInputScalarAt(0);
        Kernel * const matchK = E->CreateKernelCall<ColorizedReporter>(ColorizedBytes, SourceCoords, ColorizedCoords, callbackObject);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

};

void GrepEngine::applyColorization(const std::unique_ptr<ProgramBuilder> & E,
                                   StreamSet * SourceCoords,
                                   StreamSet * MatchSpans,
                                   StreamSet * Basis) {

    Scalar * const callbackObject = E->getInputScalar("callbackObject");

    if (UseNestedColourizationPipeline) {
        E->CreateNestedPipelineCall<GrepColourizationPipeline>(SourceCoords, MatchSpans, Basis, callbackObject);
    } else {
        std::string ESC = "\x1B";
        std::vector<std::string> colorEscapes = {ESC + "[01;31m" + ESC + "[K", ESC + "[m"};
        unsigned insertLengthBits = 4;
        std::vector<unsigned> insertAmts;
        for (auto & s : colorEscapes) {insertAmts.push_back(s.size());}

        StreamSet * const SpanMarks = E->CreateStreamSet(2, 1);
        E->CreateKernelCall<SpansToMarksKernel>(MatchSpans, SpanMarks);
        if (codegen::EnableIllustrator) {
            E->captureBixNum("SpanMarks", SpanMarks);
        }

        StreamSet * const InsertBixNum = E->CreateStreamSet(insertLengthBits, 1);
        E->CreateKernelCall<ZeroInsertBixNum>(insertAmts, SpanMarks, InsertBixNum);
        StreamSet * const SpreadMask = InsertionSpreadMask(E, InsertBixNum, InsertPosition::Before);
        if (codegen::EnableIllustrator) {
            E->captureBitstream("SpreadMask", SpreadMask);
        }

        // For each run of 0s marking insert positions, create a parallel
        // bixnum sequentially numbering the string insert positions.
        StreamSet * const InsertIndex = E->CreateStreamSet(insertLengthBits);
        E->CreateKernelCall<RunIndex>(SpreadMask, InsertIndex, nullptr, RunIndex::Kind::RunOf0);
        // Basis bit streams expanded with 0 bits for each string to be inserted.
        if (codegen::EnableIllustrator) {
            E->captureBixNum("InsertIndex", InsertIndex);
        }

        StreamSet * ExpandedBasis = E->CreateStreamSet(8);
        SpreadByMask(E, SpreadMask, Basis, ExpandedBasis);

        // Map the match start/end marks to their positions in the expanded basis.
        StreamSet * ExpandedMarks = E->CreateStreamSet(2);
        SpreadByMask(E, SpreadMask, SpanMarks, ExpandedMarks);
        if (codegen::EnableIllustrator) {
            E->captureBixNum("ExpandedMarks", ExpandedMarks);
        }

        StreamSet * ColorizedBasis = E->CreateStreamSet(8);
        E->CreateKernelCall<StringReplaceKernel>(colorEscapes, ExpandedBasis, SpreadMask, ExpandedMarks, InsertIndex, ColorizedBasis, -1);
        if (codegen::EnableIllustrator) {
            E->captureBixNum("ColorizedBasis", ColorizedBasis);
        }
        StreamSet * const ColorizedBytes  = E->CreateStreamSet(1, 8);
        E->CreateKernelCall<P2SKernel>(ColorizedBasis, ColorizedBytes);

        StreamSet * ColorizedBreaks = E->CreateStreamSet(1);
        E->CreateKernelCall<UnixLinesKernelBuilder>(ColorizedBasis, ColorizedBreaks, UnterminatedLineAtEOF::Add1);

        StreamSet * const ColorizedCoords = E->CreateStreamSet(3, sizeof(size_t) * 8);
        E->CreateKernelCall<MatchCoordinatesKernel>(ColorizedBreaks, ColorizedBreaks, ColorizedCoords, 1);
        if (codegen::EnableIllustrator) {
            E->captureBitstream("ColorizedBreaks", ColorizedBreaks);
        }

        // TODO: source coords >= colorized coords until the final stride?
        // E->AssertEqualLength(SourceCoords, ColorizedCoords);

        Kernel * const matchK = E->CreateKernelCall<ColorizedReporter>(ColorizedBytes, SourceCoords, ColorizedCoords, callbackObject);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

}

void EmitMatchesEngine::grepPipeline(ProgBuilderRef E, StreamSet * ByteStream) {
    StreamSet * Matches = initialMatches(E, ByteStream);
    StreamSet * MatchedLineEnds = matchedLines(E, Matches);

    bool hasContext = (mAfterContext != 0) || (mBeforeContext != 0);
    bool needsColoring = mColoring && !mInvertMatches;
    StreamSet * MatchesByLine = nullptr;
    if (needsColoring | hasContext) {
        MatchesByLine = E->CreateStreamSet(1, 1);
        FilterByMask(E, mLineBreakStream, MatchedLineEnds, MatchesByLine);
        if (codegen::EnableIllustrator) {
            E->captureBitstream("MatchesByLine", MatchesByLine);
        }
    }
    if (hasContext) {
        StreamSet * ContextByLine = E->CreateStreamSet(1, 1);
        E->CreateKernelCall<ContextSpan>(MatchesByLine, ContextByLine, mBeforeContext, mAfterContext);
        StreamSet * SelectedLines = E->CreateStreamSet(1, 1);
        SpreadByMask(E, mLineBreakStream, ContextByLine, SelectedLines);
        MatchedLineEnds = SelectedLines;
        MatchesByLine = ContextByLine;
    }

    if (needsColoring) {
        StreamSet * SourceCoords = E->CreateStreamSet(1, sizeof(size_t) * 8);
        Scalar * const callbackObject = E->getInputScalar("callbackObject");
        Kernel * const batchK = E->CreateKernelCall<BatchCoordinatesKernel>(MatchedLineEnds, mLineBreakStream, SourceCoords, callbackObject);
        batchK->link("get_file_count_wrapper", get_file_count_wrapper);
        batchK->link("get_file_start_pos_wrapper", get_file_start_pos_wrapper);
        batchK->link("set_batch_line_number_wrapper", set_batch_line_number_wrapper);

        StreamSet * MatchedLineStarts = E->CreateStreamSet(1, 1);
        StreamSet * lineStarts = E->CreateStreamSet(1, 1);
        E->CreateKernelCall<LineStartsKernel>(mLineBreakStream, lineStarts);
        SpreadByMask(E, lineStarts, MatchesByLine, MatchedLineStarts);
        if (codegen::EnableIllustrator) {
            E->captureBitstream("MatchedLineStarts", MatchedLineStarts);
        }
        StreamSet * Filtered = E->CreateStreamSet(1, 8);
        E->CreateKernelCall<MatchFilterKernel>(MatchedLineStarts, mLineBreakStream, ByteStream, Filtered);
        if (codegen::EnableIllustrator) {
            E->captureBixNum("Filtered", Filtered);
        }
        StreamSet * MatchedLineSpans = E->CreateStreamSet(1, 1);
        E->CreateKernelCall<LineSpansKernel>(MatchedLineStarts, MatchedLineEnds, MatchedLineSpans);

        StreamSet * MatchSpans;
        MatchSpans = getMatchSpan(E, mRE, Matches);
        if (codegen::EnableIllustrator) {
            E->captureBitstream("Matches", Matches);
            E->captureBitstream("MatchSpans", MatchSpans);
        }
        if (UnicodeIndexing) {
            StreamSet * u8index1 = E->CreateStreamSet(1, 1);
            E->CreateKernelCall<AddSentinel>(mU8index, u8index1);
            StreamSet * ExpandedSpans = E->CreateStreamSet(1, 1);
            SpreadByMask(E, u8index1, MatchSpans, ExpandedSpans);
            if (codegen::EnableIllustrator) {
                E->captureBitstream("ExpandedSpans", ExpandedSpans);
            }
            StreamSet * FilledSpans = E->CreateStreamSet(1, 1);
            E->CreateKernelCall<U8Spans>(ExpandedSpans, u8index1, FilledSpans);
            if (codegen::EnableIllustrator) {
                E->captureBitstream("FilledSpans", FilledSpans);
            }
            MatchSpans = FilledSpans;
        }

        StreamSet * FilteredMatchSpans = E->CreateStreamSet(1, 1);
        FilterByMask(E, MatchedLineSpans, MatchSpans, FilteredMatchSpans);
        if (codegen::EnableIllustrator) {
            E->captureBitstream("FilteredMatchSpans", FilteredMatchSpans);
        }
        StreamSet * FilteredBasis = E->CreateStreamSet(8, 1);
        if (codegen::SplitTransposition) {
            Staged_S2P(E, Filtered, FilteredBasis);
        } else {
            E->CreateKernelCall<S2PKernel>(Filtered, FilteredBasis);
        }

        applyColorization(E, SourceCoords, FilteredMatchSpans, FilteredBasis);

    } else { // Non colorized output
        if (MatchCoordinateBlocks > 0) {
            StreamSet * MatchCoords = E->CreateStreamSet(3, sizeof(size_t) * 8);
            E->CreateKernelCall<MatchCoordinatesKernel>(MatchedLineEnds, mLineBreakStream, MatchCoords, MatchCoordinateBlocks);
            Scalar * const callbackObject = E->getInputScalar("callbackObject");
            Kernel * const matchK = E->CreateKernelCall<MatchReporter>(ByteStream, MatchCoords, callbackObject);
            matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
            matchK->link("finalize_match_wrapper", finalize_match_wrapper);
        } else {
            Scalar * const callbackObject = E->getInputScalar("callbackObject");
            Kernel * const scanBatchK = E->CreateKernelCall<ScanBatchKernel>(MatchedLineEnds, mLineBreakStream, ByteStream, callbackObject, ScanMatchBlocks);
            scanBatchK->link("get_file_count_wrapper", get_file_count_wrapper);
            scanBatchK->link("get_file_start_pos_wrapper", get_file_start_pos_wrapper);
            scanBatchK->link("set_batch_line_number_wrapper", set_batch_line_number_wrapper);
            scanBatchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
            scanBatchK->link("finalize_match_wrapper", finalize_match_wrapper);
        }
    }
}


void EmitMatchesEngine::grepCodeGen() {
    auto & b = mGrepDriver.getBuilder();

    auto E2 = mGrepDriver.makePipeline(
                    // inputs
                    {Binding{b.getInt8PtrTy(), "buffer"},
                    Binding{b.getSizeTy(), "length"},
                    Binding{b.getIntAddrTy(), "callbackObject"},
                    Binding{b.getSizeTy(), "maxCount"}}
                    ,// output
                    {Binding{b.getInt64Ty(), "countResult"}});

    Scalar * const buffer = E2->getInputScalar("buffer");
    Scalar * const length = E2->getInputScalar("length");
    StreamSet * const InternalBytes = E2->CreateStreamSet(1, 8);
    E2->CreateKernelCall<MemorySourceKernel>(buffer, length, InternalBytes);
    grepPipeline(E2, InternalBytes);
    E2->setOutputScalar("countResult", E2->CreateConstant(b.getInt64(0)));
    mBatchMethod = E2->compile();
}

bool canMMap(const std::string & fileName) {
    if (fileName == "-") return false;
    namespace fs = boost::filesystem;
    fs::path p(fileName);
    boost::system::error_code errc;
    fs::file_status s = fs::status(p, errc);
    return !errc && is_regular_file(s);
}


uint64_t GrepEngine::doGrep(const std::vector<std::string> & fileNames, std::ostringstream & strm) {
    typedef uint64_t (*GrepFunctionType)(bool useMMap, int32_t fileDescriptor, GrepCallBackObject *, size_t maxCount);
    auto f = reinterpret_cast<GrepFunctionType>(mMainMethod);
    uint64_t resultTotal = 0;

    for (auto fileName : fileNames) {
        GrepCallBackObject handler;
        bool useMMap = mPreferMMap && canMMap(fileName);
        int32_t fileDescriptor = openFile(fileName, strm);
        if (fileDescriptor == -1) return 0;
        uint64_t grepResult = f(useMMap, fileDescriptor, &handler, mMaxCount);
        close(fileDescriptor);
        if (handler.binaryFileSignalled()) {
            llvm::errs() << "Binary file " << fileName << "\n";
        }
        else {
            showResult(grepResult, fileName, strm);
            resultTotal += grepResult;
        }
    }
    return resultTotal;
}

std::string GrepEngine::linePrefix(std::string fileName) {
    if (!mShowFileNames) return "";
    if (fileName == "-") {
        return mStdinLabel + mFileSuffix;
    }
    else {
        return fileName + mFileSuffix;
    }
}

// Default: do not show anything
void GrepEngine::showResult(uint64_t grepResult, const std::string & fileName, std::ostringstream & strm) {

}

void CountOnlyEngine::showResult(uint64_t grepResult, const std::string & fileName, std::ostringstream & strm) {
    if (mShowFileNames) strm << linePrefix(fileName);
    strm << grepResult << "\n";
}

void MatchOnlyEngine::showResult(uint64_t grepResult, const std::string & fileName, std::ostringstream & strm) {
    if (grepResult == mRequiredCount) {
       strm << linePrefix(fileName);
    }
}

uint64_t EmitMatchesEngine::doGrep(const std::vector<std::string> & fileNames, std::ostringstream & strm) {
    typedef uint64_t (*GrepBatchFunctionType)(char * buffer, size_t length, EmitMatch *, size_t maxCount);
    auto f = reinterpret_cast<GrepBatchFunctionType>(mBatchMethod);
    EmitMatch accum(mShowFileNames, mShowLineNumbers, ((mBeforeContext > 0) || (mAfterContext > 0)), mInitialTab);
    accum.setStringStream(&strm);
    std::vector<int32_t> fileDescriptor(fileNames.size());
    std::vector<size_t> fileSize(fileNames.size(), 0);
    size_t cumulativeSize = 0;
    unsigned filesOpened = 0;
    unsigned lastOpened = 0;
    for (unsigned i = 0; i < fileNames.size(); i++) {
        fileDescriptor[i] = openFile(fileNames[i], strm);
        if (fileDescriptor[i] == -1) continue;  // File error; skip.
        struct stat st;
        if (fstat(fileDescriptor[i], &st) != 0) continue;
        fileSize[i] = st.st_size;
        cumulativeSize += st.st_size;
        filesOpened++;
        lastOpened = i;
    }
    if (filesOpened == 0) return 0;
    accum.mFileNames.reserve(filesOpened);
    accum.mFileStartPositions.reserve(filesOpened);
    AlignedAllocator<char, batch_alignment> alloc;
    size_t current_start_position = 0;
    bool singleFileMMapMode = (filesOpened == 1) && canMMap(fileNames[lastOpened]);
    if (singleFileMMapMode) {
        auto mmap_ptr = mmap(NULL, fileSize[lastOpened], PROT_READ, MAP_PRIVATE, fileDescriptor[lastOpened], 0);
        accum.mBatchBuffer = reinterpret_cast<char *>(mmap_ptr);
        accum.mFileNames.push_back(fileNames[lastOpened]);
        accum.mFileStartPositions.push_back(static_cast<size_t>(0));
    } else {
        cumulativeSize += filesOpened;  // Add an extra byte per file for possible '\n'.
        size_t aligned_size = (cumulativeSize + batch_alignment - 1) & -batch_alignment;

        accum.mBatchBuffer = alloc.allocate(aligned_size, 0);
        if (accum.mBatchBuffer == nullptr) {
            llvm::report_fatal_error(llvm::StringRef("Unable to allocate batch buffer of size: ") + std::to_string(aligned_size));
        }
        char * current_base = accum.mBatchBuffer;

        for (unsigned i = 0; i < fileNames.size(); i++) {
            if (fileDescriptor[i] == -1) continue;  // Error opening file; skip.
            ssize_t bytes_read = read(fileDescriptor[i], current_base, fileSize[i]);
            close(fileDescriptor[i]);
            if (bytes_read <= 0) continue; // No data or error reading the file; skip.
            if ((mBinaryFilesMode == argv::WithoutMatch) || (mBinaryFilesMode == argv::Binary)) {
                auto null_byte_ptr = memchr(current_base, char (0), bytes_read);
                if (null_byte_ptr != nullptr) { // Binary file;
                    // Silently skip in the WithoutMatch mode
                    if (mBinaryFilesMode == argv::WithoutMatch) continue;
                    strm << "Binary file: " << fileNames[i] << " skipped.\n";
                }
            }
            accum.mFileNames.push_back(fileNames[i]);
            accum.mFileStartPositions.push_back(current_start_position);
            current_base += bytes_read;
            current_start_position += bytes_read;
            if (*(current_base - 1) != '\n') {
                *current_base = '\n';
                current_base++;
                current_start_position++;
            }
        }
        cumulativeSize = current_start_position;
    }
    if (accum.mFileNames.size() > 0) {
        accum.setFileLabel(accum.mFileNames[0]);
        accum.mFileStartLineNumbers.resize(accum.mFileNames.size());
        // Initialize to the maximum integer value so that tests
        // will not rule that we are past a given file until the
        // actual limit is computed.
        for (unsigned i = 0; i < accum.mFileStartLineNumbers.size(); i++) {
            accum.mFileStartLineNumbers[i] = ~static_cast<size_t>(0);
        }
        f(accum.mBatchBuffer, cumulativeSize, &accum, mMaxCount);
    }
    if (singleFileMMapMode) {
        munmap(reinterpret_cast<void *>(accum.mBatchBuffer), fileSize[lastOpened]);
    } else {
        alloc.deallocate(accum.mBatchBuffer, 0);
    }
    if (accum.mLineCount > 0) grepMatchFound = true;
    return accum.mLineCount;
}

// Open a file and return its file desciptor.
int32_t GrepEngine::openFile(const std::string & fileName, std::ostringstream & msgstrm) {
    if (fileName == "-") {
        return STDIN_FILENO;
    }
    else {
        struct stat sb;
        int flags = O_RDONLY;
        #ifdef __linux__
        if (NoOSFileCaching) {
            flags |= O_DIRECT;
        }
        #endif
        int32_t fileDescriptor = open(fileName.c_str(), flags);
        if (LLVM_UNLIKELY(fileDescriptor == -1)) {
            if (!mSuppressFileMessages) {
                if (errno == EACCES) {
                    msgstrm << "icgrep: " << fileName << ": Permission denied.\n";
                }
                else if (errno == ENOENT) {
                    msgstrm << "icgrep: " << fileName << ": No such file.\n";
                }
                else {
                    msgstrm << "icgrep: " << fileName << ": Failed.\n";
                }
            }
            return fileDescriptor;
        }
        if (stat(fileName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            if (!mSuppressFileMessages) {
                msgstrm << "icgrep: " << fileName << ": Is a directory.\n";
            }
            close(fileDescriptor);
            return -1;
        }
        #ifdef __APPLE__
        if (NoOSFileCaching) {
            fcntl(fileDescriptor, F_NOCACHE, 1);
            fcntl(fileDescriptor, F_RDAHEAD, 0);
        }
        #endif
        if (TraceFiles) {
            llvm::errs() << "Opened " << fileName << ".\n";
        }
        return fileDescriptor;
    }
}

// The process of searching a group of files may use a sequential or a task
// parallel approach.

void * DoGrepThreadFunction(void *args) {
    assert (args);
    return reinterpret_cast<GrepEngine *>(args)->DoGrepThreadMethod();
}

bool GrepEngine::searchAllFiles() {

    std::vector<pthread_t> threads(codegen::TaskThreads);

    for(unsigned long i = 1; i < codegen::TaskThreads; ++i) {
        const int rc = pthread_create(&threads[i], nullptr, DoGrepThreadFunction, (void *)this);
        if (rc) {
            llvm::report_fatal_error(llvm::StringRef("Failed to create thread: code ") + std::to_string(rc));
        }
    }
    // Main thread also does the work;
    DoGrepThreadMethod();
    for(unsigned i = 1; i < codegen::TaskThreads; ++i) {
        void * status = nullptr;
        const int rc = pthread_join(threads[i], &status);
        if (rc) {
            llvm::report_fatal_error(llvm::StringRef("Failed to join thread: code ") + std::to_string(rc));
        }
    }
    return grepMatchFound;
}

// DoGrep thread function.
void * GrepEngine::DoGrepThreadMethod() {

    unsigned fileIdx = mNextFileToGrep++;
    while (fileIdx < mFileGroups.size()) {

        const auto grepResult = doGrep(mFileGroups[fileIdx], mResultStrs[fileIdx]);
        mFileStatus[fileIdx] = FileStatus::GrepComplete;
        if (grepResult > 0) {
            grepMatchFound = true;
        }
        if ((mEngineKind == EngineKind::QuietMode) && grepMatchFound) {
            if (pthread_self() != mEngineThread) {
                pthread_exit(nullptr);
            }
            return nullptr;
        }
        fileIdx = mNextFileToGrep++;
        if (pthread_self() == mEngineThread) {
            while ((mNextFileToPrint < mFileGroups.size()) && (mFileStatus[mNextFileToPrint] == FileStatus::GrepComplete)) {
                const auto output = mResultStrs[mNextFileToPrint].str();
                if (!output.empty()) {
                    llvm::outs() << output;
                }
                mFileStatus[mNextFileToPrint] = FileStatus::PrintComplete;
                mNextFileToPrint++;
            }
        }
    }
    if (pthread_self() != mEngineThread) {
        pthread_exit(nullptr);
    }
    while (mNextFileToPrint < mFileGroups.size()) {
        const bool readyToPrint = (mFileStatus[mNextFileToPrint] == FileStatus::GrepComplete);
        if (readyToPrint) {
            const auto output = mResultStrs[mNextFileToPrint].str();
            if (!output.empty()) {
                llvm::outs() << output;
            }
            mFileStatus[mNextFileToPrint] = FileStatus::PrintComplete;
            mNextFileToPrint++;
        } else {
            sched_yield();
        }
    }
    if (mGrepStdIn) {
        std::ostringstream s;
        const auto grepResult = doGrep({"-"}, s);
        llvm::outs() << s.str();
        if (grepResult) grepMatchFound = true;
    }
    return nullptr;
}

InternalSearchEngine::InternalSearchEngine(BaseDriver &driver) :
mGrepRecordBreak(GrepRecordBreakKind::LF),
mCaseInsensitive(false),
mGrepDriver(driver),
mMainMethod(nullptr) {

}

void InternalSearchEngine::grepCodeGen(re::RE * matchingRE) {
    auto & b = mGrepDriver.getBuilder();

    re::CC * breakCC = nullptr;
    if (mGrepRecordBreak == GrepRecordBreakKind::Null) {
        breakCC = re::makeCC(0x0, &cc::Unicode);
    } else {// if (mGrepRecordBreak == GrepRecordBreakKind::LF)
        breakCC = re::makeCC(0x0A, &cc::Unicode);
    }

    matchingRE = re::exclude_CC(matchingRE, breakCC);
    matchingRE = resolveAnchors(matchingRE, breakCC);
    matchingRE = resolveCaseInsensitiveMode(matchingRE, mCaseInsensitive);
    matchingRE = regular_expression_passes(matchingRE);
    matchingRE = toUTF8(matchingRE);

    auto E = mGrepDriver.makePipeline({Binding{b.getInt8PtrTy(), "buffer"},
                                       Binding{b.getSizeTy(), "length"},
                                       Binding{b.getIntAddrTy(), "accumulator"}});

    Scalar * const buffer = E->getInputScalar(0);
    Scalar * const length = E->getInputScalar(1);
    Scalar * const callbackObject = E->getInputScalar(2);
    StreamSet * ByteStream = E->CreateStreamSet(1, 8);
    E->CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);

    StreamSet * RecordBreakStream = E->CreateStreamSet();
    StreamSet * BasisBits = E->CreateStreamSet(8);
    E->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    E->CreateKernelCall<CharacterClassKernelBuilder>(std::vector<re::CC *>{breakCC}, BasisBits, RecordBreakStream);

    StreamSet * u8index = E->CreateStreamSet();
    E->CreateKernelCall<UTF8_index>(BasisBits, u8index);

    StreamSet * MatchResults = E->CreateStreamSet();
    auto options = std::make_unique<GrepKernelOptions>(&cc::UTF8);
    options->setBarrier(RecordBreakStream);
    options->setRE(matchingRE);
    options->addAlphabet(&cc::UTF8, BasisBits);
    options->setResults(MatchResults);
    options->addExternal("UTF8_index", u8index);
    E->CreateKernelFamilyCall<ICGrepKernel>(std::move(options));
    StreamSet * MatchingRecords = E->CreateStreamSet();
    E->CreateKernelCall<MatchedLinesKernel>(MatchResults, RecordBreakStream, MatchingRecords);

    if (MatchCoordinateBlocks > 0) {
        StreamSet * MatchCoords = E->CreateStreamSet(3, sizeof(size_t) * 8);
        E->CreateKernelCall<MatchCoordinatesKernel>(MatchingRecords, RecordBreakStream, MatchCoords, MatchCoordinateBlocks);
        Kernel * const matchK = E->CreateKernelCall<MatchReporter>(ByteStream, MatchCoords, callbackObject);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    } else {
        Kernel * const scanMatchK = E->CreateKernelCall<ScanMatchKernel>(MatchingRecords, RecordBreakStream, ByteStream, callbackObject, ScanMatchBlocks);
        scanMatchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        scanMatchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

    mMainMethod = E->compile();
}

InternalSearchEngine::InternalSearchEngine(const std::unique_ptr<grep::GrepEngine> & engine)
    : InternalSearchEngine(engine->mGrepDriver) {}

InternalSearchEngine::~InternalSearchEngine() { }


void InternalSearchEngine::doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum) {
    typedef void (*GrepFunctionType)(const char * buffer, const size_t length, MatchAccumulator *);
    auto f = reinterpret_cast<GrepFunctionType>(mMainMethod);
    f(search_buffer, bufferLength, &accum);
}

InternalMultiSearchEngine::InternalMultiSearchEngine(BaseDriver &driver) :
mGrepRecordBreak(GrepRecordBreakKind::LF),
mCaseInsensitive(false),
mGrepDriver(driver),
mMainMethod(nullptr) {

}

InternalMultiSearchEngine::InternalMultiSearchEngine(const std::unique_ptr<grep::GrepEngine> & engine) :
    InternalMultiSearchEngine(engine->mGrepDriver) {}

void InternalMultiSearchEngine::grepCodeGen(const re::PatternVector & patterns) {
    auto & b = mGrepDriver.getBuilder();

    re::CC * breakCC = nullptr;
    if (mGrepRecordBreak == GrepRecordBreakKind::Null) {
        breakCC = re::makeByte(0x0);
    } else {// if (mGrepRecordBreak == GrepRecordBreakKind::LF)
        breakCC = re::makeByte(0x0A);
    }

    auto E = mGrepDriver.makePipeline({Binding{b.getInt8PtrTy(), "buffer"},
        Binding{b.getSizeTy(), "length"},
        Binding{b.getIntAddrTy(), "accumulator"}});

    Scalar * const buffer = E->getInputScalar(0);
    Scalar * const length = E->getInputScalar(1);
    Scalar * const callbackObject = E->getInputScalar(2);
    StreamSet * ByteStream = E->CreateStreamSet(1, 8);
    E->CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);

    StreamSet * RecordBreakStream = E->CreateStreamSet();
    StreamSet * BasisBits = E->CreateStreamSet(8);
    E->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    E->CreateKernelCall<CharacterClassKernelBuilder>(std::vector<re::CC *>{breakCC}, BasisBits, RecordBreakStream);

    StreamSet * u8index = E->CreateStreamSet();
    E->CreateKernelCall<UTF8_index>(BasisBits, u8index);

    StreamSet * resultsSoFar = RecordBreakStream;

    const auto n = patterns.size();

    for (unsigned i = 0; i < n; i++) {
        StreamSet * const MatchResults = E->CreateStreamSet();

        auto options = std::make_unique<GrepKernelOptions>();

        auto r = resolveCaseInsensitiveMode(patterns[i].second, mCaseInsensitive);
        //r = re::exclude_CC(r, breakCC);
        //r = resolveAnchors(r, breakCC);
        r = regular_expression_passes(r);
        r = toUTF8(r);

        options->setBarrier(RecordBreakStream);
        options->setRE(r);
        options->addAlphabet(&cc::UTF8, BasisBits);
        options->setResults(MatchResults);
        const auto isExclude = patterns[i].first == re::PatternKind::Exclude;
        if (i != 0 || !isExclude) {
            options->setCombiningStream(isExclude ? GrepCombiningType::Exclude : GrepCombiningType::Include, resultsSoFar);
        }
        options->addExternal("UTF8_index", u8index);
        E->CreateKernelFamilyCall<ICGrepKernel>(std::move(options));
        resultsSoFar = MatchResults;
    }

    if (MatchCoordinateBlocks > 0) {
        StreamSet * MatchCoords = E->CreateStreamSet(3, sizeof(size_t) * 8);
        E->CreateKernelCall<MatchCoordinatesKernel>(resultsSoFar, RecordBreakStream, MatchCoords, MatchCoordinateBlocks);
        Kernel * const matchK = E->CreateKernelCall<MatchReporter>(ByteStream, MatchCoords, callbackObject);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    } else {
        Kernel * const scanMatchK = E->CreateKernelCall<ScanMatchKernel>(resultsSoFar, RecordBreakStream, ByteStream, callbackObject, ScanMatchBlocks);
        scanMatchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        scanMatchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

    mMainMethod = E->compile();
}

void InternalMultiSearchEngine::doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum) {
    typedef void (*GrepFunctionType)(const char * buffer, const size_t length, MatchAccumulator *);
    auto f = reinterpret_cast<GrepFunctionType>(mMainMethod);
    f(search_buffer, bufferLength, &accum);
}

class LineNumberAccumulator : public grep::MatchAccumulator {
public:
    LineNumberAccumulator() {}
    void accumulate_match(const size_t lineNum, char * line_start, char * line_end) override;
    std::vector<uint64_t> && getAccumulatedLines() { return std::move(mLineNums); }
private:
    std::vector<uint64_t> mLineNums;
};

void LineNumberAccumulator::accumulate_match(const size_t lineNum, char * /* line_start */, char * /* line_end */) {
    mLineNums.push_back(lineNum);
}

std::vector<uint64_t> lineNumGrep(re::RE * pattern, const char * buffer, size_t bufSize) {
    LineNumberAccumulator accum;
    CPUDriver driver("driver");
    grep::InternalSearchEngine engine(driver);
    engine.setRecordBreak(grep::GrepRecordBreakKind::LF);
    engine.grepCodeGen(pattern);
    engine.doGrep(buffer, bufSize, accum);
    return accum.getAccumulatedLines();
}

class MatchOnlyAccumulator : public grep::MatchAccumulator {
public:
    MatchOnlyAccumulator() : mFoundMatch(false) {}
    void accumulate_match(const size_t lineNum, char * line_start, char * line_end) override;
    bool foundAnyMatches() { return mFoundMatch; }
private:
    bool mFoundMatch;
};

void MatchOnlyAccumulator::accumulate_match(const size_t lineNum, char * /* line_start */, char * /* line_end */) {
    mFoundMatch = true;
}

bool matchOnlyGrep(re::RE * pattern, const char * buffer, size_t bufSize) {
    MatchOnlyAccumulator accum;
    CPUDriver driver("driver");
    grep::InternalSearchEngine engine(driver);
    engine.setRecordBreak(grep::GrepRecordBreakKind::Null);
    engine.grepCodeGen(pattern);
    engine.doGrep(buffer, bufSize, accum);
    return accum.foundAnyMatches();
}

}
