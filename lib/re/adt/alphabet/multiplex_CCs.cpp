/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <map>
#include <boost/dynamic_bitset.hpp>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/re_cc.h>
#include <re/printer/re_printer.h>
#include <re/alphabet/multiplex_CCs.h>
#include <unicode/core/unicode_set.h>

namespace cc {

//
// Breakpoints of a set of character classes (CCs): each codepoint c such that
// there is some CC in CCs such that either (a) c is in the CC and c-1 is not, or
// (b) c-1 is in the CC and c is not.  Boundary cases: if codepoint 0 is in
// some CC, then 0 is a breakpoint (codepoint -1 is not in any CC).  If codepoint
// 0x10FFFF is in some CC then 0x110000 is a breakpoint.
//
// The breakpoints may be determined by iterating through the interval
// representation of each CC.   For each interval (lo, hi), lo and hi+1 
// are breakpoints. 
//
// For each breakpoint, a bitset is computed identifying the source CCs whose
// status changes at the breakpoint.
//

using bitset = boost::dynamic_bitset<>;

std::map<UCD::codepoint_t, bitset> computeBreakpoints(const CC_Set & CCs) {
    std::map<UCD::codepoint_t, bitset> breakpoints;
    for (unsigned i = 0; i < CCs.size(); i++) {
        for (const auto range : *CCs[i]) {
            const auto lo = re::lo_codepoint(range);
            const auto hi = re::hi_codepoint(range);
            auto f = breakpoints.find(lo);
            if (f == breakpoints.end()) {
                breakpoints.emplace(lo, bitset(CCs.size()));
            }
            breakpoints[lo].set(i);
            f = breakpoints.find(hi + 1);
            if (f == breakpoints.end()) {
                breakpoints.emplace(hi+1, bitset(CCs.size()));
            }
            breakpoints[hi+1].set(i);
        }
    }
    return breakpoints;
}

void doMultiplexCCs(const CC_Set & CCs,
                    std::vector<std::vector<unsigned>> & exclusiveSetIDs,
                    CC_Set & multiplexedCCs) {
    
    const auto breakpoints = computeBreakpoints(CCs);
    // Initialize the exclusiveSetIDs to have one empty vector per source CC.
    exclusiveSetIDs.clear();
    exclusiveSetIDs.resize(CCs.size());
    //
    // Exclusive set determination.   
    //
    // Set up a map from the set of source CCs for each exclusive set to the exclusive set index.

    std::map<bitset, unsigned> CC_set_to_exclusive_set_map;

    // Entry 0 is for the characters not in any of the CCs.
    CC_set_to_exclusive_set_map.emplace(bitset(CCs.size()), 0);

    unsigned current_exclusive_set_idx = 0;
    unsigned multiplexed_bit_count = 0;
    bitset current_set(CCs.size());
    
    unsigned range_lo = 0;
    unsigned next_set_index = 1;
    for (auto & bkpt_entry : breakpoints) {
        if (current_exclusive_set_idx > 0) {  // We have a range entry to close for a pending exclusive set.
            unsigned range_hi = bkpt_entry.first - 1;
            for (unsigned bit = 0; bit < multiplexed_bit_count; bit++) {
                if (((current_exclusive_set_idx >> bit) & 1) == 1) {
                    multiplexedCCs[bit]->insert_range(range_lo, range_hi);
                }
            }
        }
        // Start a new range.
        range_lo = bkpt_entry.first;
        if (range_lo > UCD::UNICODE_MAX) continue; // Nothing to do for bkpt 0x110000
        current_set ^= bkpt_entry.second;
        auto idx_iter = CC_set_to_exclusive_set_map.find(current_set);
        if (idx_iter == CC_set_to_exclusive_set_map.end()) {
            // New exclusive class; assign the next sequential integer.
            //current_exclusive_set_idx = exclusiveSetIDs.size();
            current_exclusive_set_idx = next_set_index;
            next_set_index++;
            CC_set_to_exclusive_set_map.emplace(current_set, current_exclusive_set_idx);
            
            for (unsigned CC1 = current_set.find_first(); CC1 < CCs.size(); CC1 = current_set.find_next(CC1)) {
                exclusiveSetIDs[CC1].push_back(current_exclusive_set_idx);
            }
            if ((current_exclusive_set_idx & (current_exclusive_set_idx - 1)) == 0) {
                multiplexed_bit_count++;
                multiplexedCCs.push_back(re::makeCC());
            }
        }
        else {
            current_exclusive_set_idx = idx_iter->second;
        }
    }
}



MultiplexedAlphabet::MultiplexedAlphabet(const std::string name, const CC_Set CCs)
: Alphabet(name, name, ClassTypeId::MultiplexedAlphabet)
, mUnicodeSets(std::move(CCs)) {
    if (!mUnicodeSets.empty()) {
        mSourceAlphabet = mUnicodeSets[0]->getAlphabet();
        for (unsigned i = 1; i < CCs.size(); i++) {
            if (CCs[i]->getAlphabet() != mSourceAlphabet) {
                llvm::report_fatal_error("Mismatched source alphabets for Multiplexed Alphabet");
            }
        }
    }
    doMultiplexCCs(CCs, mExclusiveSetIDs, mMultiplexedCCs);
}

unsigned long MultiplexedAlphabet::findTargetCCIndex(const re::CC * sourceCC) const {
    auto f = find(mUnicodeSets.begin(), mUnicodeSets.end(), sourceCC);
    if (f != mUnicodeSets.end()) {
        return f - mUnicodeSets.begin();
    }

    const UCD::UnicodeSet* sourceUcd = sourceCC;
    for (unsigned long i = 0; i < mUnicodeSets.size(); i++) {
        UCD::UnicodeSet* t = mUnicodeSets[i];
        if (*t == *sourceUcd) {
            return i;
        }
    }
    // llvm::report_fatal_error(Printer_RE::PrintRE(sourceCC) + " not found");
    llvm_unreachable("sourceCC not found");
}

re::CC * MultiplexedAlphabet::transformCC(const re::CC * sourceCC) const {
    if (sourceCC->getAlphabet() != mSourceAlphabet) {
        llvm::report_fatal_error("Mismatched source alphabets for transformCC");
    }

    const auto index = findTargetCCIndex(sourceCC);
    const auto exclusive_IDs = mExclusiveSetIDs[index];
    re::CC * CC_union = re::makeCC(this);
    for (auto i : exclusive_IDs) {
        CC_union = re::makeCC(CC_union, re::makeCC(i, this));
    }
    CC_union->sourceCC = sourceCC;
    return CC_union;
}

re::CC * MultiplexedAlphabet::invertCC(const re::CC * transformedCC) const {
    if (transformedCC->getAlphabet() != this) llvm::report_fatal_error("invertCC applied to non-transformed CC");
    re::CC * CC_union = re::makeCC(mSourceAlphabet);
    for (const UCD::interval_t i : *transformedCC) {
        for (unsigned cp = re::lo_codepoint(i); cp <= re::hi_codepoint(i); cp++) {
            CC_union = re::makeCC(mUnicodeSets[cp], CC_union);
        }
    }
    return CC_union;
}
    

    
}
