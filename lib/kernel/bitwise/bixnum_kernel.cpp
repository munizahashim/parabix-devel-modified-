/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/bitwise/bixnum_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>

using namespace bixnum;

void Add::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum a = getInputStreamSet("a");
    pablo::BixNum b = getInputStreamSet("b");
    pablo::Var * sumVar = getOutputStreamVar("sum");
    if (a.size() > mBixBits) a = bnc.Truncate(a, mBixBits);
    if (b.size() > mBixBits) b = bnc.Truncate(b, mBixBits);
    while(a.size() < mBixBits) {
        a.push_back(pb.createZeroes());
    }
    pablo::BixNum sum = bnc.AddModular(a, b);
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(sumVar, i), sum[i]);
    }
}
