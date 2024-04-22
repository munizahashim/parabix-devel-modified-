/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/core/callback.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_os_ostream.h>

using namespace kernel;


extern "C" void signal_dispatcher(intptr_t callback_object_addr, unsigned signal) {
    reinterpret_cast<SignallingObject *>(callback_object_addr)->handle_signal(signal);
}

void SignallingObject::handle_signal(unsigned s) {
    mSignalCount++;
    mLastSignal = s;
}

unsigned SignallingObject::getSignalCount() {
    return mSignalCount;
}

unsigned SignallingObject::getLastSignal() {
    return mLastSignal;
}
