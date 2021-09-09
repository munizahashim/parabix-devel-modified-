/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */
#ifndef ZTF_PHRASELOGIC_H
#define ZTF_PHRASELOGIC_H

#include <pablo/pablo_kernel.h>
#include <kernel/core/kernel_builder.h>
#include "ztf-logic.h"

namespace kernel {

class InverseStream : public pablo::PabloKernel {
public:
    InverseStream(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * prevMarks,
                unsigned groupNum,
                StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNum;
};

}
#endif

