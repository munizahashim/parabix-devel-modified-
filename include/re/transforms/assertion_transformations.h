/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

namespace re {

class RE;

RE * expandBoundaryAssertions(RE * r);
    
RE * lookaheadPromotion(RE * r);

}
