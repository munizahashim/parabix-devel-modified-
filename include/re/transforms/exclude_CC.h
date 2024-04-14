/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

namespace re {
    class RE;
    class CC;
    
    /* Transform a regular expression r so that matched strings do not include
       matches to any character within the given character class cc.
       (However, do not transform assertions, so that lookahead or lookbehind
        may still require matches to cc.  */
    RE * exclude_CC(RE * r, CC * cc, bool processAsserted = false);
}

