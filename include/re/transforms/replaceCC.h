/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once


namespace re {

    class CC;
    class RE;

    RE * replaceCC(RE * re, CC * toReplace, RE * replacement);
}
