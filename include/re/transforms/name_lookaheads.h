/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

namespace re {
class RE;

/* Transform a regular expression r so that all names are
   created for all lookahead assertions. */
RE * name_lookaheads(RE * r);
}

