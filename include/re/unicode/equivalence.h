/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <string>
#include <locale>
#include <codecvt>
#include <unicode/core/unicode_set.h>
#include <unicode/data/Equivalence.h>

namespace re { class RE; class CC; class Seq; class Group;}
namespace UCD { class EnumeratedPropertyObject; class StringPropertyObject;}

namespace UCD {

re::RE * addClusterMatches(re::RE * r, UCD::EquivalenceOptions options = UCD::Canonical);

re::RE * addEquivalentCodepoints(re::RE * r, UCD::EquivalenceOptions options = UCD::Canonical);

}
