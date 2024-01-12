#pragma once

namespace cc { class Alphabet;}
namespace re { class RE;}

//  Find and promote variable-length Alts from within Seqs.
//  For example,  a(bc|cde|de)f becomes a(bc|de)f|acdef.

namespace re {
RE * variableAltPromotion(RE * r, const cc::Alphabet * lengthAlphabet);
}
