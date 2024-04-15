#pragma once

#include <vector>

namespace re {

class CC;
class RE;

//  Given a fixed-length sequence of character classes, determine
//  whether any string formed by concatenating one character each
//  from these classes could be found as a substring of any string
//  matched by a given RE.
//

bool CC_Sequence_Search(std::vector<CC *> & CC_seq, RE * re);

}
