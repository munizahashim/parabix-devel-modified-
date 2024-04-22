#pragma once

namespace re {

class RE;

//A regular expression E is in star normal form if, for each starred
//subexpression H * of E, the following SNF-conditions hold:
//1> The follow(H, last(H)) and first(H) are disjoint.
//2> H is not Nullable.
//
//For example: (a + b)* is the star normal form of (a*b*)* .
//Both of them have the same Glushkov NFA.

RE * convertToStarNormalForm(RE * re);

}

