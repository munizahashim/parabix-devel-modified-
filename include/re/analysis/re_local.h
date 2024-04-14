#pragma once

namespace re {

class RE; class CC;

struct RE_Local {
    static CC * getFirstUniqueSymbol(RE * re);
};

}

