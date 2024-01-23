#ifndef ILLUSTRATOR_BINDING_H
#define ILLUSTRATOR_BINDING_H

#include <string>
#include <array>

namespace kernel {

enum class IllustratorTypeId : unsigned {
    None
    , Bitstream
    , BixNum
    , ByteStream
};

struct IllustratorBinding {
    const IllustratorTypeId IllustratorType;
    std::string Name;
    const std::array<char, 2> ReplacementCharacter;
};

}

#endif // ILLUSTRATOR_BINDING_H
