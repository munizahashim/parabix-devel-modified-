#ifndef ILLUSTRATOR_BINDING_H
#define ILLUSTRATOR_BINDING_H

#include <string>
#include <array>

namespace kernel {

class StreamSet;

enum class IllustratorTypeId : unsigned {
    None
    , Bitstream
    , BixNum
    , ByteStream
};

struct IllustratorBinding {
    const IllustratorTypeId IllustratorType;
    std::string Name;
    StreamSet * StreamSet;
    const std::array<char, 2> ReplacementCharacter;

    IllustratorBinding(IllustratorTypeId type, std::string name, kernel::StreamSet * streamSet, char rep0 = '\0', char rep1 = '\0')
    : IllustratorType(type), Name(std::move(name)), StreamSet(streamSet), ReplacementCharacter({rep0, rep1}) {

    }

};

}

#endif // ILLUSTRATOR_BINDING_H
