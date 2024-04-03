#ifndef ILLUSTRATOR_BINDING_H
#define ILLUSTRATOR_BINDING_H

#include <string>
#include <array>

namespace kernel {

class StreamSet;

enum class IllustratorTypeId : uint8_t {
    Bitstream
    , BixNum
    , ByteData
};

struct IllustratorBinding {
    const IllustratorTypeId IllustratorType;
    std::string Name;
    kernel::StreamSet * StreamSetObj;
    const std::array<char, 2> ReplacementCharacter;

    IllustratorBinding(IllustratorTypeId type, std::string name, kernel::StreamSet * streamSet, char rep0 = '\0', char rep1 = '\0')
    : IllustratorType(type), Name(std::move(name)), StreamSetObj(streamSet), ReplacementCharacter({rep0, rep1}) {

    }

};

}

#endif // ILLUSTRATOR_BINDING_H
