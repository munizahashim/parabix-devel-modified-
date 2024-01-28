#ifndef PE_ILLUSTRATOR_H
#define PE_ILLUSTRATOR_H

#include <pablo/pabloAST.h>
#include <pablo/pe_integer.h>
#include <kernel/illustrator/illustrator_binding.h>

namespace pablo {

class Illustrate final : public Statement {
    friend class PabloBlock;
public:
    enum class IllustratorTypeId : unsigned {
        Bitstream
        , BixNum
        , ByteStream
    };
    static inline bool classof(const PabloAST * e) {
        return e->getClassTypeId() == ClassTypeId::EveryNth;
    }
    static inline bool classof(const void *) {
        return false;
    }
    virtual ~Illustrate() {
    }
    inline PabloAST * getExpr() const {
        return getOperand(0);
    }

    inline IllustratorTypeId getIllustratorType() const {
        return IllustratorType;
    }

    inline char getReplacementCharacter(const size_t i) const {
        return ReplacementCharacter[i];
    }

protected:
    explicit Illustrate(PabloAST * expr, IllustratorTypeId illustratorType, const char replacement0, const char replacement1, const String * name, Allocator & allocator)
    : Statement(ClassTypeId::Illustrator, expr->getType(), {expr}, name, allocator)
    , IllustratorType(illustratorType)
    , ReplacementCharacter({replacement0, replacement1})
    {

    }
private:
    const IllustratorTypeId IllustratorType;
    const std::array<char, 2> ReplacementCharacter;
};

}

#endif // PE_ILLUSTRATOR_H
