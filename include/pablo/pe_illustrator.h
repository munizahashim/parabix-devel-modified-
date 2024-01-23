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
    inline Integer * getN() const {
        return llvm::cast<Integer>(getOperand(1));
    }
    inline IllustratorTypeId getIllustratorType() const {
        return IllustratorType;
    }
protected:
    explicit Illustrate(PabloAST * expr, PabloAST * n, const String * name, Allocator & allocator)
    : Statement(ClassTypeId::Illustrator, expr->getType(), {expr, n}, name, allocator) {
        assert(llvm::isa<Integer>(n) && llvm::cast<Integer>(n)->value() != 0);
    }
private:
    const IllustratorTypeId IllustratorType;
    const std::array<char, 2> ReplacementCharacter;
};


#endif // PE_ILLUSTRATOR_H
