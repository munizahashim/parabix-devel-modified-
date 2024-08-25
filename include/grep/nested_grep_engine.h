#include "grep_engine.h"

namespace grep {

class NestedInternalSearchEngine {
    typedef void (*GrepFunctionType)(const char * buffer, const size_t length, MatchAccumulator &);
public:

    NestedInternalSearchEngine(BaseDriver & driver);

    ~NestedInternalSearchEngine();

    void setRecordBreak(GrepRecordBreakKind b) {mGrepRecordBreak = b;}

    void setCaseInsensitive()  {mCaseInsensitive = true;}

    void init();

    void push(const re::PatternVector & REs);

    void pop();

    void grepCodeGen();

    void doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum);

private:
    GrepRecordBreakKind mGrepRecordBreak;
    bool mCaseInsensitive;
    BaseDriver & mGrepDriver;
    re::CC * mBreakCC;
    kernel::StreamSet * mBasisBits;
    kernel::StreamSet * mU8index;
    kernel::StreamSet * mBreaks;
    kernel::StreamSet * mMatches;
    std::vector<GrepFunctionType>   mMainMethod;
    std::vector<kernel::Kernel *>   mNested;

};


}
