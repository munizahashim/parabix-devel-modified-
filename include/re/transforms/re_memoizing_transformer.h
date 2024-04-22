#pragma once

#include <re/transforms/re_transformer.h>
#include <re/adt/memoization.h>
#include <map>

namespace re {

// NOTE: this class is a utility class to provide memoization functionality
// to other RE transformers and is not intended to be used on its own.
class RE_MemoizingTransformer : public RE_Transformer {
protected:
    RE_MemoizingTransformer(std::string transformationName)
    : RE_Transformer(std::move(transformationName)) {

    }

    RE * transform(RE * const from) override;
private:
    std::map<RE *, RE *, MemoizerComparator> mMap;
};

}

