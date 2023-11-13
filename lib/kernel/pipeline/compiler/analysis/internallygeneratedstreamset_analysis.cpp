#include "pipeline_analysis.hpp"

namespace kernel {

void PipelineAnalysis::mapInternallyGeneratedStreamSets() {

    InternallyGeneratedStreamSetGraph G(LastStreamSet + 1);

    if (mPipelineKernel->hasInternallyGeneratedStreamSets()) {

        const auto & S = mPipelineKernel->getInternallyGeneratedStreamSets();

        const auto n = S.size();

        flat_map<const Relationship *, unsigned> M;
        M.reserve(n);

        for (unsigned i = 0; i < n; ++i) {
            const auto s = S[i];
            InternallyGeneratedStreamSetGraph::vertex_descriptor streamSet = FirstStreamSet;
            for (; streamSet <= LastStreamSet; ++streamSet) {
                const RelationshipNode & rn = mStreamGraph[streamSet];
                assert (rn.Type == RelationshipNode::IsStreamSet);
                if (rn.Relationship == s) {
                    goto found_streamset;
                }
            }
            streamSet = add_vertex(G);
found_streamset:
            add_edge(PipelineInput, streamSet, i, G);
            M.emplace(s, streamSet);
        }

        for (auto k = FirstKernel; k <= LastKernel; ++k) {
            const auto kernel = getKernel(k);
            if (LLVM_UNLIKELY(kernel->hasInternallyGeneratedStreamSets())) {
                const auto & V = kernel->getInternallyGeneratedStreamSets();
                const auto m = V.size();
                for (unsigned j = 0; j < m; ++j) {
                    const auto f = M.find(V[j]);
                    if (LLVM_UNLIKELY(f == M.end())) {
                        SmallVector<char, 256> tmp;
                        raw_svector_ostream out(tmp);
                        out << "Compile error: " << kernel->getName() << " required an internally generated streamset "
                               " that was not provided to it by its outer pipeline, " <<
                               mPipelineKernel->getName() << ".";
                        report_fatal_error(StringRef(out.str()));
                    }
                    const auto streamSet = f->second;
                    add_edge(streamSet, k, j, G);
                }
            }
        }

    } else {

        for (auto k = FirstKernel; k <= LastKernel; ++k) {
            const auto kernel = getKernel(k);
            if (LLVM_UNLIKELY(kernel->hasInternallyGeneratedStreamSets())) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << "Compile error: " << kernel->getName() << " is marked as having internally"
                       " generated streamsets but its outer pipeline " <<
                       mPipelineKernel->getName() << " is not flagged as having any.";
                report_fatal_error(StringRef(out.str()));
            }
        }

    }

    mInternallyGeneratedStreamSetGraph = G;
}

}
