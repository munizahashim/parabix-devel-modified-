#include "root_histogram_analysis.h"

#include <TH1.h>
#include <TFitResult.h>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <ostream>

using namespace boost::multiprecision;

using cpp_bin_float_7 = number<cpp_bin_float<7>>;

using namespace ROOT;

#warning compiling ROOT histogram analysis

extern "C" {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief cern_root_analyze_histogram_data
 ** ------------------------------------------------------------------------------------------------------------- */
void cern_root_analyze_histogram_data(const HistogramKernelData * const data, const uint64_t numOfKernels, uint32_t reportType) {

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto & K = data[i];
//        maxKernelId = std::max(maxKernelId, K.Id);
//        maxKernelNameLength = std::max(maxKernelNameLength, strlen(K.KernelName));
        const auto numOfPorts = K.NumOfPorts;
        for (unsigned j = 0; j < numOfPorts; ++j) {
            const HistogramPortData & pd = K.PortData[j];
//            maxPortNum = std::max(maxPortNum, pd.PortNum);
//            maxBindingNameLength = std::max(maxBindingNameLength, strlen(pd.BindingName));

            uint64_t numOfBins = 0;
            uint64_t totalSum = 0;

            if (pd.Size == 0) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data);
                while (e) {
                    assert (numOfBins == 0 || numOfBins < e->ItemCount);
                    numOfBins = e->ItemCount; // std::max(numOfBins, e->ItemCount);
                    totalSum += e->Frequency;
                    e = e->Next;
                }
            } else {
                numOfBins = pd.Size;
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < numOfBins; ++k) {
                    totalSum += L[k];
                }
            }

            std::cerr << "numOfBins=" << numOfBins << ", totalSum=" << totalSum << std::endl;

            if (numOfBins == 0) {
                continue;
            }

            assert (totalSum > 0);

            cpp_bin_float_7 fTotalSum(totalSum);

            TH1F * const hist = new TH1F("", "", numOfBins, 0.0, 1.0);

            auto addBinValue = [&](const uint64_t x, const uint64_t y0) {
                const auto y = cpp_bin_float_7{y0} / fTotalSum;
                hist->AddBinContent(x, y.convert_to<double>());
            };

            if (pd.Size == 0) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data);
                // only the root node might have a frequency of 0
                uint64_t prior = 0;
                while (e) {
                    const auto x = e->ItemCount;
                    assert (x == 0 || prior < x);
                    for (auto i = prior; i < x; ++i) {
                        addBinValue(i, 0);
                    }
                    addBinValue(x, e->Frequency);
                    e = e->Next;
                    prior = x + 1;
                }
            } else {
                numOfBins = pd.Size;
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < numOfBins; ++k) {
                    addBinValue(k, L[k]);
                }
            }

            std::cerr << "poly1Fit" << std::endl;
            const auto poly1Fit = hist->Fit("pol 1");

            std::cerr << "expoFit" << std::endl;
            const auto expoFit = hist->Fit("expo");

            std::cerr << "gausFit" << std::endl;
            const auto gausFit = hist->Fit("gaus");

            std::cerr << "chi2" << std::endl;

            const auto poly1Chi2 = poly1Fit->Chi2();
            const auto expoChi2 = expoFit->Chi2();
            const auto gausChi2 = gausFit->Chi2();

            auto chi2 = poly1Chi2;
            auto fit = poly1Fit;

            if (expoChi2 < chi2) {
                chi2 = expoChi2;
                fit = expoFit;
            }
            if (gausChi2 < chi2) {
                chi2 = gausChi2;
                fit = gausFit;
            }


            std::cerr << K.Id << "." << K.KernelName << ":" << pd.BindingName << "\n\n";

            std::cerr << fit->ClassName() << "\n\n";

            std::cerr << "CHI2" << ": " << std::setprecision(3) << chi2 << '\n';

            const auto & params = fit->Parameters();
            const auto numOfParams = params.size();

            for (unsigned i = 0; i < numOfParams; ++i) {
                std::cerr << fit->GetParameterName(i) << ": " << std::setprecision(3) << params[i] << '\n';
            }

            std::cerr << std::endl;

        }
    }


}

}
