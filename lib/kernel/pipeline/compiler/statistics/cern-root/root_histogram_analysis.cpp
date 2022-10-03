#include "root_histogram_analysis.h"

#include <TROOT.h>
#include <TRint.h>
#include <TH1.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TCanvas.h>
#include <TSpectrum.h>
#include <TPDF.h>
#include <TMinuit.h>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <llvm/Support/Compiler.h>
#include <ostream>

using namespace boost::multiprecision;

using binfloat_t = number<cpp_bin_float<14>>;

using namespace ROOT;

#warning compiling ROOT histogram analysis

extern "C" {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief cern_root_analyze_histogram_data
 ** ------------------------------------------------------------------------------------------------------------- */
void cern_root_analyze_histogram_data(const HistogramKernelData * const data, const uint64_t numOfKernels, uint32_t reportType) {

    if (LLVM_UNLIKELY(numOfKernels == 0)) return;

 //   GetROOT()->GetStyle()->SetHistMinimumZero();

    TCanvas canvas("histograms");



   // gROOT->SetStyle(style_name);

   // SetHistMinimumZero

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
            // uint64_t maxValue = 0;
            if (LLVM_UNLIKELY(pd.Size == 0)) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data);
                while (e) {
                    const auto k = e->ItemCount;
                    assert (numOfBins == 0 || numOfBins < k);
                    numOfBins = k; // std::max(numOfBins, e->ItemCount);
                    const auto f = e->Frequency;
                    totalSum += f;
                    // maxValue = std::max(maxValue, f);
                    e = e->Next;
                }
            } else {
                numOfBins = pd.Size;
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < numOfBins; ++k) {
                    const auto f = L[k];
                    totalSum += f;
                    // maxValue = std::max(maxValue, f);
                }
            }

            if (LLVM_UNLIKELY(totalSum == 0)) {
                continue;
            }

            assert (numOfBins > 0);

            binfloat_t fTotalSum(totalSum);

            TH1D hist("", "", numOfBins, 0.0, 1.0);

            hist.Smooth(0);

            auto addBinValue = [&](const uint64_t x, const uint64_t y0) {
                const auto y = binfloat_t{y0} / fTotalSum;
                const auto d = y.convert_to<double>();
                assert (0.0 <= d && d <= 1.0);
                hist.SetBinContent(x, d);
            };

            if (LLVM_UNLIKELY(pd.Size == 0)) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data);
                unsigned prior = 0;
                while (e) {
                    const auto k = e->ItemCount;
                    while (prior < k) {
                        addBinValue(prior++, 0);
                    }
                    addBinValue(k, e->Frequency);
                    e = e->Next;
                    prior = k + 1;
                }
            } else {
                numOfBins = pd.Size;
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < numOfBins; ++k) {
                    addBinValue(k, L[k]);
                }
            }

            // look at how many peaks we have in the data
            TSpectrum S;
            const auto peaks = S.Search(&hist);





            std::array<TF1 *, 4> fits;
            fits[0] = new TF1("a", "pol 1", 0, numOfBins);
            fits[1] = new TF1("b", "pol 2", 0, numOfBins);
            fits[2] = new TF1("c", "expo", 0, numOfBins);
            fits[3] = new TF1("d", "gaus", 0, numOfBins);

            TFitResultPtr fit;
            double prob = 0.0;
            unsigned chosenModel = -1U;

            for (unsigned i = 0; i < fits.size(); ++i) {
                auto nextFit = hist.Fit(fits[i], "SL");
                if (LLVM_LIKELY(nextFit->IsValid())) {
                    auto nextProb = nextFit->Prob();
                    if (nextProb > prob) {
                        prob = nextProb;
                        fit = nextFit;
                        chosenModel = i;
                    }
                }
            }


            std::stringstream title;
            title << K.Id << "." << K.KernelName << ":" << pd.BindingName;

            const auto titleStr = title.str();


            hist.SetTitle(titleStr.c_str());
            if (chosenModel != -1U) {
                auto & fit = fits[chosenModel];
                assert (fit->IsValid());
                hist.Add(fit);
            }
            hist.Draw();

            const std::string suffix = ((i == 0) ? "(" : ((i == (numOfKernels - 1)) ? ")" : ""));

            const char * pdfName = "port-histogram.pdf";
            if (numOfKernels > 1) {
                if (i == 0 && j == 0) {
                    pdfName = "port-histogram.pdf(";
                } else if (i == (numOfKernels - 1) && (j == (numOfPorts - 1))) {
                    pdfName = "port-histogram.pdf)";
                }
            }
            canvas.Print(pdfName, ("Title:" + titleStr).c_str());

            std::cerr << titleStr << '\n' << '\n';

            std::cerr << " PEAKS: " << peaks << "\n";

            if (chosenModel != -1U) {

                const char * modelName = nullptr;

                switch (chosenModel) {
                    case 0: modelName = "Linear"; break;
                    case 1: modelName = "Quadratic"; break;
                    case 2: modelName = "Exponential"; break;
                    case 3: modelName = "Gaussian"; break;
                }


                std::cerr << modelName << '\n' << '\n';

                const auto & params = fit->Parameters();
                const auto numOfParams = params.size();

                for (unsigned i = 0; i < numOfParams; ++i) {
                    std::cerr << "  " << fit->GetParameterName(i) << ": " << std::setprecision(3) << params[i] << '\n';
                }

                std::cerr  << '\n' << "CHI2" << ": " << std::setprecision(3) << fit->Chi2() << '\n';
                std::cerr  << '\n' << "NDF" << ": " << std::setprecision(3) << fit->Ndf() << '\n';

            }

            std::cerr << std::endl;

        }
    }


}

}
