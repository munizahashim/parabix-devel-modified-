/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

//#pragma once
#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <kernel/pipeline/driver/cpudriver.h>
//#include "csv_util.hpp"
//#include "colCut.hpp"
//#include "colCut.cpp"

using namespace kernel;
using namespace llvm;
using namespace pablo;
using namespace std;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory CSV_Options("CSV Processing Options");
static cl::opt<std::string> HeaderInput(cl::Positional, cl::desc("Specified Header"),cl::init(""), cl::cat(CSV_Options));
// static cl::opt<int> HeaderIndex(cl::Positional, cl::desc("Header Index"), cl::cat(CSV_Options));
static cl::opt<std::string> DataInput(cl::Positional, cl::desc("Specified Data"),cl::init(""), cl::cat(CSV_Options));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("input file"), cl::Required, cl::cat(CSV_Options));
//static cl::opt<std::string> OutputFile(cl::Positional, cl::desc("Output file"), cl::Required, cl::cat(CSV_Options));
//static cl::opt<bool> HeaderSpecNamesFile("f", cl::desc("Interpret headers parameter as file name with header line"), cl::init(false), cl::cat(CSV_Options));
static cl::opt<std::string> HeaderSpec("headers", cl::desc("CSV column headers (explicit string or filename"), cl::init(""), cl::cat(CSV_Options));
static cl::opt<std::string> Seperator("sep", cl::desc("CSV seperators"), cl::init(","), cl::cat(CSV_Options));
static cl::opt<bool> KeepRows("keep", cl::desc("keep rows only containing the specified data"), cl::init(false),cl::cat(CSV_Options));
static cl::opt<bool> DropRows("drop", cl::desc("drop rows only containing the specified data"), cl::init(false),cl::cat(CSV_Options));
static cl::opt<bool> CutCols("cut", cl::desc("delete columns only containing the specified data"), cl::init(false),cl::cat(CSV_Options));
static cl::opt<bool> ColsIndex("index", cl::desc("delete columns by index"), cl::init(false),cl::cat(CSV_Options));

typedef void (*CSVFunctionType)(uint32_t fd);

class colCut{
    public:
        colCut(string filename);
        void cutByName(string userInput, char symbol=',');
        void cutByIndex(int index, char symbol=',');
        ~colCut();
    private:
        ifstream fin;
        ofstream fout;


};

vector<vector<string>> readFile(ifstream &inputFile, char sym) {
    vector<vector<string>> list;
    vector<string> row;
    string line, data;


    while(getline(inputFile, line)) {
        stringstream str(line);
        row.clear();                // empty the vector to store new line data

        while(getline(str, data, sym)) {
            row.push_back(data);
        }
        list.push_back(row);
    }
    return list;
}

void writeFile(vector<vector<string>> v) {
    ofstream output("output.csv");

    for(int i = 0; i < v.size(); i++) {
        int len = v[i].size();
        for(int j = 0; j < len; j++) {
            output << v[i][j];
            if(j < len - 1) {
                output << ",";          // add separator if not the last column
            }else {
                output << "\n";         // add newline if the last column
            }
        }
    }
    output.close();
}
void filtration(string inputFile, string HeaderInput, string DataInput, bool drop, string Seperator ) {
    ifstream myFile (inputFile);
    char c =Seperator[0];
    vector<vector<string>> dataList = readFile(myFile, c);
    vector<vector<string>> newList;         // new list for the output file
    newList.push_back(dataList[0]);         // insert the first row, the column names

    if(HeaderInput == "*") {
        for(int i = 1; i < dataList.size(); i++) {
            if(find(dataList[i].begin(), dataList[i].end(), DataInput) == dataList[i].end()) { // cannot find target data


                if(drop) {                              // drop mode
                    newList.push_back(dataList[i]);     // add the row to newList
                }

            }else if(!drop) {           // keep mode
                newList.push_back(dataList[i]);     // add the row to newList
            }
        }
    }else {
        int targetPos;

        if(HeaderInput.find_first_not_of("0123456789") == string::npos) {       // if colName is a integer

            targetPos = stoi(HeaderInput) - 1;
        }else {
            vector<string>::iterator colPos;
            colPos = find(dataList[0].begin(), dataList[0].end(), HeaderInput);
            targetPos = distance(dataList[0].begin(), colPos);
        }


        for(int i = 1; i < dataList.size(); i++) {

            // drop == true => drop; drop == false => keep
            // in drop mode, if the data at that column does not match
            if(dataList[i][targetPos] != DataInput) {
                if(drop) {
                    newList.push_back(dataList[i]);         // insert the whole row to new list
                }
            // in keep mode, if the data at that column match
            }else if(!drop) {
                newList.push_back(dataList[i]);         // insert the whole row to new list
            }
        }
    }

    writeFile(newList);
    myFile.close();
}

//CSVFunctionType generatePipeline(CPUDriver & pxDriver, std::vector<std::string> templateStrs) {
//    // A Parabix program is build as a set of kernel calls called a pipeline.
//    // A pipeline is construction using a Parabix driver object.
//    auto & b = pxDriver.getBuilder();
//    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}}, {});
//    //  The program will use a file descriptor as an input.
//    Scalar * fileDescriptor = P->getInputScalar("inputFileDecriptor");
//    StreamSet * ByteStream = P->CreateStreamSet(1, 8);
//    //  ReadSourceKernel is a Parabix Kernel that produces a stream of bytes
//    //  from a file descriptor.
//    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
//
//    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
//    //  S2P stands for serial-to-parallel.
//    StreamSet * BasisBits = P->CreateStreamSet(8);
//    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
//
//    //  We need to know which input positions are dquotes and which are not.
//    StreamSet * csvCCs = P->CreateStreamSet(5);
//    P->CreateKernelCall<CSVlexer>(BasisBits, csvCCs);
//
//    StreamSet * recordSeparators = P->CreateStreamSet(1);
//    StreamSet * fieldSeparators = P->CreateStreamSet(1);
//    StreamSet * quoteEscape = P->CreateStreamSet(1);
//    StreamSet * toKeep = P->CreateStreamSet(1);
//    P->CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldSeparators, quoteEscape, toKeep, HeaderSpec == "");
//
//    StreamSet * filteredBasis = P->CreateStreamSet(8);
//    FilterByMask(P, toKeep, BasisBits, filteredBasis);
//    StreamSet * filtered = P->CreateStreamSet(1, 8);
//    P->CreateKernelCall<P2SKernel>(filteredBasis, filtered);
//    P->CreateKernelCall<StdOutKernel>(filtered);
//
//    P->CreateKernelCall<DebugDisplayKernel>("fieldSeparators", fieldSeparators);
//    P->CreateKernelCall<DebugDisplayKernel>("recordSeparators", recordSeparators);
//    //P->CreateKernelCall<DebugDisplayKernel>("compressedSepNum", compressedSepNum);
//    //P->CreateKernelCall<DebugDisplayKernel>("compressedFieldNum", compressedFieldNum);
//    //P->CreateKernelCall<DebugDisplayKernel>("fieldNum", fieldNum);
//    //P->CreateKernelCall<DebugDisplayKernel>("InsertBixNum", InsertBixNum);
//    //P->CreateKernelCall<DebugDisplayKernel>("ExpandedBasis", ExpandedBasis);
//
//
//    if(KeepRows){
//        printf("keep success");
//        printf("\n");
//    }else if(CutCols){
//        printf("cut success");
//        printf("\n");
//    }else if(DropRows){
//        printf("drop success");
//        printf("\n");
//    }
//
//
//    return reinterpret_cast<CSVFunctionType>(P->compile());
//}

const unsigned MaxHeaderSize = 24;

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&CSV_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});


        if(KeepRows){
            bool drop = false;
            if(ColsIndex){
            filtration(inputFile,HeaderInput,DataInput,drop,Seperator );
            printf("keep rows(by index) success");
            printf("\n");
            }

            else if(ColsIndex==false){
            
            filtration(inputFile,HeaderInput,DataInput,drop,Seperator );
            printf("keep rows success(by name)");
            printf("\n");
            }
        }else if(CutCols){
            colCut* cCut;
            cCut= new colCut(inputFile);

            if (ColsIndex){
            int index=stoi(HeaderInput);
            cCut->cutByIndex(index);
            printf("cut cols(by index) success");
            printf("\n");

            }

            else if (ColsIndex==false){
                cCut->cutByName(HeaderInput);
                printf("cut cols(by name) success");
                printf("\n");

            }
        }else if(DropRows){
            bool drop = true;

            if(ColsIndex){
            filtration(inputFile,HeaderInput,DataInput,drop,Seperator );
            printf("drop rows(by index) success");
            printf("\n");
            }
            
            if(ColsIndex==false){
            filtration(inputFile,HeaderInput,DataInput,drop,Seperator );
            printf("keep rows(by name) success");
            printf("\n");
            }
        }

//    std::vector<std::string> headers;
//    if (HeaderSpec == "") {
//        headers = get_CSV_headers(inputFile);
//    }
//    } else if (HeaderSpecNamesFile) {
//       headers = get_CSV_headers(HeaderSpec);
//  } else {
//      headers = parse_CSV_headers(HeaderSpec);
//    //   }
//    for (auto & s : headers) {
//        if (s.size() > MaxHeaderSize) {
//            s = s.substr(0, MaxHeaderSize);
//        }
//    }
//
//    std::vector<std::string> templateStrs = createJSONtemplateStrings(headers);
    // ...
//    for (std::string i: templateStrs){
//        std::cout << i << ' ';}
    //  A CPU driver is capable of compiling and running Parabix programs on the CPU.
   // CPUDriver driver("csv_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    //CSVFunctionType fn = generatePipeline(driver, templateStrs);
    //  The compile function "fn"  can now be used.   It takes a file
    //  descriptor as an input, which is specified by the filename given by
    //  the inputFile command line option.]

    //const int fd = open(inputFile.c_str(), O_RDONLY);
    //if (LLVM_UNLIKELY(fd == -1)) {
    //    llvm::errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    //} else {
        //  Run the pipeline.
//        fflush(stdout); //writing date to
//        fn(fd);
//        close(fd);
//        printf("\n");
//    }
    return 0;
}


colCut::colCut(string filename){
    fin.open(filename);
    fout.open("ModifedFile.csv");

}

colCut::~colCut(){}

void colCut::cutByName(string userInput, char symbol){
    string row;
    string tempRow;
    string tempColName;
    string colName;
    int colIndex=0;
    int maxIndex=0;
    int tempIndex=0;


    fin>>row;

    // find max column index

    for (char& c:row){
        if(c==symbol){
            maxIndex++;
        }
    }


    // check user input

    for (char& c:row){
        if(c!=symbol){
            colName=colName+c;
        }
        else{
            if(colName==userInput){
                break;
            }
            else{
                colName="";
                colIndex++;
            }
        }
    }




    // cut first row

    for (char& c:row){
        if(c!=symbol){
            tempColName=tempColName+c;
        }
        else{

            if(tempColName!=colName){
                tempRow=tempRow+tempColName+',';
                tempIndex=tempIndex+1;
            }

            tempColName="";
        }
    }
    if(colIndex<maxIndex){
        tempRow=tempRow+tempColName;
    }
    if(colIndex==maxIndex){
        tempRow=tempRow.substr(0,tempRow.length()-1);
    }

    fout<<tempRow<<endl;


    // cut following rows

    while (fin>>row){
        tempRow="";
        tempIndex=0;
        tempColName="";
        for (char& c:row){
            if(c!=symbol){
                tempColName=tempColName+c;
            }
            else{

                if(tempIndex!=colIndex){
                    tempRow=tempRow+tempColName+',';
                    tempIndex++;
                }
                else if(tempIndex==colIndex){
                    tempIndex++;
                }

                tempColName="";
            }
        }
        if(colIndex<maxIndex){
            tempRow=tempRow+tempColName;
        }
        if(colIndex==maxIndex){
            tempRow=tempRow.substr(0,tempRow.length()-1);
        }

        fout<<tempRow<<endl;
    }




    fin.close();
    fout.close();

}


void colCut::cutByIndex(int index, char symbol){

    string row;
    string colName;
    int colIndex=0;

    index--;
    fin>>row;


    // find corresbonding column name

    for (char& c:row){
        if(c!=symbol){
            colName=colName+c;
        }
        else{

            if(colIndex==index){
                break;
            }
            else{
                colName="";
                colIndex++;
            }
        }
    }
    // cout<<colName<<endl;
    fin.seekg(0);
    cutByName(colName, symbol);


}
