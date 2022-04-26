#include <kernel/core/idisa_target.h>

#include <re/alphabet/alphabet.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/SJIS_data.h>
#include <pablo/bixnum/utf8gen.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/util/error_monitor_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/boolean.h>
#include <pablo/pablo_kernel.h>
#include <toolchain/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <toolchain/toolchain.h>
#include <llvm/Support/CommandLine.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <fstream>
#include <map>
#include <typeinfo>
#include <string>

using namespace std;
using namespace pablo;
using namespace kernel;
using namespace llvm;
using namespace codegen;
using namespace re;

static cl::OptionCategory shiftJISOptions("shiftJIS Options", "Transcoding control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(shiftJISOptions));

enum class SJISerrorMode{Abort, DropBadInput, UseReplacementChar};
static cl::opt<SJISerrorMode> SJISerrorOption(cl::desc("Treatment of erroneous Shift JIS input (default Abort)"),
    cl::values(
        clEnumValN(SJISerrorMode::Abort, "abort-on-error", "generate output for all valid Shift JIS input up to the first error"),
        clEnumValN(SJISerrorMode::DropBadInput, "drop-bad-characters", "drop bad Shift JIS characters from the input"),
        clEnumValN(SJISerrorMode::UseReplacementChar, "use-replacement-char", "replace bad input characters with a replacement character")
        CL_ENUM_VAL_SENTINEL), cl::cat(shiftJISOptions), cl::init(SJISerrorMode::Abort));

enum SJISAddOptions {
    fromUTF16Encoding, fromUniToUTF8Encoding, InEncoding, OutEncoding, fromASCIItoSJIS, fromSJIStoASCII, Usage, VersionInfo, Verbose
};
static cl::opt<SJISAddOptions> SJISerrorOption2(cl::desc("Treatment of encoding and non-encoding Shift JIS options"),
  cl::values(
             clEnumValN(SJISAddOptions::fromUTF16Encoding, "f2", "Input encoding (default: UTF-16)"),
             clEnumValN(SJISAddOptions::fromUniToUTF8Encoding, "f1", "Input encoding (default: UTF-8)"),
             clEnumValN(SJISAddOptions::InEncoding, "f", "Input encoding (default: Unicode)"),
             clEnumValN(SJISAddOptions::OutEncoding, "t", "Output encoding (default: SJIS)"),
             clEnumValN(SJISAddOptions::fromASCIItoSJIS, "a", "Input ASCII encoding and output SJIS (default: ASCII)"),
             clEnumValN(SJISAddOptions::fromSJIStoASCII, "s", "Input SJIS encoding and output ASCII (default: SJIS)"),
             clEnumValN(SJISAddOptions::Usage, "u", "Ouput Usage Information"),
             clEnumValN(SJISAddOptions::VersionInfo, "shift-version", "Ouput Version Information"),
             clEnumValN(SJISAddOptions::Verbose, "verbose", "Ouput Verbose Characters")
             CL_ENUM_VAL_SENTINEL), cl::cat(shiftJISOptions));

//function definitions
void fromUTF16 (map<string, string> UTF16_to_Unicode, string inputFile);
void fromUniToUTF8EncodingFunc(map<string, string> unicode_to_utf8, string inputFile);
void fromEncoding(map<string, string> sjis_to_unicodeMap, map<string, string> char_to_sjis, string inputFile);
void toEncoding(map<string, string> unicode_to_sjisMap, map<string, string> char_to_unicode, string inputFile);
void asciiToSJIS(map<string, string> ascii_to_sjisMap, map<string, string> char_to_ascii, string inputFile);
void sjisToASCII(map<string, string> unicode_to_sjisMap, map<string, string> char_to_sjis, string inputFile);
void u_usage();
void v_version();
void v_verbose(map<string, string> char_to_unicode, map<string, string> unicode_to_sjisMap, map<string, string> sjis_to_char, string inputFile);

int main(int argc, char *argv[]) {

    cl::ParseCommandLineOptions(argc, argv);
        //===================================UNICODE & UTF-16 MAP==========================================
    string line4;
    fstream u16Map_File;
    u16Map_File.open("../tools/transcoders/UnicodeuUTF16Map.txt"); //Input file to create the map

    //check that the file can be opened
    if (!u16Map_File.is_open()) {
        cout << "Unable to open file UnicodeUTF16Map.txt" << endl << endl;
        return -1;
    }

    getline(u16Map_File, line4); //get the first line

    string chars, unicode4, utf16;

    map<string, string> unicode_to_utf16;
    map<string, string> utf16_to_unicode;

    while (getline(u16Map_File, line4)) {
        istringstream ss(line4); //Input stream class to operate on strings

        getline(ss, chars, '\t'); //get unicode separated by tab
        getline(ss, unicode4, '\t'); //get unicode separated by tab
        getline(ss, utf16, '\n'); //get utf16 equivalent

        unicode_to_utf16[unicode4] = utf16;
        utf16_to_unicode[utf16] = unicode4;
    }
    u16Map_File.close();

    //===================================UNICODE & UTF-8 MAP==========================================
    string line3;
    fstream u8Map_File;
    u8Map_File.open("../tools/transcoders/UnicodeuUTF8Map.txt"); //Input file to create the map

    //check that the file can be opened
    if (!u8Map_File.is_open()) {
        cout << "Unable to open file UnicodeUTF8Map.txt" << endl << endl;
        return -1;
    }

    getline(u8Map_File, line3); //get the first line

    string unicode3, utf8;

    map<string, string> unicode_to_utf8;
    map<string, string> utf8_to_unicode;

    while (getline(u8Map_File, line3)) {
        istringstream ss(line3); //Input stream class to operate on strings

        getline(ss, unicode3, '\t'); //get unicode separated by tab
        getline(ss, utf8, '\n'); //get utf8 equivalent

        unicode_to_utf8[unicode3] = utf8;
        utf8_to_unicode[utf8] = unicode3;
    }
    u8Map_File.close();

    //===================================CHACTER & UNICODE & SJIS MAP==========================================
    string line2;
    fstream fullMap_File;
    fullMap_File.open("../tools/transcoders/CharUnicodeSJISMap.txt"); //Input file to create the map

    //check that the file can be opened
    if (!fullMap_File.is_open()) {
        cout << "Unable to open file CharUnicodeSJISMap.txt" << endl << endl;
        return -1;
    }

    getline(fullMap_File, line2); //get the first line

    //Declare variables used in the main function
    string charVal, unicode2, sjis2, extrachars2;
    string yenSign = "¥";

    int asciiCounter2 = 0;

     //define the maps to be filled
    map<string, string> unicode_to_char;
    map<string, string> char_to_unicode;
    map<string, string> sjis_to_char;
    map<string, string> char_to_sjis;
    map<string, string> char_to_ascii;
    map<string, string> ascii_to_char;

    while (getline(fullMap_File, line2)) {
        istringstream ss(line2); //Input stream class to operate on strings

        getline(ss, charVal, '\t'); //get character separated by tab
        getline(ss, sjis2, '\t'); //get unicode separated by tab
        getline(ss, unicode2, '\t'); //get shift-JIS separated by tab
        getline(ss, extrachars2, '\n'); //store extra info to be discarded

        //fill in map
        if(charVal == "\\"){//handle the Half-Yen sign case
            char_to_unicode[yenSign] = "U+00A5";
            unicode_to_char["U+00A5"] = yenSign;
            sjis_to_char["0x5C"] = "\\";
            char_to_sjis["¥"] = "0x5C";

            if(asciiCounter2 <=127){//map the first 0xFF characters normally
                char_to_ascii[yenSign] = "92";
                ascii_to_char["92"] = yenSign;
            }
            else{ //set the rest of the characters to 0
                ascii_to_char[to_string(asciiCounter2)] = "0";
                char_to_ascii[charVal] = "0";
            }  
        }
        else{
            unicode_to_char[unicode2] = charVal;
            char_to_unicode[charVal] = unicode2;
            sjis_to_char[sjis2] = charVal;
            char_to_sjis[charVal] = sjis2;
            
            if(asciiCounter2 <=127){//map the first 0xFF characters normally
                char_to_ascii[charVal] = to_string(asciiCounter2);
                ascii_to_char[to_string(asciiCounter2)] = charVal;
            }
            else{ //set the rest of the characters to 0
                ascii_to_char[to_string(asciiCounter2)] = "0";
                char_to_ascii[charVal] = "0";
            }  
        }
        asciiCounter2++;
        
    }

    asciiCounter2 = 0;
    fullMap_File.close(); //close the file

    //Optional: uncomment the following code to see the maps
    /*int counter = 0;
    int counter2 = 0;

    for (map<string,string>::iterator it = unicode_to_char.begin(); it!=unicode_to_char.end(); ++it){
        if (counter==40) break;
        cout << it->first <<" => " << it->second << '\n';
        counter++;
    }

    for (map<string,string>::iterator it = char_to_unicode.begin(); it!=char_to_unicode.end(); ++it){
        if (counter2==40) break;
        cout << it->first <<" => " << it->second << '\n';
        counter2++;
    }*/

    //===================================SJIS & UNICODE MAP==========================================
    string line;
    fstream shiftJIS_File;
    shiftJIS_File.open("../tools/transcoders/mapshiftjis2004.txt"); //Input file to create the map

    //check that the file can be opened
    if (!shiftJIS_File.is_open()) {
        cout << "Unable to open file mapshiftjis2004.txt" << endl << endl;
        return -1;
    }

    getline(shiftJIS_File, line); //get the first line
    //cout << "File format: " << line << endl << endl;

    //Declare variables used in the main function
    string unicode, sjis, extrachars;
    int asciiCounter = 0;

    //define the maps to be filled
    map<string, string> unicode_to_sjisMap;
    map<string, string> sjis_to_unicodeMap;
    map<string, string> ascii_to_sjisMap;
    map<string, string> sjis_to_asciiMap;
    
    while (getline(shiftJIS_File, line)) {
        istringstream ss(line); //Input stream class to operate on strings

        getline(ss, sjis, '\t'); //get unicode separated by comma
        getline(ss, unicode, '\t'); //get shift-JIS separated by comma
        getline(ss, extrachars, '\n'); //store extra info to be discarded

        //fill the sjis, unicode, and the ascii maps
        unicode_to_sjisMap[unicode] = sjis;
        sjis_to_unicodeMap[sjis] = unicode;

        if(asciiCounter <=127){//map the first 0xFF characters normally
            ascii_to_sjisMap[to_string(asciiCounter)] = sjis;
            sjis_to_asciiMap[sjis] = to_string(asciiCounter);
        }
        else if(asciiCounter <=160){ //set the rest of the characters to 0
          ascii_to_sjisMap[to_string(asciiCounter)] = "0";
          sjis_to_asciiMap[sjis] = "0";
        }
        asciiCounter++;
    }

    asciiCounter = 0;
    shiftJIS_File.close();

    //provide the according output based on chosen option
    if(SJISerrorOption2 == InEncoding){
        fromEncoding(sjis_to_unicodeMap, char_to_sjis, inputFile);
    }
    else if(SJISerrorOption2 == OutEncoding){
        toEncoding(unicode_to_sjisMap, char_to_unicode, inputFile);
    }
    else if(SJISerrorOption2 == fromASCIItoSJIS){
        asciiToSJIS(ascii_to_sjisMap, char_to_ascii, inputFile);
    }
    else if(SJISerrorOption2 == fromSJIStoASCII){
        sjisToASCII(sjis_to_asciiMap, char_to_sjis, inputFile);
    }
    else if(SJISerrorOption2 == Usage){
        u_usage();
    }
    else if(SJISerrorOption2 == VersionInfo){
        v_version();
    }
    else if(SJISerrorOption2 == Verbose){
        v_verbose(char_to_unicode, unicode_to_sjisMap, sjis_to_char, inputFile);
    }

    return 0;
}

/*Function to convert Unicode characters to UTF-8 characters 
Input: Map from Unicode to UTF-16 and an input file
Output: None. An output file is created with the Unicode characters corresponding 
        Unicode encoding
*/
void fromUTF16 (map<string, string> UTF16_to_Unicode, string inputFile){
    fstream inFile1;
    string inData;

    inFile1.open(inputFile.c_str());

    if (!inFile1) {
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    while (!inFile1.eof()) {
        inFile1 >> inData;
        cout << UTF16_to_Unicode.find(inData)->second << " "; //obtain the Shift-JIS equivalent of the character

        if (!inFile1.eof()) {
            cout << "feff0020 ";
        }
    }
    cout << endl;
}

/*Function to convert Unicode characters to UTF-8 characters 
Input: Map from Unicode to UTF-8 and an input file
Output: None. An output file is created with the Unicode characters corresponding 
        Unicode encoding
*/
void fromUniToUTF8EncodingFunc(map<string, string> unicode_to_utf8, string inputFile){
    fstream inFile1;
    string inData;

    string utf8Data;

    inFile1.open(inputFile.c_str());

    if(!inFile1){
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    while (!inFile1.eof()){
        inFile1 >> inData;
        cout << unicode_to_utf8.find(inData)->second << " "; //obtain the Shift-JIS equivalent of the character

        if(!inFile1.eof()){
            cout << "20 ";
        }
    }
    cout << endl; 

    inFile1.close();
}

/*Function to convert Shift-JIS characters to Unicode characters 
Input: Map from Shift-JIS to Unicode and an input file
Output: None. An output file is created with the Shift-JIS characters corresponding 
        Unicode encoding
*/
void fromEncoding(map<string, string> sjis_to_unicodeMap, map<string, string> char_to_sjis, string inputFile){
    fstream inFile1;
    string inData;

    string sjisData;

    inFile1.open(inputFile.c_str());

    if(!inFile1){
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    while (!inFile1.eof()){
        inFile1 >> inData;
        sjisData = char_to_sjis.find(inData)->second; //obtain the Shift-JIS equivalent of the character
        cout << sjis_to_unicodeMap.find(sjisData)->second << " "; 

        if(!inFile1.eof()){
            cout << "U+0020 ";
        }
    }
    cout << endl; 

    inFile1.close();
}

/*Function to convert Unicode characters to Shift-JIS characters 
Input: Map from unicode to Shift-JIS and an input file
Output: None. An output file is created with the Unicode characters corresponding 
        Shift-JIS encoding
*/
void toEncoding(map<string, string> unicode_to_sjisMap, map<string, string> char_to_unicode, string inputFile){
    fstream inFile1;
    string inData;

    string unicodeData;

    inFile1.open(inputFile.c_str());

    if(!inFile1){
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    while (!inFile1.eof()){
        inFile1 >> inData;
        unicodeData = char_to_unicode.find(inData)->second; //obtain the unicode equivalent of the character
        cout << unicode_to_sjisMap.find(unicodeData)->second << " "; 
    }

    if(!inFile1.eof()){
        cout  << "0x20 ";
    }
    cout << endl; 

    inFile1.close();
}

/*Function to convert ASCII characters to Shift-JIS characters 
Input: Map from ASCII to Shift-JIS
Output: None. An output file is created with the ASCII characters corresponding 
        Shift-JIS encoding
*/
void asciiToSJIS(map<string, string> ascii_to_sjisMap, map<string, string> char_to_ascii, string inputFile){
    fstream inFile;
    string inData;

    string asciiData;

    inFile.open(inputFile.c_str());

    if(!inFile){
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    while (!inFile.eof()){
        inFile >> inData;
        asciiData = char_to_ascii.find(inData)->second; //obtain the unicode equivalent of the character
        cout << ascii_to_sjisMap.find(asciiData)->second << " ";  
    }
    if(!inFile.eof()){
        cout  << "0x20 ";
    }
    cout << endl; 

    inFile.close();
}

/*Function to convert Shift-JIS characters to ASCII characters 
Input: Map from Shift-JIS to ASCII
Output: None. An output file is created with the Shift-JIS characters corresponding 
        ASCII encoding
*/
void sjisToASCII(map<string, string> sjis_to_asciiMap, map<string, string> char_to_sjis, string inputFile){
    fstream inFile;
    string inData;

    string sjisData;

    inFile.open(inputFile.c_str());

    if(!inFile){
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    while (!inFile.eof()){
        inFile >> inData;
        sjisData = char_to_sjis.find(inData)->second; //obtain the Shift-JIS equivalent of the character
        cout << sjis_to_asciiMap.find(sjisData)->second << " "; 
    }
    if(!inFile.eof()){
        cout  << "20 ";
    }
    cout << endl; 

    inFile.close();
}

/*Function to print a short usage summary and exit.
Input: fileName (string)
Output: None. An output message is printed to the console.
*/
void u_usage(){
    printf("Usage: bin/shiftSJIS [-lcs?V] [-f NAME] [-t NAME] [-o FILE] [--from-code=NAME]\n[--to-code=NAME] [--list] [--output=FILE] [--silent] [--verbose]\n[--help] [-u] [--version] [FILE...]\n");
}

/*Function to print a short version summary and exit.
Input: fileName (string)
Output: None. An output message is printed to the console.
*/
void v_version(){
    printf("Sjift JIS (Ubuntu) 2013\nCopyright (C) 2021 Free Software Foundation, Inc.\nThis is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\nWritten by Ulrich Drepper.\n");
}

/*Function to output the actual characters after the encoding has been performed.
Input: fileName (string)
Output: None. An output message is printed to the console.
*/
void v_verbose(map<string, string> char_to_unicode, map<string, string> unicode_to_sjisMap, map<string, string> sjis_to_char, string inputFile){
    fstream inFile;
    string inData;

    string unicodeData;
    string sjisData;

    inFile.open(inputFile.c_str());

    if(!inFile){
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    cout << inputFile.c_str() << ": "; 
    while (!inFile.eof()){
        inFile >> inData;
        unicodeData = char_to_unicode.find(inData)->second; //obtain the unicode equivalent of the character
        sjisData = unicode_to_sjisMap.find(unicodeData)->second; //obtain the Shift-JIS equivalent of the unicode character
        cout << sjis_to_char.find(sjisData)->second << " ";  //obtain the character equivalent of the Shift-JIS encoding
    }
    cout << endl;

    inFile.close();
}
