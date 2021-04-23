#include <fstream>
#include <string>
#include <iostream>
#include <stdio.h> 
#include <assert.h>
#include <algorithm>
using namespace std;
int main(int argc, char **argv)
{
    cout << "****************************\n";
    cout << "*    ins.txt -> ins.bin    *\n";
    cout << "****************************\n";
    if (argc != 3) {
        cout << "2 command line argument needed!\n";
        cout << "1st arg: input.txt\n";
        cout << "2nd arg: output.bin\n";
        assert(argc == 3);
    } else {
        cout << "input.txt: " << argv[1] << "\n";
        cout << "output.bin: " << argv[2] << "\n";
    }
    
    // Read text
    ifstream infile(argv[1]);
    int lineCnt = std::count(std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>(), '\n');
    cout << "line count in input.txt: " << lineCnt << "\n";
    unsigned int* a = (unsigned int*)malloc(lineCnt * sizeof(unsigned int));
    string line;
    int i = 0;
    infile.seekg(0, std::ios::beg);
    while(getline(infile, line)){     
      if(i < lineCnt){        
        a[i] = stoull(line.substr(0,32), 0, 2);       
        i++;
      } else {
        break;
      }
    }
    cout << "line readed: " << i << "\n";
    infile.close();
    
    // Write bin
    FILE* pFile;
    cout  << "write to " << argv[2] << "\n";
    pFile = fopen(argv[2], "wb");
    fwrite(a, 1, lineCnt * sizeof(unsigned int), pFile);
    fclose(pFile);
    
    cout << "****************************\n";
    cout << "*         complete!        *\n";
    cout << "****************************\n";
    
    return 0;
}