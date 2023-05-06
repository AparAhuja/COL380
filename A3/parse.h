#include <getopt.h>

struct Args{
    int taskid = 1, startk, endk, p = -1;
    string inputpath, headerpath, outputpath;
    bool verbose = 0;

    void parseInput(int argc, char **argv){
        for(int i = 0; i < argc; i++) {
            static struct option long_options[] = {
                {"taskid", required_argument, 0, 't'},
                {"inputpath", required_argument, 0, 'i'},
                {"headerpath", required_argument, 0, 'h'},
                {"outputpath", required_argument, 0, 'o'},
                {"verbose", required_argument, 0, 'v'},
                {"startk", required_argument, 0, 's'},
                {"endk", required_argument, 0, 'e'},
                {"p", required_argument, 0, 'p'}
            };
            int option_index = 0;
            int c = getopt_long(argc, argv, "t:i:h:o:v:s:e:p:", long_options, &option_index);
            
            switch (c) {
                case 't':
                    taskid = stoi(optarg);
                    break;
                case 'i':
                    inputpath = optarg;
                    break;
                case 'h':
                    headerpath = optarg;
                    break;
                case 'o':
                    outputpath = optarg;
                    break;
                case 'v':
                    verbose = stoi(optarg);
                    break;
                case 's':
                    startk = stoi(optarg);
                    break;
                case 'e':
                    endk = stoi(optarg);
                    break;
                case 'p':
                    p = stoi(optarg);
                    break;
            }
        }
    }

    void printInput(){
        // Print the values of the variables
        cout << "taskid: " << taskid << endl;
        cout << "startk: " << startk << endl;
        cout << "endk: " << endk << endl;
        cout << "p: " << p << endl;
        cout << "inputpath: " << inputpath << endl;
        cout << "headerpath: " << headerpath << endl;
        cout << "outputpath: " << outputpath << endl;
        cout << "verbose: " << verbose << endl;
    }
};