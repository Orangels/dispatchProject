//#include <iostream>
#include "Hungarian.h"
using namespace std;


extern "C" 
{
    HungarianAlgorithm* init_Hungarian(){
        return new HungarianAlgorithm();
    }
    int* Solve(HungarianAlgorithm* new_HA,float* DistMatrix_flatten,int row,int col){
        vector<int> assignment_tracking_id;
        vector <vector<double> > DistMatrix;
        for(int i=0;i<col;i++){
            vector<double> Matrix_line;
            for(int j=0;j<row;j++){
                Matrix_line.push_back(DistMatrix_flatten[i*row+j]);
            }
            DistMatrix.push_back(Matrix_line);
        }
        new_HA->Solve(DistMatrix, assignment_tracking_id);
        static vector <int> result;
        result.resize(0);
        for(int i =0; i< assignment_tracking_id.size();i++){
            result.push_back(assignment_tracking_id[i]);
        }
        return result.data();
    }
}
