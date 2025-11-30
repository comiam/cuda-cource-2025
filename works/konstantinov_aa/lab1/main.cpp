#include <iostream>
#include <cmath>
using namespace std;

int main(){
    int radius = 5;
    for (int y = radius; y >= -radius; --y){
        for (int x = -radius; x <= radius; ++x){
            if (round(sqrt(x*x + y*y)) == radius) {
                cout << "* ";
            } else {
                cout << " ";
            }
        }
    cout << endl;
    }


}