#include "cppCode.h"

static int currentInt = 0;

double multiply(double n1, double n2) {
    return n1*n2;
}
double add(double n1, double n2) {
    return n1+n2;
}
int nextInteger() {
    int _int = currentInt;
    currentInt+=1;
    return _int;
}