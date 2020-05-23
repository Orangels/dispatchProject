#include<iostream>
#include <functional>

typedef void(*fun)(int);
void register_callback(fun f) {
    f(42); // a test
}
// ------------------------------------

#include <functional>
#include <iostream>

void foo(const char* ptr, int v, float x) {
    std::cout << ptr << " " << v << " " << x << std::endl;
}

namespace {
    std::function<void(int)> callback;
    extern "C" void wrapper(int i) {
        callback(i);
    }
}

int main() {
    callback = std::bind(&foo, "test", std::placeholders::_1, 3.f);
    register_callback(wrapper); // <-- How to do this?
}