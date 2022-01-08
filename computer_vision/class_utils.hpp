#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include <numeric>

namespace class_utils {

    // returns are by reference to avoid copying lambda functions each time they are accessed
    template<typename F> struct FunctionHolder : public F {

        auto& function() { return static_cast<F&>(*this); }
        auto& function() const { return static_cast<const F&>(*this); }
    };

    template<typename F> struct FunctionHolder<F*> {

        FunctionHolder(F *f) : mfunc{f} {}

        auto& function() { return mfunc; }
        auto& function() const { return mfunc; }

        private:
            F *mfunc;
    };

    // take advantage of pair functor
    template<typename F> struct fmap : public FunctionHolder<F> {

        template<typename U>
        explicit fmap(U &&u) : FunctionHolder<F>{std::forward<U>(u)} {}

        // row and column manipulation
        template<typename T> auto operator() (std::pair<T, T> input, int param1, int param2) const {

            return std::make_pair<T, T>(this->function()(std::get<0>(input), param1, param2), 
                this->function()(std::get<1>(input), param1, param2));
        }

        // overload for transposition
        template<typename T> auto operator() (std::pair<T, T> input) const {

            return std::make_pair<T, T>(this->function()(std::get<0>(input)), 
                this->function()(std::get<1>(input)));
        }
    };

    template<typename F> fmap(F f) -> fmap<F>;
    
}