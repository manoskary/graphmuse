// taken from here https://barrgroup.com/embedded-systems/how-to/define-assert-macro

#ifndef GM_DEBUG_OFF
#define ASSERT(expr) \
    if(expr)\
    {}\
    else{\
        printf("Assertion failed in %s at line %d\n>>\t%s\t<<\n", __FILE__, __LINE__, #expr); \
        abort(); \
    }
#endif

#ifdef GM_DEBUG_OFF
#define ASSERT(expr){}
#endif