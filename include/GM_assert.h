#ifndef GM_DEBUG_OFF
#define ASSERT(expr) \
    if (expr) \
        {} \
    else{ printf("Assertion failed in %s at line %d\n>>\t%s\t<<\n", __FILE__, __LINE__, #expr); exit(0);}
#endif