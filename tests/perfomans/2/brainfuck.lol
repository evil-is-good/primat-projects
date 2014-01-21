#define lol ++(*p);
#define olo --(*p);
#define oll ++p;
#define llo --p;
#define loo while (*p){
#define ool }
#define ooo putchar(*p);
#define lll *p = getchar();
#define start \
    int main () \
    { \
      unsigned char * p; \
      unsigned char * first_byte; \
      p = malloc(256); \
      first_byte = 0;
#define end \
      free(first_byte); \
      return 0; \
    }
