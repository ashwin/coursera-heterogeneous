// from http://stackoverflow.com/questions/922492/use-of-clock-getres-newbie-linux-c
// compile with gcc -o test_timers.cpp -lrt
#include <time.h>
#include <stdio.h>

int main( int argc, char** argv )
{
  clockid_t types[] = { CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID, (clockid_t) - 1 };

  struct timespec spec;
  int i = 0;
  for ( i; types[i] != (clockid_t) - 1; i++ )
  {
    if ( clock_getres( types[i], &spec ) != 0 )
    {
      printf( "Timer %d not supported.\n", types[i] );
    }
    else
    {
      printf( "Timer: %d, Seconds: %ld Nanos: %ld\n", i, spec.tv_sec, spec.tv_nsec );
    }
  }
}
