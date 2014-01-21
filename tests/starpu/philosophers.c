#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

#define NBPHIL 5

enum state {
  THINK  = 0,
  HUNGRY = 1,
  EAT    = 2
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
sem_t          *semphil;
int             philosophes[5];
double          timestamp = 0.0;
FILE           *tracefile;

void  *thread_function(void *arg);
void   eat            (int   time);
void   think          (int   time);
void   take_forks     (int   i);
void   release_forks  (int   i);
void   test           (int   i);
double clockGet       (void);

void *thread_function(void * arg)
{
  int i;
  int *tab = (int *)arg;
  
  for (i = 0; i < 5; i++)
    {
      think(tab[1]);
      take_forks(tab[0]);
      eat(tab[2]);
      release_forks(tab[0]);
    }
  return (void*)NULL;
}

void eat(int time)
{
  sleep(time);
}

void think(int time)
{
  sleep(time);
}

void take_forks(int i)
{
  pthread_mutex_lock(&mutex);
  philosophes[i] = HUNGRY;
  fprintf (tracefile, "10 %lf ST_ThreadState C_Thread%d S_H\n", clockGet(), i);
  fflush(tracefile);
  test(i);
  pthread_mutex_unlock(&mutex);
  sem_wait(&(semphil[i])); 
  fprintf(tracefile, "53 %lf V_Sem C_Thread%d 1.0\n", clockGet(), i);
  fflush(tracefile);
}

void release_forks(int i)
{
  pthread_mutex_lock(&mutex);
  philosophes[i] = THINK;
  fprintf (tracefile, "10 %lf ST_ThreadState C_Thread%d S_T\n", clockGet(), i);
  fflush(tracefile);
  test((i+NBPHIL-1)%NBPHIL);
  test((i+NBPHIL+1)%NBPHIL);
  pthread_mutex_unlock(&mutex);
}

void test(int i)
{
  if ((philosophes[i] == HUNGRY) &&
      (philosophes[(i+NBPHIL-1)%NBPHIL] != EAT) && 
      (philosophes[(i+NBPHIL+1)%NBPHIL] != EAT))
    {
      double time = clockGet();
      philosophes[i] = EAT;
      fprintf(tracefile, "10 %lf ST_ThreadState C_Thread%d S_E\n", time, i);
      fprintf(tracefile, "52 %lf V_Sem C_Thread%d 1.0\n", time, i);
      fflush(tracefile);
      sem_post(&semphil[i]);
    }
}

double clockGet (void)
{
  struct timespec tp;
  
  clock_gettime (CLOCK_REALTIME, &tp);            /* Elapsed time */

  return (((double) tp.tv_sec + (double) tp.tv_nsec * (double)1.0e-9L) - timestamp);
}

int main(int argc, char *argv[])
{
  pthread_t *calltab;
  int       *param;
  int        i;

  tracefile = fopen("philosophers.trace", "w");

  /* Write Header */
  fprintf(tracefile, "%%EventDef PajeDefineContainerType 1\n");
  fprintf(tracefile, "%% Alias string \n");
  fprintf(tracefile, "%% ContainerType string \n");
  fprintf(tracefile, "%% Name string \n");
  fprintf(tracefile, "%%EndEventDef \n");
  fprintf(tracefile, "%%EventDef PajeDefineStateType 3\n");
  fprintf(tracefile, "%% Alias string \n");
  fprintf(tracefile, "%% ContainerType string \n");
  fprintf(tracefile, "%% Name string \n");
  fprintf(tracefile, "%%EndEventDef \n");
  fprintf(tracefile, "%%EventDef PajeDefineEntityValue 6\n");
  fprintf(tracefile, "%% Alias string  \n");
  fprintf(tracefile, "%% EntityType string  \n");
  fprintf(tracefile, "%% Name string  \n");
  fprintf(tracefile, "%% Color color \n");
  fprintf(tracefile, "%%EndEventDef  \n");
  fprintf(tracefile, "%%EventDef PajeCreateContainer 7\n");
  fprintf(tracefile, "%% Time date  \n");
  fprintf(tracefile, "%% Alias string  \n");
  fprintf(tracefile, "%% Type string  \n");
  fprintf(tracefile, "%% Container string  \n");
  fprintf(tracefile, "%% Name string  \n");
  fprintf(tracefile, "%%EndEventDef  \n");
  fprintf(tracefile, "%%EventDef PajeDestroyContainer 8\n");
  fprintf(tracefile, "%% Time date  \n");
  fprintf(tracefile, "%% Name string  \n");
  fprintf(tracefile, "%% Type string  \n");
  fprintf(tracefile, "%%EndEventDef  \n");
  fprintf(tracefile, "%%EventDef PajeSetState 10\n");
  fprintf(tracefile, "%% Time date  \n");
  fprintf(tracefile, "%% Type string  \n");
  fprintf(tracefile, "%% Container string  \n");
  fprintf(tracefile, "%% Value string  \n");
  fprintf(tracefile, "%%EndEventDef \n");
  fprintf(tracefile, "%%EventDef PajeDefineVariableType 50\n");
  fprintf(tracefile, "%% Alias string\n");
  fprintf(tracefile, "%% Name  string\n");
  fprintf(tracefile, "%% ContainerType string \n");
  fprintf(tracefile, "%%EndEventDef \n");
  fprintf(tracefile, "%%EventDef PajeSetVariable 51\n");
  fprintf(tracefile, "%% Time date \n");
  fprintf(tracefile, "%% Type string \n");
  fprintf(tracefile, "%% Container string \n");
  fprintf(tracefile, "%% Value double \n");
  fprintf(tracefile, "%%EndEventDef  \n");
  fprintf(tracefile, "%%EventDef PajeAddVariable 52\n");
  fprintf(tracefile, "%% Time date \n");
  fprintf(tracefile, "%% Type string \n");
  fprintf(tracefile, "%% Container string \n");
  fprintf(tracefile, "%% Value double \n");
  fprintf(tracefile, "%%EndEventDef  \n");
  fprintf(tracefile, "%%EventDef PajeSubVariable 53\n");
  fprintf(tracefile, "%% Time date \n");
  fprintf(tracefile, "%% Type string \n");
  fprintf(tracefile, "%% Container string \n");
  fprintf(tracefile, "%% Value double \n");
  fprintf(tracefile, "%%EndEventDef\n");

  /* Type of container And associates states */
  fprintf(tracefile, "1 CT_Prog   0       'Program'\n");
  fprintf(tracefile, "1 CT_Thread CT_Prog 'Thread'\n");
  fprintf(tracefile, "3 ST_ThreadState CT_Thread 'Thread State'\n");
  fprintf(tracefile, "6 S_T ST_ThreadState 'Think'  '0.500000 0.500000 0.500000'\n");
  fprintf(tracefile, "6 S_H ST_ThreadState 'Hungry' '1.000000 0.000000 0.000000'\n");
  fprintf(tracefile, "6 S_E ST_ThreadState 'Eat'    '0.000000 0.000000 1.000000'\n");

  /* Create semaphore variables */
  fprintf(tracefile, "50 V_Sem Semaphore CT_Thread\n");

  /* Create the programme container */
  fprintf(tracefile, "7 0.000000 C_Prog CT_Prog 0 'Programme'\n");

  semphil = (sem_t *)malloc(NBPHIL*sizeof(sem_t));
  calltab = (pthread_t *)malloc(NBPHIL*sizeof(pthread_t));
  param   = (int *)malloc(3*NBPHIL*sizeof(int));
  timestamp = clockGet();

  for (i=0; i<NBPHIL; i++)
    {
      param[3*i]   = i;
      param[3*i+1] = random()%5  + 1;
      param[3*i+2] = random()%10 + 1;

      /* Create the thread container */
      fprintf(tracefile, "7  0.000000 C_Thread%d CT_Thread C_Prog 'Thread %d'\n", i, i);
      fprintf(tracefile, "51 0.000000 V_Sem C_Thread%d 0.0\n", i);
      sem_init(&(semphil[i]), 0, 0);
      pthread_create(&calltab[i], NULL, thread_function, (void *)(param+i*3));
    }
  
  for (i=0; i<NBPHIL; i++)
    {
      pthread_join(calltab[i],(void**)NULL);
      sem_destroy(&(semphil[i]));
      
      /* Destroy each thread container */
      fprintf(tracefile, "8 %lf C_Thread%d CT_Thread\n", clockGet(), i);
      fflush(tracefile);
    }

  /* Destroy the program container */
  fprintf(tracefile, "8 %lf C_Prog CT_Prog\n", clockGet());
  fclose(tracefile);
  free(semphil);
  free(calltab);
  free(param);

  return EXIT_SUCCESS;
}
