#include "kernel/types.h"
#include "kernel/syscall.h"
#include "user/user.h"
#include "kernel/fcntl.h"

void run_for(int ticks) {
  int t0 = uptime();
  while (uptime() - t0 < ticks) { }
}

void sleep1(char *s) {
  printf("%s yielding\n", s);
  sleep(1);
}

void print_performance(struct perf *perf) {
  printf("creation time:    %d\n", perf->ctime);
  printf("termination time: %d\n", perf->ttime);
  printf("running time:     %d\n", perf->rutime);
  printf("runnable time:    %d\n", perf->retime);
  printf("sleeping time:    %d\n", perf->stime);
  printf("burst time:       %d\n", perf->average_bursttime);
}

void print_wait_stat() {
  int status;
  struct perf perf;
  int pid = wait_stat(&status, &perf);
  printf("child %d exited with status %d\n", pid, status);
  print_performance(&perf);
}

void test_wait_stat_task(void) {
  int status;
  int ccount = 20;

  sleep(10);
  for (int i = 0; i < ccount; i++) {
    if (fork() == 0) {
      run_for(2);
      exit(0);
    }
  }
  for (int i = 0; i < ccount; i++) {
    wait(&status);
  }
  run_for(2);
  printf("child (%d) exiting\n", getpid());
  exit(7);
}

void srt_child0() {
  printf("0 running\n");
  run_for(4);
  sleep1("0");
  printf("0 running\n");
  run_for(8);
  sleep1("0");
  printf("0 running\n");
  run_for(7);
  sleep1("0");
}
void srt_child1() {
  printf("1 running\n");
  run_for(6);
  sleep1("1");
}
void srt_child2() {
  printf("2 running\n");
  run_for(6);
  sleep1("2");
  printf("2 running\n");
  run_for(3);
  sleep1("2");
}
void srt_child3() {
  printf("3 running\n");
  run_for(2);
  sleep1("3");
}
void test_srt(void) {
  void (*tasks[])(void) = {
    &srt_child0,
    &srt_child1,
    &srt_child2,
    &srt_child3,
  };
  int pids[sizeof(tasks)/sizeof(void*)];
  struct perf perfs[sizeof(tasks)/sizeof(void*)];
  int len = sizeof(tasks)/sizeof(void*);
  for (int i = 0; i < len; i++) {
    if ((pids[i] = fork()) == 0) {
      tasks[i]();
      exit(0);
    }
  }
  for (int i = 0; i < len; i++) {
    int status;
    struct perf perf;
    int pid = wait_stat(&status, &perf);
    int j = pid - pids[0];
    printf("i = %d exited\n", j);
    perfs[j] = perf;
  }
  printf("\n");
  for (int i = 0; i < len; i++) {
    printf("i = %d, pid = %d\n", i, pids[i]);
    print_performance(&perfs[i]);
    printf("\n");
  }

  printf("SRT test stats:\n");
}

void test_bursttime(void) {
  run_for(18);
}

void test_set_priority() {
  #ifdef SCHED_CFSD
  int pid = fork();
  if (pid == 0) {
    if (set_priority(6) >= 0) {
      printf("set priority: call didn't fail on 6.\n");
      exit(7);
    }
    for (int i = 5; i > 0; i--) {
      if (set_priority(i) < 0) {
        printf("set priority: call failed on %d.\n", i);
        exit(i + 1);
      }
    }
    if (set_priority(0) >= 0) {
      printf("set priority: call didn't fail on 0.\n");
      exit(1);
    }
    run_for(4);
    sleep(2);
    printf("child exiting...\n");
    exit(0);
  }
  else {
    wait(0);
  }
  #endif
}

void measure_performance(void (*child_task)(void)) {
  int pid = fork();
  if (pid == 0) {
    child_task();
    exit(0);
  }
  else {
    print_wait_stat();
  }
}

void test_uptime() {
  int t0 = uptime();
  sleep(100);
  int t1 = uptime();
  int dt = t1 - t0;
  printf("%d, %d, %d\n", t0, t1, dt);
}

void test_trace() {
  char *str = 0;
  trace((1 << SYS_getpid) | (1 << SYS_fork) | (1 << SYS_sbrk), getpid());

  if(fork() == 0){
    trace((1 << SYS_sbrk), getpid());
    fprintf(2, "child process id: %d\n", getpid());
    str = malloc(1024);
  } else {
    wait(0);
    fprintf(2, "parent process id: %d\n", getpid());
    str = malloc(1024);
    memcpy(str, "hello", 6);
  }
}

void main(int argc, char *argv[]) {
  // measure_performance(&test_srt);
  test_set_priority();
  exit(0);
}
