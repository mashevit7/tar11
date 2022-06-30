#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#ifdef SCHED_FCFS
  #include "proc_array_queue.h"
#endif

struct cpu cpus[NCPU];

struct proc proc[NPROC];

struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

extern void forkret(void);
static void freeproc(struct proc *p);

extern char trampoline[]; // trampoline.S

// helps ensure that wakeups of wait()ing
// parents are not lost. helps obey the
// memory model when using p->parent.
// must be acquired before any p->lock.
struct spinlock wait_lock;

#ifdef SCHED_FCFS
  struct proc_array_queue ready_queue;

  static void
  insert_to_ready_queue(struct proc *proc, int pid, char *from)
  {
    if (!proc_array_queue_enqueue(&ready_queue, proc)) {
      panic("insert to ready queue - full");
    }
  }
#endif

// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void
proc_mapstacks(pagetable_t kpgtbl) {
  struct proc *p;
  
  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}

// initialize the proc table at boot time.
void
procinit(void)
{
  struct proc *p;
  
  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  for(p = proc; p < &proc[NPROC]; p++) {
      initlock(&p->lock, "proc");
      p->kstack = KSTACK((int) (p - proc));
  }

  #ifdef SCHED_FCFS
    proc_array_queue_init(&ready_queue, "readyQueue");
  #endif
}

// Must be called with interrupts disabled,
// to prevent race with process being moved
// to a different CPU.
int
cpuid()
{
  int id = r_tp();
  return id;
}

// Return this CPU's cpu struct.
// Interrupts must be disabled.
struct cpu*
mycpu(void) {
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}

// Return the current struct proc *, or zero if none.
struct proc*
myproc(void) {
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}

int
allocpid() {
  int pid;
  
  acquire(&pid_lock);
  pid = nextpid;
  nextpid = nextpid + 1;
  release(&pid_lock);

  return pid;
}

// Look in the process table for an UNUSED proc.
// If found, initialize state required to run in the kernel,
// and return with p->lock held.
// If there are no free procs, or a memory allocation fails, return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;
  p->perf_stats = (struct perf){
    .ctime = uptime(),
    .ttime = -1,
    .stime = 0,
    .retime = 0,
    .rutime = 0,
    .average_bursttime = QUANTUM*BURSTTIME_PRECESION,
  };

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  return p;
}

struct proc*
find_proc(int pid){
  struct proc *p = 0;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->pid == pid) {
      release(&p->lock);
      return p;
    } else {
      release(&p->lock);
    }
  }

  return 0;
}

// free a proc structure and the data hanging from it,
// including user pages.
// p->lock must be held.
static void
freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
  p->trace_mask = 0;
  #ifdef SCHED_CFSD
  p->priority = 2;
  #ifdef SCHED_CFSD_ACCUM_STATS
  p->perf_stats_parent = (struct perf){
    .ctime = uptime(),
    .ttime = -1,
    .stime = 0,
    .retime = 0,
    .rutime = 0,
    .average_bursttime = QUANTUM*BURSTTIME_PRECESION,
  };
  #endif
  #endif
}

// Create a user page table for a given process,
// with no user memory, but with trampoline pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe just below TRAMPOLINE, for trampoline.S.
  if(mappages(pagetable, TRAPFRAME, PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}

// Free a process's page table, and free the
// physical memory it refers to.
void
proc_freepagetable(pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME, 1, 0);
  uvmfree(pagetable, sz);
}

// a user program that calls exec("/init")
// od -t xC initcode
uchar initcode[] = {
  0x17, 0x05, 0x00, 0x00, 0x13, 0x05, 0x45, 0x02,
  0x97, 0x05, 0x00, 0x00, 0x93, 0x85, 0x35, 0x02,
  0x93, 0x08, 0x70, 0x00, 0x73, 0x00, 0x00, 0x00,
  0x93, 0x08, 0x20, 0x00, 0x73, 0x00, 0x00, 0x00,
  0xef, 0xf0, 0x9f, 0xff, 0x2f, 0x69, 0x6e, 0x69,
  0x74, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

// Set up first user process.
void
userinit(void)
{
  struct proc *p;
  #ifdef SCHED_FCFS
  int pid = 0;
  #endif

  p = allocproc();
  initproc = p;
  
  // allocate one user page and copy init's instructions
  // and data into it.
  uvminit(p->pagetable, initcode, sizeof(initcode));
  p->sz = PGSIZE;

  // prepare for the very first "return" from kernel to user.
  p->trapframe->epc = 0;      // user program counter
  p->trapframe->sp = PGSIZE;  // user stack pointer

  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");

  p->state = RUNNABLE;

  #ifdef SCHED_CFSD
  #ifdef SCHED_CFSD_ACCUM_STATS
  p->perf_stats_parent = p->perf_stats;
  #endif
  p->priority = 2;
  #endif
  #ifdef SCHED_FCFS
  pid = p->pid;
  #endif
  release(&p->lock);

  #ifdef SCHED_FCFS
    insert_to_ready_queue(p, pid, "userinit");
  #endif
}

// Grow or shrink user memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n)
{
  uint sz;
  struct proc *p = myproc();

  sz = p->sz;
  if(n > 0){
    if((sz = uvmalloc(p->pagetable, sz, sz + n)) == 0) {
      return -1;
    }
  } else if(n < 0){
    sz = uvmdealloc(p->pagetable, sz, sz + n);
  }
  p->sz = sz;
  return 0;
}

#if SCHED_CFSD && SCHED_CFSD_ACCUM_STATS
void
perf_copy_from_parent(struct proc *p, struct proc *np)
{
  // this is fine not having acquired the lock of the parent here since:
  // * perf_stats_parent is only updated in here, allocproc and freeproc.
  // * stime, retime, rutime of are the only fields that actually matter.
  // * the maybe change in the parent stats isn't that critical (we may be 1 off)
  struct perf *np_perf_parent = &np->perf_stats_parent;
  struct perf *p_perf = &p->perf_stats;
  struct perf *p_perf_parent = &p->perf_stats_parent;
  np_perf_parent->ctime = p_perf->ctime;
  np_perf_parent->ttime = p_perf->ttime;
  np_perf_parent->stime = p_perf->stime + p_perf_parent->stime;
  np_perf_parent->retime = p_perf->retime + p_perf_parent->retime;
  np_perf_parent->rutime = p_perf->rutime + p_perf_parent->rutime;
  np_perf_parent->average_bursttime = p->perf_stats.average_bursttime;
}
#endif

// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int
fork(void)
{
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // Allocate process.
  if((np = allocproc()) == 0){
    return -1;
  }

  // Copy user memory from parent to child.
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  //TODO maybe we need to acuqire the parent
  np->trace_mask = p->trace_mask;
  #ifdef SCHED_CFSD
  #ifdef SCHED_CFSD_ACCUM_STATS
  perf_copy_from_parent(p, np);
  #endif
  np->priority = p->priority;
  #endif

  // copy saved user registers.
  *(np->trapframe) = *(p->trapframe);

  // Cause fork to return 0 in the child.
  np->trapframe->a0 = 0;

  // increment reference counts on open file descriptors.
  for(i = 0; i < NOFILE; i++)
    if(p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  safestrcpy(np->name, p->name, sizeof(p->name));

  pid = np->pid;
  release(&np->lock);

  acquire(&wait_lock);
  np->parent = p;
  release(&wait_lock);

  acquire(&np->lock);
  np->state = RUNNABLE;
  release(&np->lock);

  #ifdef SCHED_FCFS
    insert_to_ready_queue(np, pid, "fork");
  #endif

  return pid;
}

// Pass p's abandoned children to init.
// Caller must hold wait_lock.
void
reparent(struct proc *p)
{
  struct proc *pp;

  for(pp = proc; pp < &proc[NPROC]; pp++){
    if(pp->parent == p){
      pp->parent = initproc;
      wakeup(initproc);
    }
  }
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait().
void
exit(int status)
{
  struct proc *p = myproc();
  // int pid = 0;

  if(p == initproc)
    panic("init exiting");

  // Close all open files.
  for(int fd = 0; fd < NOFILE; fd++){
    if(p->ofile[fd]){
      struct file *f = p->ofile[fd];
      fileclose(f);
      p->ofile[fd] = 0;
    }
  }

  begin_op();
  iput(p->cwd);
  end_op();
  p->cwd = 0;

  acquire(&wait_lock);

  // Give any children to init.
  reparent(p);

  // Parent might be sleeping in wait().
  wakeup(p->parent);
  
  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;
  p->perf_stats.ttime = uptime();
  // pid = p->pid;

  release(&wait_lock);

  // printf("%d: %d exited", cpuid(), pid);

  // Jump into the scheduler, never to return.
  sched();
  panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
wait(uint64 addr, uint64 performance)
{
  struct proc *np;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for(;;){
    // Scan through table looking for exited children.
    havekids = 0;
    for(np = proc; np < &proc[NPROC]; np++){
      if(np->parent == p){
        // make sure the child isn't still in exit() or swtch().
        acquire(&np->lock);

        havekids = 1;
        if(np->state == ZOMBIE){
          // Found one.
          pid = np->pid;
          if(addr != 0 && copyout(p->pagetable, addr, (char *)&np->xstate,
                                  sizeof(np->xstate)) < 0) {
            release(&np->lock);
            release(&wait_lock);
            return -1;
          }
          if (performance && copyout(p->pagetable, (uint64)performance, (char*)&np->perf_stats, sizeof(np->perf_stats)) < 0) {
            release(&np->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(np);
          release(&np->lock);
          release(&wait_lock);
          return pid;
        }
        release(&np->lock);
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || p->killed){
      release(&wait_lock);
      return -1;
    }
    
    // Wait for a child to exit.
    sleep(p, &wait_lock);  //DOC: wait-sleep
  }
}

static void
calc_burst_time(struct proc *p, int actual_bursttime)
{
  int actual_bursttime_weight = ALPHA*actual_bursttime;
  int prev_bursttime = p->perf_stats.average_bursttime;
  int prev_bursttime_weight = ((BURSTTIME_PRECESION - ALPHA)*prev_bursttime) / BURSTTIME_PRECESION;
  p->perf_stats.average_bursttime = actual_bursttime_weight + prev_bursttime_weight;
}

// Runs the process up to a QUANTOM of ticks
// or until he gives up the running time.
// Requires the lock of the process to be acquired before calling
static void
run_proc_swtch(struct proc *p, struct cpu *c) {
  // Switch to chosen process.  It is the process's job
  // to release its lock and then reacquire it
  // before jumping back to us.
  swtch(&c->context, &p->context);
}

static void
run_proc_core(struct proc *p)
{
  struct cpu *c = mycpu();
  uint32 tick_start;
  
  tick_start = uptime();
  p->state = RUNNING;
  c->proc = p;
  run_proc_swtch(p, c);
  calc_burst_time(p, uptime() - tick_start);
  
  // Process is done running for now.
  // It should have changed its p->state before coming back.
  c->proc = 0;
}

#ifndef SCHED_FCFS
static void
run_proc_lock_if_runnable(struct proc *p)
{
  acquire(&p->lock);
  if(p->state == RUNNABLE) {
    run_proc_core(p);
  }
  release(&p->lock);
}
#endif

#ifdef SCHED_FCFS
static void
run_nullable_proc_lock(struct proc *p)
{
  if (p) {
    acquire(&p->lock);
    run_proc_core(p);
    release(&p->lock);
  }
}
#endif

#if SCHED_SRT || SCHED_CFSD
static void
run_nullable_proc_lock_if_runnable(struct proc *p)
{
  if (p) {
    run_proc_lock_if_runnable(p);
  }
}

static struct proc*
find_min_proc(int (*get_value)(struct proc*), int (*compare)(int min_value, int p_value))
{
  struct proc *p_to_run = 0;
  int min_value = -1;
  int p_value;
  for (struct proc *p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == RUNNABLE) {
      p_value = get_value(p);
      if (!p_to_run || compare(min_value, p_value)) {
        min_value = p_value;
        p_to_run = p;
      }
    }
    release(&p->lock);
  }

  return p_to_run;
}
#endif

#ifdef SCHED_DEFAULT
void scheduler_round_robin(void) __attribute__((noreturn));;
void
scheduler_round_robin(void)
{
  struct proc *p;
  mycpu()->proc = 0;
  for(;;) {
    // Avoid deadlock by ensuring that devices can interrupt.
    intr_on();

    for(p = proc; p < &proc[NPROC]; p++) {
      run_proc_lock_if_runnable(p);
    }
  }
}
#endif

#ifdef SCHED_FCFS
void scheduler_fcfs(void) __attribute__((noreturn));;
void
scheduler_fcfs(void)
{
  struct proc *p;
  struct cpu *c = mycpu();

  c->proc = 0;
  for(;;){
    // Avoid deadlock by ensuring that devices can interrupt.
    intr_on();

    p = proc_array_queue_dequeue(&ready_queue);
    run_nullable_proc_lock(p);
  }
}
#endif

#ifdef SCHED_SRT
int get_value_srt(struct proc *p)
{
  return p->perf_stats.average_bursttime;
}
int compare_procs_srt(int min_value, int value)
{
  return min_value > value;
}

void scheduler_srt(void) __attribute__((noreturn));;
void
scheduler_srt(void)
{
  struct proc *p;
  for (;;) {
    // Avoid deadlock by ensuring that devices can interrupt.
    intr_on();

    p = find_min_proc(&get_value_srt, &compare_procs_srt);
    run_nullable_proc_lock_if_runnable(p);
  }
}
#endif

#ifdef SCHED_CFSD
static uint32 decay_factors[] = { 1, 3, 5, 7, 25 };

uint32
calc_runtime_ratio(struct proc *p)
{
  #ifdef SCHED_CFSD_ACCUM_STATS
  int stime = p->perf_stats.stime + p->perf_stats_parent.stime;
  int rutime = p->perf_stats.rutime + p->perf_stats_parent.rutime;
  #else
  int stime = p->perf_stats.stime;
  int rutime = p->perf_stats.rutime;
  #endif

  uint32 denominator = rutime + stime;
  uint32 rutime_weighted;
  uint32 rtratio;
  if (denominator == 0) {
    rtratio = 0;
  }
  else {
    rutime_weighted = rutime * decay_factors[p->priority];
    rtratio = rutime_weighted / denominator;
  }

  return rtratio;
}

int get_value_cfsd(struct proc *p)
{
  uint32 rtratio = calc_runtime_ratio(p);
  return *(int*)(&rtratio);
}
int compare_procs_cfsd(int min_value, int value)
{
  return (*(uint32*)(&min_value)) > (*(uint32*)(&value));
}

void scheduler_cfsd(void) __attribute__((noreturn));;
void
scheduler_cfsd(void)
{
  struct proc *p;
  for (;;) {
    // Avoid deadlock by ensuring that devices can interrupt.
    intr_on();

    p = find_min_proc(&get_value_cfsd, &compare_procs_cfsd);
    run_nullable_proc_lock_if_runnable(p);
  }
}
#endif

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run.
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void
scheduler(void)
{
#ifdef SCHED_DEFAULT
  scheduler_round_robin();
#elif SCHED_FCFS
  scheduler_fcfs();
#elif SCHED_SRT
  scheduler_srt();
#elif SCHED_CFSD
  scheduler_cfsd();
#else
  panic("scheduler no policy");
#endif
}

// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched running");
  if(intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
  struct cpu *c = mycpu();
  c->ticks_running++;
  if (c->ticks_running % QUANTUM != 0) {
    return;
  }
  c->ticks_running = 0;

  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  sched();
  release(&p->lock);
}

// A fork child's very first scheduling by scheduler()
// will swtch to forkret.
void
forkret(void)
{
  static int first = 1;

  // Still holding p->lock from scheduler.
  release(&myproc()->lock);

  if (first) {
    // File system initialization must be run in the context of a
    // regular process (e.g., because it calls sleep), and thus cannot
    // be run from main().
    first = 0;
    fsinit(ROOTDEV);
  }

  usertrapret();
}

// Atomically release lock and sleep on chan.
// Reacquires lock when awakened.
void
sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();
  
  // Must acquire p->lock in order to
  // change p->state and then call sched.
  // Once we hold p->lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup locks p->lock),
  // so it's okay to release lk.

  acquire(&p->lock);  //DOC: sleeplock1
  release(lk);

  // Go to sleep.
  p->chan = chan;
  p->state = SLEEPING;

  sched();

  // Tidy up.
  p->chan = 0;

  // Reacquire original lock.
  release(&p->lock);
  acquire(lk);
}

// Wake up all processes sleeping on chan.
// Must be called without any p->lock.
void
wakeup(void *chan)
{
  struct proc *p;
  #ifdef SCHED_FCFS
  int add_to_ready_queue = 0;
  int pid = 0;
  #endif

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()){
      #ifdef SCHED_FCFS
      add_to_ready_queue = 0;
      #endif
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;
        #ifdef SCHED_FCFS
        add_to_ready_queue = 1;
        pid = p->pid;
        #endif
      }
      release(&p->lock);
      #ifdef SCHED_FCFS
      if (add_to_ready_queue) {
        insert_to_ready_queue(p, pid, "wakeup");
      }
      #endif
    }
  }
}

// Kill the process with the given pid.
// The victim won't exit until it tries to return
// to user space (see usertrap() in trap.c).
int
kill(int pid)
{
  struct proc *p;
  #ifdef SCHED_FCFS
  int add_to_ready_queue = 0;
  #endif

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;
      if(p->state == SLEEPING){
        // Wake process from sleep().
        p->state = RUNNABLE;
        #ifdef SCHED_FCFS
        add_to_ready_queue = 1;
        #endif
      }
      release(&p->lock);
      #ifdef SCHED_FCFS
      if (add_to_ready_queue) {
        insert_to_ready_queue(p, pid, "kill");
      }
      #endif

      return 0;
    }
    release(&p->lock);
  }
  return -1;
}

// Copy to either a user address, or kernel address,
// depending on usr_dst.
// Returns 0 on success, -1 on error.
int
either_copyout(int user_dst, uint64 dst, void *src, uint64 len)
{
  struct proc *p = myproc();
  if(user_dst){
    return copyout(p->pagetable, dst, src, len);
  } else {
    memmove((char *)dst, src, len);
    return 0;
  }
}

// Copy from either a user address, or kernel address,
// depending on usr_src.
// Returns 0 on success, -1 on error.
int
either_copyin(void *dst, int user_src, uint64 src, uint64 len)
{
  struct proc *p = myproc();
  if(user_src){
    return copyin(p->pagetable, dst, src, len);
  } else {
    memmove(dst, (char*)src, len);
    return 0;
  }
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  struct proc *p;
  char *state;

  printf("\n");
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s", p->pid, state, p->name);
    printf("\n");
  }
}

int 
trace(int mask, int pid){
  struct proc *p;

  p = find_proc(pid);
  if(!p){
    return -1;
  }

  acquire(&p->lock);
  p->trace_mask = mask;
  release(&p->lock);
  return 0;
}

void
update_pref_stats() {
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    int *p_time_stat = 0;
    switch (p->state)
    {
      case SLEEPING:
        p_time_stat = &p->perf_stats.stime;
        break;
      case RUNNABLE:
        p_time_stat = &p->perf_stats.retime;
        break;
      case RUNNING:
        p_time_stat = &p->perf_stats.rutime;
        break;
      default:
        break;
    }
    if (p_time_stat) {
      *p_time_stat += 1;
    }
    release(&p->lock);
  }
}

#ifdef SCHED_CFSD
int
set_priority(struct proc *p, int priority)
{
  if (!(0 <= priority && priority <= 4)) {
    return -1;
  }
  acquire(&p->lock);
  p->priority = priority;
  release(&p->lock);
  return 0;
}
#endif