#define PROC_ARRAY_QUEUE_CAPACITY NPROC

struct proc_array_queue {
  struct proc *array[PROC_ARRAY_QUEUE_CAPACITY];
  struct spinlock lock;
  int base_index;
  int size;
};

void            proc_array_queue_init(struct proc_array_queue *queue, char *lock_name);
int             proc_array_queue_enqueue(struct proc_array_queue *queue, struct proc *proc);
struct proc*    proc_array_queue_dequeue(struct proc_array_queue *queue);