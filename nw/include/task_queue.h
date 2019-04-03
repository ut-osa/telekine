#ifndef __VGPU_TASK_QUEUE_H__
#define __VGPU_TASK_QUEUE_H__


#ifndef __KERNEL__
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <sys/stat.h>
#endif

#include "devconf.h"
#include "ctype_util.h"
#include "socket.h"
#include "cmd_channel_impl.h"


typedef struct _MINI_TASK_NODE {
    int       vm_id;
    uint8_t   rt_type;
    uintptr_t data_ptr; /* offset in fifo for param */

    struct command_base *cmd_in;

    long      node_id;
    BOOLEAN   IsSwap;
    BOOLEAN   IsHighPriority;

} MiniTaskNode, MINI_TASK_NODE, *PMINI_TASK_NODE, TASK_NODE, *PTASK_NODE;


#ifndef __KERNEL__

static inline const char *task_queue_sem_name(const char *prefix, const char *suffix)
{
    static char sem_name[32];
    sprintf(sem_name, "%s_taskq_%s", prefix, suffix);
    return sem_name;
}


typedef struct _TASK_QUEUE_SEM
{
    sem_t *start;
    sem_t *done;
} TaskQueueSem, TASK_QUEUE_SEM, *PTASK_QUEUE_SEM;


typedef struct _TASK_QUEUE
{
    size_t size;                                /* EXEC_TASK_QUEUE_SIZE */
    MINI_TASK_NODE tasks[EXEC_TASK_QUEUE_SIZE]; /* offset in FIFO */
    size_t   head;                              /* proceed */
    size_t   tail;                              /* insert */

    pthread_mutex_t push_lock;
    TASK_QUEUE_SEM sem;

    //
    // Used to serialize the invocations.
    //
    TASK_QUEUE_SEM Serial;

} TaskQueue, TASK_QUEUE, *PTASK_QUEUE;


static inline void init_task_queue_sem(PTASK_QUEUE_SEM s)
{
    s->start = (sem_t *)malloc(sizeof(sem_t));
    sem_init(s->start, 0, 0);
    s->done = (sem_t *)malloc(sizeof(sem_t));
    sem_init(s->done, 0, EXEC_TASK_QUEUE_SIZE);
}

static inline void free_task_queue_sem(TaskQueueSem *s)
{
    sem_destroy(s->start);
    free(s->start);
    sem_destroy(s->done);
    free(s->done);
}

static inline void init_task_queue(TaskQueue *q, size_t size)
{
    int i;

    q->size = size;
    q->head = q->tail = 0;
    for (i = 0; i < size - 1; ++i) {
        q->tasks[i].vm_id = -1;
        q->tasks[i].node_id = -1;
        q->tasks[i].data_ptr = 0;
    }
    pthread_mutex_init(&q->push_lock, NULL);
}

#endif

#endif
