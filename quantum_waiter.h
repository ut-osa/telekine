#ifndef _QUANTUM_WAITER_H_
#define _QUANTUM_WAITER_H_

#include <sys/timerfd.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>

class QuantumWaiter {
public:
    QuantumWaiter(int interval_us) : timer_fd_(-1), interval_us_(interval_us) {}
    
    void WaitNextQuantum() {
        if (timer_fd_ == -1) {
            struct itimerspec new_value;
            if (clock_gettime(CLOCK_REALTIME, &new_value.it_value) != 0) {
                fprintf(stderr, "clock_gettime failed\n");
                exit(1);
            }
            new_value.it_interval.tv_sec = interval_us_ / 1000000;
            new_value.it_interval.tv_nsec = interval_us_ % 1000000 * 1000;
            timer_fd_ = timerfd_create(CLOCK_REALTIME, 0);
            if (timer_fd_ == -1) {
                fprintf(stderr, "timerfd_create failed\n");
                exit(1);
            }
            if (timerfd_settime(timer_fd_, TFD_TIMER_ABSTIME, &new_value, NULL) != 0) {
                fprintf(stderr, "timerfd_settime failed\n");
                exit(1);
            }
        }
        uint64_t val;
        if (read(timer_fd_, &val, sizeof(uint64_t)) != sizeof(uint64_t)) {
            fprintf(stderr, "Failed to read from timer\n");
            exit(1);
        }
        if (val > 1) {
            fprintf(stderr, "[QuantumWaiter interval=%d] miss %d quanta\n",
                    interval_us_, (int)(val-1));
        }
    }

private:
    int timer_fd_;
    int interval_us_;
};

#endif
