#ifndef _TLKINE_CONFIG_
#define _TLKINE_CONFIG_

#include "command_scheduler.h"
#include "check_env.h"

namespace tlkine {

struct global_config {
   SchedulerType sched_type;
   bool check_batch_complete;
   int batch_size;
   int fixed_rate_interval_us;
   int memcpy_fixed_rate_interval_us;
   int memcpy_n_staging_buffers;
private:
   global_config() {
      char *s;

      if (CHECK_ENV("HIP_ENABLE_SEP_MEMCPY_COMMAND_SCHEDULER", false)) {
         if (CHECK_ENV("LGM_MEMCPY_ENABLE_ENCRYPTION", false))
            sched_type = ENCRYPTED;
         else
            sched_type = MANAGED;
      } else if (CHECK_ENV("HIP_ENABLE_COMMAND_SCHEDULER", false)) {
         sched_type = BATCHED;
      } else {
         sched_type = BASELINE;
      }
      check_batch_complete = CHECK_ENV("HIP_SYNC_CHECK_BATCH_COMPLETE", false);
      batch_size = CHECK_ENV("HIP_COMMAND_SCHEDULER_BATCH_SIZE", DEFAULT_BATCH_SIZE);
      fixed_rate_interval_us = CHECK_ENV("HIP_COMMAND_SCHEDULER_FR_INTERVAL_US", DEFAULT_FIXED_RATE_INTERVAL_US);
      memcpy_fixed_rate_interval_us = CHECK_ENV("HIP_COMMAND_SCHEDULER_MEMCPY_FR_INTERVAL_US", DEFAULT_FIXED_RATE_INTERVAL_US);
      memcpy_n_staging_buffers = CHECK_ENV("HIP_MEMCPY_N_STAGING_BUFFERS", DEFAULT_N_STAGING_BUFFERS);
   };
   global_config(const global_config &) = delete;
   friend global_config &__config();
};

extern global_config &config;

}

#endif
