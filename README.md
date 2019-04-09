=====
Regenerating AVA files
=====

1. modify hip.nw.cpp with your desired changes
2. run "make regen" to regenerate ava files
3. compare the files in ./hip_nw/ to the files in ./
4. manually copy the bits out of ./hip_nw/*.* that are new
   - pay close attention to the "command_channel_free_command" calls and the
     "pthread_once(&guestlib_init, init_hip_guestlib)" calls which need to be
     maunually applied to any source code generated since ava doesn't generate
     these calls
