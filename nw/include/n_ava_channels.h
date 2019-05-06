#ifndef _NW_INCLUDE_N_AVA_CHANNELS_H_
#define _NW_INCLUDE_N_AVA_CHANNELS_H_ 1

#define REP0(X...)
#define REP1(X...) X
#define REP2(X...) REP1(X), X
#define REP3(X...) REP2(X), X
#define REP4(X...) REP2(X), REP2(X)
#define REP5(X...) REP4(X), X
#define REP6(X...) REP5(X), X
#define REP7(X...) REP6(X), X
#define REP8(X...) REP4(X), REP4(X)

#define REP16(X...) REP8(X), REP8(X)
#define REP32(X...) REP16(X), REP16(X)
#define REP64(X...) REP32(X), REP32(X)
#define REP128(X...) REP64(X), REP64(X)
#define REP256(X...) REP128(X), REP128(X)

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

#define REP_N(N, args...) CAT(REP, N)(args)
#define REP_N_CHANNELS(args...) REP_N(N_AVA_CHANNELS, args)

#define N_AVA_CHANNELS 3

#ifdef __cplusplus
extern "C" {
#endif

void set_ava_chan_no(int chan_no);
int get_ava_chan_no(void);

#ifdef __cplusplus
}
#endif

#endif
