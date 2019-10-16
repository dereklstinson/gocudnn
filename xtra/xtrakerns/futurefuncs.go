package xtrakerns

/*
const futurefuncs = `
__device__ __half2  h2agtb(__half2 a, __half2 b, __half gtval, __half leval ){
    if (__hbgt2(a,b)){
        return __halves2half2(gtval,gtval);
    }
        return __halves2half2(__hgt(__low2half(a),__low2half(b)) ? gtval : leval,
                              __hgt(__high2half(a),__high2half(b)) ? gtval : leval);

 }
__device__ __half2  h2ageb(__half2 a, __half2 b, __half geval, __half ltval ){

    if (__hbge2(a,b)){
    return __halves2half2(geval,geval);
    }
    return __halves2half2(__hge(__low2half(a),__low2half(b)) ? geval : ltval,
                          __hge(__high2half(a),__high2half(b)) ? geval : ltval);

}
__device__ __half2  h2altb(__half2 a, __half2 b, __half geval, __half ltval ){
    if (__hblt2(a,b)){
    return __halves2half2(ltval,ltval);
    }
    return __halves2half2(__hlt(__low2half(a),__low2half(b)) ?ltval: geval,
                          __hlt(__high2half(a),__high2half(b)) ?ltval: geval);
  }
__device__ __half2  h2aleb(__half2 a, __half2 b, __half gtval, __half leval ){
    if (__hble2(a,b)){
    return __halves2half2(leval,leval);
    }
    return __halves2half2(__hle(__low2half(a),__low2half(b)) ?leval: gtval,
                          __hle(__high2half(a),__high2half(b)) ?leval: gtval);
}`
*/
