// VERTEX SHADER

#version 410 core

#define useUx <useUx>
#define inTx <inTxType>
#define outTx <outTxType>

#define useUy <useUy>
#define inTy <inTyType>
#define outTy <outTyType>

// #define useUz <useUz>
// #define inTz float
// #define outTz float

#define useUsize <useUsize>
#define inTsize <inTsizeType>
#define outTsize <outTsizeType>

#define useUfillColor <useUfillColor>
#define inTfillColor <inTfillColorType>
#define outTfillColor <outTfillColorType>

#define useUid <useUid>

#define useKey <useKey>
#define useUkey <useUkey>

#if useUx == 1
uniform inTx x;
#else
in inTx x;
#endif

#if useUy == 1
uniform inTy y;
#else
in inTy y;
#endif

// #if useUz == 1
// uniform inTz z;
// #else
// in inTz z;
// #endif

#if useUsize == 1
uniform inTsize size;
#else
in inTsize size;
#endif

#if useUfillColor == 1
uniform inTfillColor fillColor;
#else
in inTfillColor fillColor;
#endif

#if useUid == 1
uniform uint id;
#else
in uint id;
#endif

#if useKey == 1
// TODO(croot): use the NV_vertex_attrib_integer_64bit extension to use an int64 for the key and invalidKey.
#if useUkey == 1
uniform int key;
#else
in int key;
#endif
uniform int invalidKey;
#endif

outTx getx(in inTx x) {
  return x;
}

outTy gety(in inTy y) {
  return y;
}

// outTz getz(in inTz z) {
//     return z;
// }

outTsize getsize(in inTsize size) {
  return size;
}

outTfillColor getfillColor(in inTfillColor fillColor) {
  return fillColor;
}

////////////////////////////////////////////////////////////////
/**
 * Non-interpolated shader outputs.
 */
flat out uint fPrimitiveId;  // the id of the primitive
flat out vec4 fColor;        // the output color of the primitive
flat out float fPointSize;   // the point size of the vertex

void main() {
#if useKey == 1
  if (key != invalidKey) {
    gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);
    float sz = getsize(size);
    fPointSize = sz;
    gl_PointSize = sz;

    fColor = getfillColor(fillColor);
  } else {
    gl_Position = vec4(0, 0, 0, 0);
    fPointSize = 0.0;
    gl_PointSize = 0.0;
    fColor = vec4(0, 0, 0, 0);
  }
#else
  gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);
  float sz = getsize(size);
  fPointSize = sz;
  gl_PointSize = sz;

  fColor = getfillColor(fillColor);
#endif

  // ids from queries go from 0 to numrows-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  fPrimitiveId = id + 1;
}
