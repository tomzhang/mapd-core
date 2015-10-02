// VERTEX SHADER

#version 410 core

#ifndef inTx
#define inTx float
#endif

#ifndef outTx
#define outTx float
#endif

#ifndef inTy
#define inTy float
#endif

#ifndef outTy
#define outTy float
#endif

#ifndef inTz
#define inTz float
#endif

#ifndef outTz
#define outTz float
#endif

#ifndef inTsize
#define inTsize float
#endif

#ifndef outTsize
#define outTsize float
#endif

#ifndef inTcolor
#define inTcolor vec4
#endif

#ifndef outTcolor
#define outTcolor vec4
#endif


in inTx x;
in inTy y;
in inTz z;

in inTsize size;
in inTcolor color;

in uint id;


outTx getx() {
    return x;
}

outTy gety() {
    return y;
}

outTz getz() {
    return z;
}

outTsize getsize() {
    return size;
}

outTcolor getcolor() {
    return color;
}


////////////////////////////////////////////////////////////////
/**
 * Non-interpolated shader outputs.
 */
flat out uint fPrimitiveId;  // the id of the primitive
flat out vec4 fColor;        // the output color of the primitive

void main() {
    // gl_Position = vec4(x * x_scale + x_offset, y*y_scale + y_offset, 0.0, 1.0);
    // gl_PointSize = getSize(float(size));

    gl_Position = vec4(float(getx()), float(gety()), float(getz()), 1.0);
    gl_PointSize = float(getsize());

    fColor = getcolor();
    fPrimitiveId = id;
}
